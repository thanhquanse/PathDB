package com.gdblab.execution;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.gdblab.algebra.queryplan.logical.BinaryLogicalOperator;
import com.gdblab.algebra.queryplan.logical.LogicalOperator;
import com.gdblab.algebra.queryplan.logical.NullaryLogicalOperator;
import com.gdblab.algebra.queryplan.logical.UnaryLogicalOperator;
import com.gdblab.graph.schema.Edge;
import com.gdblab.graph.schema.GraphObject;
import com.gdblab.graph.schema.Node;
import com.gdblab.graph.schema.Path;
import com.gdblab.privacy.NoiseCache;
import com.gdblab.privacy.SensitivePatternPolicy;

/**
 * Operator-level differential-privacy evaluation for an EDGE-pattern-sensitive query.
 *
 * PUBLIC ENTRY POINTS
 *   run(query, pattern, epsilon, clipBound)                 -> one run, SENSITIVE_ONLY, detailed tables
 *   run(query, pattern, epsilon, clipBound, scope)          -> one run, chosen scope, detailed tables
 *   compare(query, pattern, epsilons, clipBound, trials, seed) -> SENSITIVE_ONLY vs ALL_EDGES error table
 *
 * WHAT IS RELEASED (all Laplace mechanism, noise cached so the same target is never re-noised)
 *   1. Person.birthYear   : clipped to public range [YEAR_MIN, YEAR_MAX], sensitivity = range,
 *                           noise Lap(range/eps), ONE draw per person (cache key = node id).
 *                           A node is protected if it is an endpoint of ANY protected edge anywhere in
 *                           the plan, so every released copy carries the same noisy year (no leak by join).
 *   2. Edge count         : exact count of non-protected edges + noisy count of protected edges (sens. 1).
 *   3. Message.length SUM : exact sum of non-sensitive messages + clipped sum of sensitive ones + one
 *                           Lap(clipBound/eps) draw. AVG = released SUM / (public message count).
 *
 * SCOPE
 *   SENSITIVE_ONLY : protected = edges where pattern.isSensitiveEdge is true (messages: isSensitivePath).
 *   ALL_EDGES      : every edge (and every message) is protected.
 *
 * UTILITY (vs. ground truth, which is the ORIGINAL unclipped value)
 *   MAE, RMSE in original units; MRE with sanity bound s=1 for counts and sums (not meaningful for years).
 *
 * Post-processing (rounding, clamping, dividing for AVG) costs no additional privacy budget.
 */
public final class Evaluator {

    public enum Scope { SENSITIVE_ONLY, ALL_EDGES }

    // ---- public, data-independent bounds for birthYear ----
    private static final String YEAR_PROP = "birthYear";
    private static final int YEAR_MIN = 1976;
    private static final int YEAR_MAX = 2000;
    private static final double YEAR_SENSITIVITY = YEAR_MAX - YEAR_MIN;

    // =====================================================================================
    //  Result rows
    // =====================================================================================
    private static final class OpRow {
        String label;
        int depth;
        int totalPaths;

        // edge-count aggregate
        int uniqueEdges;
        long sensitiveEdgeCount;       // number of PROTECTED edges here (per scope)
        long nonSensitiveEdgeCount;
        double releasedEdgeCount;
        boolean edgeNoiseApplied;
        boolean edgeNoiseIsNew;

        // value (SUM/AVG) aggregate
        boolean hasTargetValue;
        int sensitiveMessages;
        int nonSensitiveMessages;
        double trueSensitiveSum;
        double trueNonSensitiveSum;
        double releasedSum;
        double releasedAvg;
        boolean valueNoiseApplied;
        boolean valueNoiseIsNew;
    }

    public record Metrics(int n, double mae, double rmse, double mre) {}

    /** Everything produced by one evaluation pass (one eps, one scope, one noise cache). */
    private static final class Result {
        final List<OpRow> rows = new ArrayList<>();
        // nodeId -> {truth, released, protectedFlag(1/0)}
        final Map<String, double[]> years = new LinkedHashMap<>();
        final List<double[]> counts = new ArrayList<>();  // {truthTotalEdges, releasedTotalEdges}
        final List<double[]> sums = new ArrayList<>();    // {truthSum, releasedSum}
        final Map<String, Node> releasedNodes = new HashMap<>();
        final List<Edge> rootReleasedEdges = new ArrayList<>();

        Metrics yearProtected() {
            return metrics(years.values().stream().filter(a -> a[2] == 1.0).toList(), false);
        }
        Metrics yearAll() { return metrics(years.values(), false); }
        Metrics countAll() { return metrics(counts, true); }
        Metrics sumAll() { return metrics(sums, true); }
        double protectedShare() {
            return years.isEmpty() ? 0.0
                    : years.values().stream().filter(a -> a[2] == 1.0).count() / (double) years.size();
        }
    }

    // =====================================================================================
    //  Materialized plan (query is executed ONCE, evaluation afterwards is in-memory)
    // =====================================================================================
    private static final class OpData {
        String label;
        int depth;
        int totalPaths;
        final List<Edge> edges = new ArrayList<>();                       // unique edges at this operator
        final Map<String, Double> msgValue = new LinkedHashMap<>();       // message id -> length
        final Set<String> sensitiveMsgIds = new HashSet<>();              // reached via a sensitive path
    }

    private static final class Prepared {
        final List<OpData> ops = new ArrayList<>();                       // children first, root LAST
        final Map<String, Boolean> edgeSensitive = new HashMap<>();       // edge id -> isSensitiveEdge
    }

    // =====================================================================================
    //  main (example)
    // =====================================================================================
    public static void main(String[] args) throws Exception {
        Tools.loadDefaultGraph();
        SensitivePatternPolicy pattern = new SensitivePatternPolicy();
        pattern.addSensitiveNodeName("President");
        String query = "MATCH p = (x)-[knows+.posted]->(msg) RETURN x.name, msg.length;";

        // single detailed run
        run(query, pattern, 1.0, 600.0);

        // graph is reset after each public call, so load again for the second call
        Tools.loadDefaultGraph();
        compare(query, pattern, new double[]{0.1, 0.5, 1, 2, 5, 10}, 600.0, 50, 42L);
    }

    // =====================================================================================
    //  PUBLIC API
    // =====================================================================================
    public static void run(final String query, final SensitivePatternPolicy pattern,
                           final double epsilon, final double clipBound) throws Exception {
        run(query, pattern, epsilon, clipBound, Scope.SENSITIVE_ONLY);
    }

    public static void run(final String query, final SensitivePatternPolicy pattern,
                           final double epsilon, final double clipBound, final Scope scope) throws Exception {
        System.out.println("=== Operator-level DP evaluation ===");
        System.out.println("query: " + query);
        System.out.println("scope: " + scope + "   epsilon=" + epsilon + "   clipBound=" + clipBound);
        System.out.println("sensitive edges: label in " + pattern.getSensitiveEdgeLabels()
                + ", or touching a node with id in " + pattern.getSensitiveNodeIds()
                + ", name in " + pattern.getSensitiveNodeNames()
                + ", or status matching " + pattern.getSensitiveStatusValues() + "\n");

        Prepared prep = prepare(query, pattern);

        NoiseCache noiseCache = new NoiseCache(); // ONE cache for the whole run, shared across operators
        Result res = evaluate(prep, epsilon, clipBound, scope, noiseCache);
        List<OpRow> rows = res.rows;

        System.out.println("--- EDGE COUNT, per operator (protected edges are noised; noise cached by edge-set) ---");
        System.out.printf("%-3s %-26s %7s %8s %10s %10s %12s %-8s%n",
                "lvl", "operator", "paths", "edges", "protected", "non-prot", "released", "noise");
        for (OpRow r : rows) {
            System.out.printf("%-3d %-26s %7d %8d %10d %10d %12.2f %-8s%n",
                    r.depth, r.label, r.totalPaths, r.uniqueEdges, r.sensitiveEdgeCount, r.nonSensitiveEdgeCount,
                    r.releasedEdgeCount, !r.edgeNoiseApplied ? "NONE" : (r.edgeNoiseIsNew ? "NEW" : "REUSED"));
        }
        System.out.println();

        boolean anyValueReached = rows.stream().anyMatch(r -> r.hasTargetValue);
        if (anyValueReached) {
            System.out.println("--- Message.length SUM/AVG, per operator (noise cached by sensitive-message-set) ---");
            System.out.printf("%-3s %-26s %10s %10s %14s %16s %12s %10s %-8s%n",
                    "lvl", "operator", "sens-msgs", "nonsens", "true-sens-sum", "true-nonsens-sum",
                    "released-sum", "avg", "noise");
            for (OpRow r : rows) {
                if (r.hasTargetValue) {
                    System.out.printf("%-3d %-26s %10d %10d %14.1f %16.1f %12.2f %10.2f %-8s%n",
                            r.depth, r.label, r.sensitiveMessages, r.nonSensitiveMessages,
                            r.trueSensitiveSum, r.trueNonSensitiveSum, r.releasedSum, r.releasedAvg,
                            !r.valueNoiseApplied ? "NONE" : (r.valueNoiseIsNew ? "NEW" : "REUSED"));
                }
            }
            System.out.println();
        } else {
            System.out.println("--- Message.length SUM/AVG: this query never reaches a Message node ---\n");
        }

        // ---- edge-by-edge verdicts and released birthYear at the root operator ----
        OpData rootData = prep.ops.get(prep.ops.size() - 1);
        System.out.println("--- edge verdicts + released birthYear at the FINAL (root) operator (capped at 30) ---");
        System.out.printf("%-6s %-34s %-14s %-22s %-22s%n", "id", "edge", "verdict", "source year true->rel", "target year true->rel");
        int shown = 0;
        for (Edge e : rootData.edges) {
            if (shown++ >= 30) {
                System.out.println("  ... (" + (rootData.edges.size() - 30) + " more, truncated for display)");
                break;
            }
            boolean sens = prep.edgeSensitive.get(e.getId());
            System.out.printf("%-6s (%s:%s)-[%s]->(%s:%s) %-14s %-22s %-22s%n",
                    e.getId(), e.getSource().getId(), safeName(e.getSource()), e.getLabel(),
                    e.getTarget().getId(), safeName(e.getTarget()), sens ? "SENSITIVE" : "not sensitive",
                    yearPair(e.getSource(), res), yearPair(e.getTarget(), res));
        }

        OpRow rootRow = rows.get(rows.size() - 1);
        System.out.printf("%n=== Real releases (root operator), eps=%.2f each, scope=%s ===%n", epsilon, scope);
        System.out.printf("Edge count:          released=%.2f (true=%d)%n",
                rootRow.releasedEdgeCount, rootRow.sensitiveEdgeCount + rootRow.nonSensitiveEdgeCount);
        if (rootRow.hasTargetValue) {
            System.out.printf("Message length SUM:  released=%.2f (true=%.1f), AVG=%.2f%n",
                    rootRow.releasedSum, rootRow.trueSensitiveSum + rootRow.trueNonSensitiveSum, rootRow.releasedAvg);
        }

        System.out.println("\n--- UTILITY vs ground truth ---");
        printMetrics("birthYear (protected persons)", res.yearProtected(), false);
        printMetrics("birthYear (all persons)      ", res.yearAll(), false);
        printMetrics("edge count (per operator)    ", res.countAll(), true);
        if (anyValueReached) printMetrics("message SUM (per operator)   ", res.sumAll(), true);
        System.out.printf("protected persons: %.1f%% of released persons%n", 100 * res.protectedShare());
        System.out.printf("Total distinct noise draws this run (years + counts + sums): %d%n%n", noiseCache.size());
    }

    /**
     * Repeats the evaluation for each epsilon, for BOTH scopes, and prints the mean (+/- std) error
     * against ground truth. Same seeds are used for both scopes.
     */
    public static void compare(final String query, final SensitivePatternPolicy pattern,
                               final double[] epsilons, final double clipBound,
                               final int trials, final long baseSeed) throws Exception {
        System.out.println("=== SENSITIVE_ONLY vs ALL_EDGES  (" + trials + " trials per cell, mean +/- std) ===");
        System.out.println("query: " + query + "\n");

        Prepared prep = prepare(query, pattern);
        boolean hasValues = prep.ops.stream().anyMatch(o -> !o.msgValue.isEmpty());

        System.out.printf("%-15s %5s %6s | %-15s %-15s %-15s | %-12s %-8s | %-12s %-8s%n",
                "scope", "eps", "prot%", "yearMAE(prot)", "yearMAE(all)", "yearRMSE(all)",
                "countMAE", "countMRE", "sumMAE", "sumMRE");

        for (double eps : epsilons) {
            for (Scope scope : Scope.values()) {
                double[] share = new double[trials], ymp = new double[trials], yma = new double[trials],
                        yra = new double[trials], cma = new double[trials], cre = new double[trials],
                        sma = new double[trials], sre = new double[trials];
                for (int t = 0; t < trials; t++) {
                    NoiseCache cache = new NoiseCache(baseSeed + t);   // fresh noise per trial
                    Result r = evaluate(prep, eps, clipBound, scope, cache);
                    share[t] = r.protectedShare();
                    ymp[t] = r.yearProtected().mae();
                    yma[t] = r.yearAll().mae();
                    yra[t] = r.yearAll().rmse();
                    cma[t] = r.countAll().mae();
                    cre[t] = r.countAll().mre();
                    sma[t] = r.sumAll().mae();
                    sre[t] = r.sumAll().mre();
                }
                System.out.printf("%-15s %5.1f %5.1f%% | %6.2f±%-7.2f %6.2f±%-7.2f %6.2f±%-7.2f | %-12.2f %-8.3f | %-12s %-8s%n",
                        scope, eps, 100 * mean(share),
                        mean(ymp), std(ymp), mean(yma), std(yma), mean(yra), std(yra),
                        mean(cma), mean(cre),
                        hasValues ? String.format("%.2f", mean(sma)) : "n/a",
                        hasValues ? String.format("%.3f", mean(sre)) : "n/a");
            }
        }
        System.out.println();
        System.out.println("Reading the table: yearMAE(prot) should match across scopes (same mechanism);");
        System.out.println("yearMAE(all) ~ yearMAE(prot) * protected-share, so SENSITIVE_ONLY is lower, but it protects less.\n");
    }

    // =====================================================================================
    //  Step 1: run the query ONCE and snapshot every operator
    // =====================================================================================
    private static Prepared prepare(final String query, final SensitivePatternPolicy pattern) throws Exception {
        try {
            LogicalOperator root = IntermediateResultsExplainer.parseToLogicalRoot(query);
            Prepared prep = new Prepared();
            collect(root, 0, pattern, prep);
            return prep;
        } finally {
            Tools.resetContext(); // everything needed is now in memory
        }
    }

    private static void collect(final LogicalOperator node, final int depth,
                                final SensitivePatternPolicy pattern, final Prepared prep) {
        if (node instanceof UnaryLogicalOperator u) {
            collect(u.getChild(), depth + 1, pattern, prep);
        } else if (node instanceof BinaryLogicalOperator b) {
            collect(b.getLeftChild(), depth + 1, pattern, prep);
            collect(b.getRightChild(), depth + 1, pattern, prep);
        } else if (node instanceof NullaryLogicalOperator) {
            // leaf
        }

        List<Path> paths = IntermediateResultsExplainer.materialize(node);
        OpData op = new OpData();
        op.label = node.getClass().getSimpleName();
        op.depth = depth;
        op.totalPaths = paths.size();

        Map<String, Edge> unique = new LinkedHashMap<>();
        for (Path p : paths) {
            for (GraphObject go : p.getSequence()) {
                if (go instanceof Edge e) unique.putIfAbsent(e.getId(), e);
            }
            // message-length contributor: last node of the path, if it has a "length" property
            Node last = p.last();
            String v = (last != null && last.getProperties() != null) ? last.getProperties().get("length") : null;
            if (v != null && !v.isEmpty()) {
                try {
                    op.msgValue.putIfAbsent(last.getId(), Double.parseDouble(v.trim()));
                    // a message is sensitive if ANY path reaching it is sensitive (conservative)
                    if (pattern.isSensitivePath(p)) op.sensitiveMsgIds.add(last.getId());
                } catch (NumberFormatException ignore) { /* skip malformed value */ }
            }
        }
        op.edges.addAll(unique.values());
        for (Edge e : op.edges) {
            prep.edgeSensitive.computeIfAbsent(e.getId(), k -> pattern.isSensitiveEdge(e));
        }
        prep.ops.add(op); // post-order => root is last
    }

    // =====================================================================================
    //  Step 2: one DP evaluation pass (in memory)
    // =====================================================================================
    private static Result evaluate(final Prepared prep, final double eps, final double clipBound,
                                   final Scope scope, final NoiseCache cache) {
        Result res = new Result();

        // ---- (a) GLOBAL protected-node set: endpoint of ANY protected edge in ANY operator ----
        Set<String> protectedNodes = new HashSet<>();
        for (OpData op : prep.ops) {
            for (Edge e : op.edges) {
                if (isProtectedEdge(e, prep, scope)) {
                    protectedNodes.add(e.getSource().getId());
                    protectedNodes.add(e.getTarget().getId());
                }
            }
        }

        // ---- (b) release each person ONCE (cached noise); unprotected persons are released as-is ----
        for (OpData op : prep.ops) {
            for (Edge e : op.edges) {
                for (Node n : new Node[]{e.getSource(), e.getTarget()}) {
                    if (!res.releasedNodes.containsKey(n.getId())) {
                        res.releasedNodes.put(n.getId(),
                                releaseNode(n, protectedNodes.contains(n.getId()), eps, cache, res));
                    }
                }
            }
        }

        // ---- (c) per-operator aggregates ----
        final double countScale = 1.0 / eps;           // count query, sensitivity 1
        final double sumScale = clipBound / eps;       // clipped sum, add/remove sensitivity = clipBound
        for (int i = 0; i < prep.ops.size(); i++) {
            OpData op = prep.ops.get(i);
            OpRow r = new OpRow();
            r.label = op.label;
            r.depth = op.depth;
            r.totalPaths = op.totalPaths;
            r.uniqueEdges = op.edges.size();

            // EDGE COUNT
            List<String> protIds = new ArrayList<>();
            for (Edge e : op.edges) if (isProtectedEdge(e, prep, scope)) protIds.add(e.getId());
            r.sensitiveEdgeCount = protIds.size();
            r.nonSensitiveEdgeCount = op.edges.size() - protIds.size();
            long noisyProtected = 0;
            if (!protIds.isEmpty()) {
                String key = canonicalKey("edgecount", protIds) + "@" + countScale;
                r.edgeNoiseApplied = true;
                r.edgeNoiseIsNew = !cache.isCached(key);
                double noise = cache.laplaceNoiseFor(key, countScale);
                noisyProtected = Math.max(0, Math.round(protIds.size() + noise)); // post-processing
            }
            r.releasedEdgeCount = r.nonSensitiveEdgeCount + noisyProtected;
            res.counts.add(new double[]{op.edges.size(), r.releasedEdgeCount});

            // MESSAGE SUM / AVG
            if (!op.msgValue.isEmpty()) {
                r.hasTargetValue = true;
                List<String> sensIds = new ArrayList<>();
                double sensClipped = 0;
                for (Map.Entry<String, Double> en : op.msgValue.entrySet()) {
                    boolean sens = scope == Scope.ALL_EDGES || op.sensitiveMsgIds.contains(en.getKey());
                    if (sens) {
                        sensIds.add(en.getKey());
                        r.trueSensitiveSum += en.getValue();
                        sensClipped += clip(en.getValue(), clipBound);
                    } else {
                        r.trueNonSensitiveSum += en.getValue();
                        r.nonSensitiveMessages++;
                    }
                }
                r.sensitiveMessages = sensIds.size();
                double noise = 0;
                if (!sensIds.isEmpty()) {
                    String key = canonicalKey("msgsum", sensIds) + "@" + sumScale;
                    r.valueNoiseApplied = true;
                    r.valueNoiseIsNew = !cache.isCached(key);
                    noise = cache.laplaceNoiseFor(key, sumScale);
                }
                r.releasedSum = r.trueNonSensitiveSum + sensClipped + noise;
                int totalMsgs = op.msgValue.size();          // message count treated as public
                r.releasedAvg = totalMsgs == 0 ? 0 : r.releasedSum / totalMsgs;
                res.sums.add(new double[]{r.trueSensitiveSum + r.trueNonSensitiveSum, r.releasedSum});
            }
            res.rows.add(r);

            // released edge list for the root operator (consistent noisy endpoints)
            if (i == prep.ops.size() - 1) {
                for (Edge e : op.edges) {
                    Node s = res.releasedNodes.get(e.getSource().getId());
                    Node t = res.releasedNodes.get(e.getTarget().getId());
                    HashMap<String, String> p = e.getProperties() == null ? null : new HashMap<>(e.getProperties());
                    res.rootReleasedEdges.add(new Edge(e.getId(), e.getLabel(), s, t, p));
                }
            }
        }
        return res;
    }

    private static boolean isProtectedEdge(final Edge e, final Prepared prep, final Scope scope) {
        return scope == Scope.ALL_EDGES || Boolean.TRUE.equals(prep.edgeSensitive.get(e.getId()));
    }

    // =====================================================================================
    //  birthYear release (Laplace mechanism with clipping; one cached draw per person)
    // =====================================================================================
    private static Node releaseNode(final Node n, final boolean protect, final double eps,
                                    final NoiseCache cache, final Result res) {
        HashMap<String, String> props = n.getProperties();
        String raw = (props == null) ? null : props.get(YEAR_PROP);
        if (raw == null || raw.isBlank()) return n;

        int trueYear;
        try {
            trueYear = Integer.parseInt(raw.trim());
        } catch (NumberFormatException ex) {
            return n;
        }

        if (!protect) {
            res.years.putIfAbsent(n.getId(), new double[]{trueYear, trueYear, 0});
            return n;
        }

        double scale = YEAR_SENSITIVITY / eps;
        String key = "birthYear:" + n.getId() + "@" + scale;       // one draw per person (and per scale)
        double noise = cache.laplaceNoiseFor(key, scale);          // NEW or REUSED: same value either way
        int noisyYear = clampYear(Math.round(clampYear(trueYear) + noise)); // post-processing: free
        res.years.put(n.getId(), new double[]{trueYear, noisyYear, 1});

        HashMap<String, String> newProps = new HashMap<>(props);   // copy; never mutate the shared original
        newProps.put(YEAR_PROP, String.valueOf(noisyYear));
        return new Node(n.getId(), n.getLabel(), newProps);
    }

    private static int clampYear(final long y) {
        return (int) Math.min(Math.max(y, YEAR_MIN), YEAR_MAX);
    }

    // =====================================================================================
    //  Metrics and helpers
    // =====================================================================================
    private static Metrics metrics(final Collection<double[]> pairs, final boolean withMre) {
        int n = pairs.size();
        if (n == 0) return new Metrics(0, Double.NaN, Double.NaN, Double.NaN);
        double abs = 0, sq = 0, rel = 0;
        for (double[] p : pairs) {
            double err = Math.abs(p[1] - p[0]);
            abs += err;
            sq += err * err;
            if (withMre) rel += err / Math.max(Math.abs(p[0]), 1.0); // sanity bound s = 1
        }
        return new Metrics(n, abs / n, Math.sqrt(sq / n), withMre ? rel / n : Double.NaN);
    }

    private static void printMetrics(final String name, final Metrics m, final boolean mre) {
        if (m.n() == 0) {
            System.out.printf("%-30s n=0%n", name);
        } else if (mre) {
            System.out.printf("%-30s n=%-4d MAE=%-9.3f RMSE=%-9.3f MRE=%.4f%n", name, m.n(), m.mae(), m.rmse(), m.mre());
        } else {
            System.out.printf("%-30s n=%-4d MAE=%-9.3f RMSE=%-9.3f%n", name, m.n(), m.mae(), m.rmse());
        }
    }

    private static String yearPair(final Node n, final Result res) {
        String raw = (n.getProperties() != null) ? n.getProperties().get(YEAR_PROP) : null;
        if (raw == null) return "-";
        Node rel = res.releasedNodes.get(n.getId());
        String relYear = (rel != null && rel.getProperties() != null) ? rel.getProperties().get(YEAR_PROP) : "?";
        return raw + "->" + relYear;
    }

    private static String safeName(final Node n) {
        String name = (n.getProperties() != null) ? n.getProperties().get("name") : null;
        return name != null ? name : "";
    }

    private static String canonicalKey(final String prefix, final Collection<String> ids) {
        return prefix + ":" + ids.stream().sorted().collect(Collectors.joining(","));
    }

    private static double clip(final double v, final double bound) {
        return Math.max(-bound, Math.min(bound, v));
    }

    private static double mean(final double[] a) {
        double s = 0;
        int n = 0;
        for (double v : a) if (!Double.isNaN(v)) { s += v; n++; }
        return n == 0 ? Double.NaN : s / n;
    }

    private static double std(final double[] a) {
        double m = mean(a);
        double s = 0;
        int n = 0;
        for (double v : a) if (!Double.isNaN(v)) { s += (v - m) * (v - m); n++; }
        return n < 2 ? 0.0 : Math.sqrt(s / (n - 1));
    }
}