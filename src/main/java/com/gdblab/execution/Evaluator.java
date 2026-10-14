package com.gdblab.execution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.stream.Collectors;

import com.gdblab.algebra.queryplan.logical.BinaryLogicalOperator;
import com.gdblab.algebra.queryplan.logical.LogicalOperator;
import com.gdblab.algebra.queryplan.logical.NullaryLogicalOperator;
import com.gdblab.algebra.queryplan.logical.UnaryLogicalOperator;
import com.gdblab.graph.schema.Edge;
import com.gdblab.graph.schema.GraphObject;
import com.gdblab.graph.schema.Node;
import com.gdblab.graph.schema.Path;
import com.gdblab.privacy.DPMetrics;
import com.gdblab.privacy.ElasticSensitivity;
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
    private static final int YEAR_MIN = 1980;
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

    /** One distinct noise release: what it spent and which records ("E:edge", "M:message", "P:person") it covers. */
    private record Release(String kind, double eps, Set<String> records) {}

    /** Everything produced by one evaluation pass (one eps, one scope, one noise cache). */
    private static final class Result {
        final List<OpRow> rows = new ArrayList<>();
        // nodeId -> {truth, released, protectedFlag(1/0)}
        final Map<String, double[]> years = new LinkedHashMap<>();
        final List<double[]> counts = new ArrayList<>();  // {truthTotalEdges, releasedTotalEdges}
        final List<double[]> sums = new ArrayList<>();    // {truthSum, releasedSum}
        final Map<String, Node> releasedNodes = new HashMap<>();
        final Map<String, Release> ledger = new LinkedHashMap<>();   // key -> release (reused keys count once)
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
        final Map<String, Integer> labelSeqs = new LinkedHashMap<>();     // "knows,knows,posted" -> #paths
        final List<String[]> pathEdges = new ArrayList<>();               // edge ids of each path
        final List<String> pathMsg = new ArrayList<>();                   // message id reached (or null)
    }

    private static final class Prepared {
        final List<OpData> ops = new ArrayList<>();                       // children first, root LAST
        final Map<String, Boolean> edgeSensitive = new HashMap<>();       // edge id -> isSensitiveEdge
        final Map<String, Edge> allEdges = new LinkedHashMap<>();         // union of edges seen in the plan
    }

    // =====================================================================================
    //  main (example)
    // =====================================================================================
    public static void main(String[] args) throws Exception {
        Tools.loadDefaultGraph();
        SensitivePatternPolicy pattern = new SensitivePatternPolicy();
        pattern.addSensitiveNodeName("President");
        String query = "MATCH p = (x)-[knows+.posted]->(msg) RETURN x.name, msg.length LIMIT 500;";

        // single detailed run
        run(query, pattern, 1.0, 600.0);

        // graph is reset after each public call, so load again for the second call
        Tools.loadDefaultGraph();
        compare(query, pattern, new double[]{0.1, 0.5, 1, 2, 5, 10}, 600.0, 50, 42L);

        // ---- elastic sensitivity for the COUNT of result paths ----
        // The query below is knows+ with at most 4 hops. Replace graphSize with the number of nodes
        // (individuals) in your loaded graph, and add withDegreeBound("knows", B) if you declare a bound.
        String pathQuery = "MATCH ACYCLIC p = (x)-[knows+]{..4}->(y) WHERE x.name = \"Alice\" AND y.name = \"Bob\" "
                + "RETURN p LIMIT 500;";
        ElasticConfig cfg = ElasticConfig.repeat("knows", 1, 4); // .withGraphSize(<#nodes>)
        Tools.loadDefaultGraph();
        runElastic(pathQuery, pattern, 1.0, cfg);
        Tools.loadDefaultGraph();
        compareElastic(pathQuery, pattern, new double[]{0.1, 0.5, 1, 2, 5, 10}, cfg, 50, 42L);

        // ---- THE MAIN EVALUATION: selective operator-level DP vs uniform post-processing vs ground truth ----
        Tools.loadDefaultGraph();
        compareSelectiveVsUniform(query, pattern, new double[]{0.5, 1, 2, 5}, 600.0, 100, 42L, 4000);

        // ---- how the benefit changes with the share of sensitive edges (OSDP-style sweep) ----
        Tools.loadDefaultGraph();
        sweepSensitiveFraction(query, pattern, new double[]{0.01, 0.10, 0.25, 0.50, 0.75, 1.0}, 1.0, 600.0, 100, 42L);
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
            List<String> seq = new ArrayList<>();
            List<String> ids = new ArrayList<>();
            for (GraphObject go : p.getSequence()) {
                if (go instanceof Edge e) {
                    unique.putIfAbsent(e.getId(), e);
                    seq.add(String.valueOf(e.getLabel()));
                    ids.add(e.getId());
                }
            }
            String pathMsgId = null;
            op.labelSeqs.merge(String.join(",", seq), 1, Integer::sum);
            // message-length contributor: last node of the path, if it has a "length" property
            Node last = p.last();
            String v = (last != null && last.getProperties() != null) ? last.getProperties().get("length") : null;
            if (v != null && !v.isEmpty()) {
                try {
                    op.msgValue.putIfAbsent(last.getId(), Double.parseDouble(v.trim()));
                    pathMsgId = last.getId();
                    // a message is sensitive if ANY path reaching it is sensitive (conservative)
                    if (pattern.isSensitivePath(p)) op.sensitiveMsgIds.add(last.getId());
                } catch (NumberFormatException ignore) { /* skip malformed value */ }
            }
            op.pathEdges.add(ids.toArray(new String[0]));   // per-path detail: used by the audit and by reclassify()
            op.pathMsg.add(pathMsgId);
        }
        op.edges.addAll(unique.values());
        for (Edge e : op.edges) {
            prep.allEdges.putIfAbsent(e.getId(), e);
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
                Set<String> recs = new HashSet<>();
                for (String id : protIds) recs.add("E:" + id);
                res.ledger.putIfAbsent(key, new Release("edgecount", eps, recs));
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
                    Set<String> recs = new HashSet<>();
                    for (String id : sensIds) recs.add("M:" + id);
                    res.ledger.putIfAbsent(key, new Release("msgsum", eps, recs));
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
        res.ledger.putIfAbsent(key, new Release("year", eps, Set.of("P:" + n.getId())));

        HashMap<String, String> newProps = new HashMap<>(props);   // copy; never mutate the shared original
        newProps.put(YEAR_PROP, String.valueOf(noisyYear));
        return new Node(n.getId(), n.getLabel(), newProps);
    }

    private static int clampYear(final long y) {
        return (int) Math.min(Math.max(y, YEAR_MIN), YEAR_MAX);
    }

    // =====================================================================================
    //  ELASTIC SENSITIVITY for the COUNT of result paths
    //  (Buil-Aranda, Lobo, Olmedo, Semantic Web 15 (2024); see ElasticSensitivity for the formulas)
    // =====================================================================================

    /**
     * What the query can produce, plus graph-level parameters that must be PUBLIC.
     *  hopPatterns : every edge-label sequence the query may return, e.g. repeat("knows", 1, 4) for knows+{..4}.
     *                Taken from the query text, never from the data.
     *  graphSize   : number of individuals (nodes) in the WHOLE loaded graph = size(G) in the paper.
     *  degreeBound : administrator-declared multiplicity bound kappa(tp) per edge label (max out-degree of one
     *                individual). The paper assumes this is declared, not measured; if absent it is measured.
     */
    public static final class ElasticConfig {
        public final List<List<String>> hopPatterns = new ArrayList<>();
        public final Map<String, Double> degreeBounds = new HashMap<>();
        public long graphSize = 0;
        public double delta = 0;                       // <= 0 -> delta = n^(-ln n), as in the paper's evaluation
        public Collection<Edge> fullGraphEdges = null; // optional: the whole graph, for degree statistics

        /** label{minHops..maxHops}, optionally followed by fixed tail labels, e.g. repeat("knows",1,4,"posted"). */
        public static ElasticConfig repeat(final String label, final int minHops, final int maxHops,
                                           final String... tail) {
            ElasticConfig c = new ElasticConfig();
            for (int h = minHops; h <= maxHops; h++) {
                List<String> seq = new ArrayList<>();
                for (int i = 0; i < h; i++) seq.add(label);
                seq.addAll(Arrays.asList(tail));
                c.hopPatterns.add(seq);
            }
            return c;
        }
        public ElasticConfig withGraphSize(final long n) { this.graphSize = n; return this; }
        public ElasticConfig withDelta(final double d) { this.delta = d; return this; }
        public ElasticConfig withDegreeBound(final String label, final double b) { degreeBounds.put(label, b); return this; }
        public ElasticConfig withFullGraph(final Collection<Edge> edges) { this.fullGraphEdges = edges; return this; }
    }

    /** Max out-/in-degree per edge label, the star (source node label) of each label, number of individuals. */
    private static final class GraphStats {
        final Map<String, Double> maxOut = new HashMap<>();
        final Map<String, Double> maxIn = new HashMap<>();
        final Map<String, String> starOfLabel = new HashMap<>();
        final long individuals;

        GraphStats(final Collection<Edge> edges) {
            Map<String, Map<String, Integer>> out = new HashMap<>();
            Map<String, Map<String, Integer>> in = new HashMap<>();
            Set<String> nodes = new HashSet<>();
            for (Edge e : edges) {
                String l = String.valueOf(e.getLabel());
                String s = e.getSource().getId();
                String t = e.getTarget().getId();
                out.computeIfAbsent(l, x -> new HashMap<>()).merge(s, 1, Integer::sum);
                in.computeIfAbsent(l, x -> new HashMap<>()).merge(t, 1, Integer::sum);
                starOfLabel.putIfAbsent(l, String.valueOf(e.getSource().getLabel()));
                nodes.add(s);
                nodes.add(t);
            }
            out.forEach((l, m) -> maxOut.put(l, (double) m.values().stream().mapToInt(Integer::intValue).max().orElse(0)));
            in.forEach((l, m) -> maxIn.put(l, (double) m.values().stream().mapToInt(Integer::intValue).max().orElse(0)));
            this.individuals = nodes.size();
        }
    }

    private static final class ElasticOutcome {
        double trueCount;
        long n;
        boolean tightKappa;
        GraphStats stats;
        Map<String, Double> kappaStar = new LinkedHashMap<>();
        List<List<String>> labelPatterns = new ArrayList<>();
        List<List<ElasticSensitivity.Hop>> hopPatterns = new ArrayList<>();
        ElasticSensitivity.Smooth smooth;
        final List<String> warnings = new ArrayList<>();
        List<String> rootEdgeIds = new ArrayList<>();
    }

    /**
     * One elastic-sensitivity analysis of COUNT(paths) at the root operator.
     * tightKappa=false : kappa(S) = product of the per-label multiplicities of the star (paper, Definition 9).
     * tightKappa=true  : kappa = multiplicity of the hop's own triple pattern only (tighter; my deviation, flagged).
     */
    private static ElasticOutcome elasticBound(final Prepared prep, final ElasticConfig cfg,
                                               final double eps, final boolean tightKappa) {
        ElasticOutcome o = new ElasticOutcome();
        o.tightKappa = tightKappa;
        OpData root = prep.ops.get(prep.ops.size() - 1);
        o.trueCount = root.totalPaths;
        for (Edge e : root.edges) o.rootEdgeIds.add(e.getId());

        Collection<Edge> source = cfg.fullGraphEdges;
        if (source == null) {
            source = prep.allEdges.values();
            o.warnings.add("degree statistics come from the edges seen in the plan; pass withFullGraph(...) if the "
                    + "leaf operators are filtered, otherwise max-degrees (and the bound) can be UNDER-estimated");
        }
        GraphStats gs = new GraphStats(source);
        o.stats = gs;
        o.n = cfg.graphSize > 0 ? cfg.graphSize : gs.individuals;
        if (cfg.graphSize <= 0) {
            o.warnings.add("graphSize not given: using the " + gs.individuals + " nodes seen in the edges; a smaller "
                    + "size than the real graph shrinks max_k and UNDER-estimates the noise; pass withGraphSize(...)");
        }

        // kappa(S) = product of kappa(tp) over the edge labels of the star
        for (Map.Entry<String, String> en : gs.starOfLabel.entrySet()) {
            double kl = kappaOf(en.getKey(), gs, cfg, o);
            o.kappaStar.merge(en.getValue(), kl, (a, b) -> a * b);
        }

        for (List<String> labels : cfg.hopPatterns) {
            List<ElasticSensitivity.Hop> hops = new ArrayList<>();
            for (String l : labels) {
                double out = gs.maxOut.getOrDefault(l, 0.0);
                double in = gs.maxIn.getOrDefault(l, 0.0);
                String star = gs.starOfLabel.getOrDefault(l, "?" + l);
                double kappa = tightKappa ? kappaOf(l, gs, cfg, o) : o.kappaStar.getOrDefault(star, 1.0);
                hops.add(new ElasticSensitivity.Hop(l, star, out, in, kappa));
            }
            o.labelPatterns.add(labels);
            o.hopPatterns.add(hops);
        }

        Set<String> allowed = new HashSet<>();
        for (List<String> p : cfg.hopPatterns) allowed.add(String.join(",", p));
        for (String observed : root.labelSeqs.keySet()) {
            if (!allowed.contains(observed)) {
                o.warnings.add("root returns label sequence [" + observed + "] that is NOT in hopPatterns: the bound does not cover it");
            }
        }

        double lnTwoOverDelta = cfg.delta > 0 ? ElasticSensitivity.lnTwoOverDelta(cfg.delta)
                                               : ElasticSensitivity.lnTwoOverDeltaPaper(o.n);
        o.smooth = ElasticSensitivity.smooth(o.hopPatterns, o.n, eps, lnTwoOverDelta);
        return o;
    }

    private static double kappaOf(final String label, final GraphStats gs, final ElasticConfig cfg, final ElasticOutcome o) {
        Double declared = cfg.degreeBounds.get(label);
        double measured = gs.maxOut.getOrDefault(label, 1.0);
        if (declared != null) {
            if (measured > declared) {
                String w = "label '" + label + "': measured max out-degree " + measured + " exceeds the declared bound "
                        + declared + " -> the graph does not comply with kappa";
                if (!o.warnings.contains(w)) o.warnings.add(w);
            }
            return Math.max(1.0, declared);
        }
        return Math.max(1.0, measured);
    }

    /** Laplace release of a count with the given noise scale; cached by (root edge set, scale). */
    private static long releaseCount(final ElasticOutcome o, final double scale, final NoiseCache cache) {
        String key = canonicalKey("pathcount", o.rootEdgeIds) + "@" + scale;
        double noise = cache.laplaceNoiseFor(key, scale);
        return Math.max(0, Math.round(o.trueCount + noise)); // rounding + clamping = post-processing
    }

    /** One detailed elastic-sensitivity run: bound, noise scale and a single released count. */
    public static void runElastic(final String query, final SensitivePatternPolicy pattern,
                                  final double epsilon, final ElasticConfig cfg) throws Exception {
        System.out.println("=== ELASTIC SENSITIVITY: COUNT of result paths, (eps, delta)-DP ===");
        System.out.println("query: " + query);
        System.out.println("DP unit: one individual = a node with its outgoing edges and properties; "
                + "neighbour = one individual replaced.\n");

        Prepared prep = prepare(query, pattern);
        OpData root = prep.ops.get(prep.ops.size() - 1);
        System.out.println("root operator label sequences (from the data, for checking only): " + root.labelSeqs);

        for (boolean tight : new boolean[]{false, true}) {
            ElasticOutcome o = elasticBound(prep, cfg, epsilon, tight);
            System.out.printf("%n--- kappa mode: %s ---%n", tight ? "TIGHT (per triple pattern; deviation from paper)"
                    : "PAPER (kappa(S) = product over the star)");
            if (!tight) {
                System.out.println("graph statistics (n = " + o.n + " individuals):");
                System.out.printf("  %-12s %-10s %10s %10s%n", "edge label", "star", "max-out", "max-in");
                for (String l : o.stats.maxOut.keySet()) {
                    System.out.printf("  %-12s %-10s %10.0f %10.0f%n", l, o.stats.starOfLabel.get(l),
                            o.stats.maxOut.get(l), o.stats.maxIn.getOrDefault(l, 0.0));
                }
                System.out.println("  star multiplicity kappa(S): " + o.kappaStar);
            }
            System.out.printf("hop patterns:%n");
            for (int i = 0; i < o.hopPatterns.size(); i++) {
                System.out.printf("  %-40s ES(0) = %s%n", o.labelPatterns.get(i),
                        fmt(ElasticSensitivity.stability(o.hopPatterns.get(i), 0)));
            }
            ElasticSensitivity.Smooth sm = o.smooth;
            System.out.printf("ES(0) total          = %s%n", fmt(sm.es0()));
            System.out.printf("beta                 = %.5f%n", sm.beta());
            System.out.printf("smooth bound U       = %s  (attained at k* = %d)%n", fmt(sm.u()), sm.argK());
            System.out.printf("noise  Lap(2U/eps)   = Lap(%s)%n", fmt(sm.noiseScale()));
            NoiseCache cache = new NoiseCache();
            long released = releaseCount(o, sm.noiseScale(), cache);
            System.out.printf("true count = %s,  released = %d,  noise scale / true count = %s%n",
                    fmt(o.trueCount), released, o.trueCount > 0 ? fmt(sm.noiseScale() / o.trueCount) : "n/a");
            for (String w : o.warnings) System.out.println("  WARNING: " + w);
        }
        System.out.println();
    }

    /**
     * Error vs ground truth of the path COUNT for: elastic sensitivity (both kappa modes) and the naive
     * Lap(1/eps) calibration (sensitivity 1, NOT a valid bound for join outputs, shown as a lower reference).
     */
    public static void compareElastic(final String query, final SensitivePatternPolicy pattern,
                                      final double[] epsilons, final ElasticConfig cfg,
                                      final int trials, final long baseSeed) throws Exception {
        System.out.println("=== ELASTIC SENSITIVITY vs naive Lap(1/eps): COUNT of result paths (" + trials + " trials) ===");
        System.out.println("query: " + query + "\n");
        Prepared prep = prepare(query, pattern);
        List<String> printedWarnings = new ArrayList<>();

        System.out.printf("%-26s %5s %9s %11s %11s %6s %12s | %10s %10s %9s%n",
                "mechanism", "eps", "true", "ES(0)", "U", "k*", "noise scale", "MAE", "RMSE", "MRE");
        for (double eps : epsilons) {
            for (boolean tight : new boolean[]{false, true}) {
                ElasticOutcome o = elasticBound(prep, cfg, eps, tight);
                for (String w : o.warnings) if (!printedWarnings.contains(w)) printedWarnings.add(w);
                printCompareRow(tight ? "elastic (tight kappa)" : "elastic (paper kappa)", eps, o,
                        o.smooth.es0(), o.smooth.u(), o.smooth.argK(), o.smooth.noiseScale(), trials, baseSeed);
            }
            ElasticOutcome o = elasticBound(prep, cfg, eps, false);
            printCompareRow("naive Lap(1/eps) [unsound]", eps, o, 1.0, 1.0, 0, 1.0 / eps, trials, baseSeed);
            System.out.println();
        }
        for (String w : printedWarnings) System.out.println("WARNING: " + w);
        System.out.println("MRE below ~0.10 is what the paper calls 'enough utility'. The naive row only shows how much "
                + "error a (wrong) sensitivity of 1 would hide.\n");
    }

    private static void printCompareRow(final String name, final double eps, final ElasticOutcome o,
                                        final double es0, final double u, final long kStar, final double scale,
                                        final int trials, final long baseSeed) {
        double[] err = new double[trials];
        double[] rel = new double[trials];
        for (int t = 0; t < trials; t++) {
            NoiseCache cache = new NoiseCache(baseSeed + t);
            long r = releaseCount(o, scale, cache);
            err[t] = Math.abs(r - o.trueCount);
            rel[t] = err[t] / Math.max(o.trueCount, 1.0);
        }
        double mae = mean(err);
        double[] sq = new double[trials];
        for (int t = 0; t < trials; t++) sq[t] = err[t] * err[t];
        System.out.printf("%-26s %5.1f %9s %11s %11s %6d %12s | %10s %10s %9s%n",
                name, eps, fmt(o.trueCount), fmt(es0), fmt(u), kStar, fmt(scale),
                fmt(mae), fmt(Math.sqrt(mean(sq))), fmt(mean(rel)));
    }

    private static String fmt(final double v) {
        if (Double.isNaN(v)) return "NaN";
        return Math.abs(v) >= 1e6 ? String.format("%.3e", v) : String.format("%.2f", v);
    }

    // =====================================================================================
    //  SELECTIVE operator-level DP  vs  UNIFORM post-processing  vs  OSDP-style  vs  ground truth
    // =====================================================================================
    //
    //  ground truth   : the exact result of the query (no noise anywhere)
    //  SELECTIVE_OP   : your idea. At EVERY operator, sensitive contributions get Laplace noise (one cached
    //                   draw per sensitive set); everything else is exact. The root release is the final result.
    //  UNIFORM_POST   : execute exactly, then noise the whole final result (every edge, message, person) once.
    //                   (Uniform noise at every operator gives the same final-result error but spends more budget.)
    //  OSDP_STYLE     : edge count only. After Doudalis et al. (VLDB 2017, one-sided DP): drop the sensitive
    //                   records, count the non-sensitive ones, add ONE-SIDED Laplace noise (OsdpLaplaceL1).
    //                   Its error against the TRUE total therefore includes the dropped sensitive part.
    //
    //  Utility  : relative error (trimmed mean [R2T/DP-S4S protocol], median, 95th percentile [OSDP "Rel95"]),
    //             MAE, RMSE, bias, sd (bias vs noise decomposition), win-rate against uniform.
    //  Privacy  : coverage / exact exposure, per-record composed epsilon (Shrinkwrap/OSDP composition),
    //             empirical audit of the edge-count release (Jagielski et al. / Tramer et al.).
    //  Sweeps   : sweepSensitiveFraction varies the share of sensitive edges and whether sensitivity is random
    //             or concentrated on hubs (OSDP's non-sensitive ratio and Close/Far policies).

    private static final String[] ARM_NAMES = {"selective-op", "uniform-post", "osdp-style"};

    private static final class ArmOutcome {
        double countTrue, countRel;
        boolean hasSum;
        double sumTrue, sumRel;
        final Map<String, double[]> years = new LinkedHashMap<>();   // root persons: {truth, released, protectedFlag}
        Map<String, Release> ledger = new LinkedHashMap<>();
        Result detail;                                               // per-operator rows (selective arm only)
        double ms;
    }

    private static final class Series {
        final double[] abs, rel, signed, sq;
        Series(final int n) { abs = new double[n]; rel = new double[n]; signed = new double[n]; sq = new double[n]; }
        void set(final int t, final double truth, final double est, final double relDenomSanity) {
            double e = est - truth;
            abs[t] = Math.abs(e);
            signed[t] = e;
            sq[t] = e * e;
            rel[t] = DPMetrics.relErr(truth, est, relDenomSanity);
        }
    }

    private static final class TrialResults {
        final Series[] cnt, sum, yr;               // [0]=selective, [1]=uniform, [2]=osdp (counts only)
        final double[][] opRel;
        final double[] ms = new double[3];
        ArmOutcome firstSel, firstUni, firstOsdp;
        TrialResults(final int trials, final int nOps) {
            cnt = new Series[]{new Series(trials), new Series(trials), new Series(trials)};
            sum = new Series[]{new Series(trials), new Series(trials)};
            yr = new Series[]{new Series(trials), new Series(trials)};
            opRel = new double[nOps][trials];
        }
    }

    private static ArmOutcome selectiveOutcome(final Prepared prep, final Set<String> rootPersons,
                                               final double eps, final double clipBound, final NoiseCache cache) {
        long t0 = System.nanoTime();
        Result r = evaluate(prep, eps, clipBound, Scope.SENSITIVE_ONLY, cache);
        ArmOutcome o = new ArmOutcome();
        OpRow root = r.rows.get(r.rows.size() - 1);
        o.countTrue = prep.ops.get(prep.ops.size() - 1).edges.size();
        o.countRel = root.releasedEdgeCount;
        if (root.hasTargetValue) {
            o.hasSum = true;
            o.sumTrue = root.trueSensitiveSum + root.trueNonSensitiveSum;
            o.sumRel = root.releasedSum;
        }
        for (Map.Entry<String, double[]> en : r.years.entrySet()) {
            if (rootPersons.contains(en.getKey())) o.years.put(en.getKey(), en.getValue());
        }
        o.ledger = r.ledger;
        o.detail = r;
        o.ms = (System.nanoTime() - t0) / 1e6;
        return o;
    }

    private static ArmOutcome uniformPostOutcome(final Prepared prep, final Map<String, Node> rootPersonNodes,
                                                 final double eps, final double clipBound, final NoiseCache cache) {
        long t0 = System.nanoTime();
        OpData root = prep.ops.get(prep.ops.size() - 1);
        ArmOutcome o = new ArmOutcome();

        List<String> edgeIds = new ArrayList<>();
        Set<String> eRecs = new HashSet<>();
        for (Edge e : root.edges) { edgeIds.add(e.getId()); eRecs.add("E:" + e.getId()); }
        double cs = 1.0 / eps;
        String ck = canonicalKey("post:edgecount", edgeIds) + "@" + cs;
        o.countTrue = root.edges.size();
        o.countRel = Math.max(0, Math.round(o.countTrue + cache.laplaceNoiseFor(ck, cs)));
        o.ledger.put(ck, new Release("edgecount", eps, eRecs));

        if (!root.msgValue.isEmpty()) {
            double ss = clipBound / eps;
            double truth = 0, clipped = 0;
            Set<String> mRecs = new HashSet<>();
            for (Map.Entry<String, Double> en : root.msgValue.entrySet()) {
                truth += en.getValue();
                clipped += clip(en.getValue(), clipBound);
                mRecs.add("M:" + en.getKey());
            }
            String sk = canonicalKey("post:msgsum", new ArrayList<>(root.msgValue.keySet())) + "@" + ss;
            o.hasSum = true;
            o.sumTrue = truth;
            o.sumRel = clipped + cache.laplaceNoiseFor(sk, ss);
            o.ledger.put(sk, new Release("msgsum", eps, mRecs));
        }

        double ys = YEAR_SENSITIVITY / eps;
        for (Node n : rootPersonNodes.values()) {
            String raw = (n.getProperties() == null) ? null : n.getProperties().get(YEAR_PROP);
            if (raw == null || raw.isBlank()) continue;
            int truthYear;
            try { truthYear = Integer.parseInt(raw.trim()); } catch (NumberFormatException ex) { continue; }
            String yk = "post:birthYear:" + n.getId() + "@" + ys;
            int noisy = clampYear(Math.round(clampYear(truthYear) + cache.laplaceNoiseFor(yk, ys)));
            o.years.put(n.getId(), new double[]{truthYear, noisy, 1});
            o.ledger.put(yk, new Release("year", eps, Set.of("P:" + n.getId())));
        }
        o.ms = (System.nanoTime() - t0) / 1e6;
        return o;
    }

    /**
     * OSDP-style edge count (OsdpLaplaceL1 of Doudalis et al.): count only the NON-sensitive edges of the final
     * result, add one-sided (non-positive) Laplace noise, clamp at 0 and add back the noise median ln(2)/eps.
     * |Lap(b)| is Exp(b), so the one-sided draw is -|Lap(1/eps)| and can come from the same cache.
     */
    private static ArmOutcome osdpOutcome(final Prepared prep, final double eps, final NoiseCache cache) {
        long t0 = System.nanoTime();
        OpData root = prep.ops.get(prep.ops.size() - 1);
        ArmOutcome o = new ArmOutcome();
        List<String> nonSens = new ArrayList<>();
        Set<String> sensRecs = new HashSet<>();
        for (Edge e : root.edges) {
            if (Boolean.TRUE.equals(prep.edgeSensitive.get(e.getId()))) sensRecs.add("E:" + e.getId());
            else nonSens.add(e.getId());
        }
        double scale = 1.0 / eps;
        String key = canonicalKey("osdp:edgecount", nonSens) + "@" + scale;
        double y = nonSens.size() - Math.abs(cache.laplaceNoiseFor(key, scale));
        y = y < 0 ? 0 : y + Math.log(2) * scale;
        o.countTrue = root.edges.size();                     // ground truth is the TOTAL, sensitive part included
        o.countRel = Math.round(y);
        o.ledger.put(key, new Release("osdp-count", eps, sensRecs));
        o.ms = (System.nanoTime() - t0) / 1e6;
        return o;
    }

    private static TrialResults runTrials(final Prepared prep, final Map<String, Node> rootPersonNodes,
                                          final double eps, final double clipBound, final int trials, final long seed) {
        Set<String> rootPersons = rootPersonNodes.keySet();
        int nOps = prep.ops.size();
        TrialResults tr = new TrialResults(trials, nOps);
        for (int t = 0; t < trials; t++) {
            ArmOutcome a = selectiveOutcome(prep, rootPersons, eps, clipBound, new NoiseCache(DPMetrics.mix(seed, t, 1)));
            ArmOutcome b = uniformPostOutcome(prep, rootPersonNodes, eps, clipBound, new NoiseCache(DPMetrics.mix(seed, t, 2)));
            ArmOutcome c = osdpOutcome(prep, eps, new NoiseCache(DPMetrics.mix(seed, t, 3)));
            if (t == 0) { tr.firstSel = a; tr.firstUni = b; tr.firstOsdp = c; }
            tr.ms[0] += a.ms;
            tr.ms[1] += b.ms;
            tr.ms[2] += c.ms;
            ArmOutcome[] arms = {a, b, c};
            for (int k = 0; k < 3; k++) tr.cnt[k].set(t, arms[k].countTrue, arms[k].countRel, 1.0);
            for (int k = 0; k < 2; k++) {
                if (arms[k].hasSum) tr.sum[k].set(t, arms[k].sumTrue, arms[k].sumRel, 1.0);
                double sa = 0, ss = 0, sq = 0;
                int n = 0;
                for (double[] y : arms[k].years.values()) {
                    double e = y[1] - y[0];
                    sa += Math.abs(e); ss += e; sq += e * e; n++;
                }
                if (n > 0) {
                    tr.yr[k].abs[t] = sa / n; tr.yr[k].signed[t] = ss / n; tr.yr[k].sq[t] = sq / n;
                    tr.yr[k].rel[t] = (sa / n) / YEAR_SENSITIVITY;         // range-normalised error
                }
            }
            for (int i = 0; i < nOps; i++) {
                tr.opRel[i][t] = DPMetrics.relErr(prep.ops.get(i).edges.size(), a.detail.rows.get(i).releasedEdgeCount, 1.0);
            }
        }
        return tr;
    }

    private record LedgerSummary(int releases, double maxRecordEps, double meanRecordEps, int records) {}

    /** Sequential composition: a record's epsilon is the sum over the DISTINCT releases that cover it. */
    private static LedgerSummary summarize(final Map<String, Release> ledger) {
        Map<String, Double> per = new HashMap<>();
        for (Release r : ledger.values()) {
            for (String id : r.records()) per.merge(id, r.eps(), Double::sum);
        }
        double max = 0, sum = 0;
        for (double v : per.values()) { max = Math.max(max, v); sum += v; }
        return new LedgerSummary(ledger.size(), max, per.isEmpty() ? 0 : sum / per.size(), per.size());
    }

    private static Map<String, Node> rootPersonNodes(final OpData root) {
        Map<String, Node> m = new LinkedHashMap<>();
        for (Edge e : root.edges) {
            m.putIfAbsent(e.getSource().getId(), e.getSource());
            m.putIfAbsent(e.getTarget().getId(), e.getTarget());
        }
        return m;
    }

    private static long sensitiveRootEdges(final Prepared prep, final OpData root) {
        return root.edges.stream().filter(e -> Boolean.TRUE.equals(prep.edgeSensitive.get(e.getId()))).count();
    }

    /**
     * Compare SELECTIVE_OP, UNIFORM_POST and OSDP_STYLE against the ground truth, on utility AND privacy.
     *
     * @param trials      independent runs per epsilon (100 follows the protocol of the R2T line of work)
     * @param auditTrials released samples per world in the empirical audit (>= 2000 recommended)
     */
    public static void compareSelectiveVsUniform(final String query, final SensitivePatternPolicy pattern,
                                                 final double[] epsilons, final double clipBound,
                                                 final int trials, final long seed, final int auditTrials) throws Exception {
        System.out.println("=== SELECTIVE operator-level DP vs UNIFORM post-processing vs OSDP-style vs ground truth ===");
        System.out.println("query: " + query + "\n");
        printAggregateNoiseMap();
        Prepared prep = prepare(query, pattern);
        OpData root = prep.ops.get(prep.ops.size() - 1);
        Map<String, Node> rootPersonNodes = rootPersonNodes(root);
        long sensRootEdges = sensitiveRootEdges(prep, root);
        System.out.printf("ground truth at the final result: %d paths, %d distinct edges (%d sensitive), %d messages (%d sensitive), %d persons%n%n",
                root.totalPaths, root.edges.size(), sensRootEdges, root.msgValue.size(), root.sensitiveMsgIds.size(), rootPersonNodes.size());

        double focus = epsilons[0];
        for (double e : epsilons) if (Math.abs(e - 1.0) < Math.abs(focus - 1.0)) focus = e;
        final int nOps = prep.ops.size();

        for (double eps : epsilons) {
            TrialResults tr = runTrials(prep, rootPersonNodes, eps, clipBound, trials, seed);

            System.out.printf("%n################  epsilon = %.2f  (%d trials per arm)  ################%n", eps, trials);
            System.out.println("UTILITY of the final result vs ground truth   (rel = |est-truth|/max(truth,1); years: |err|/range; sd = noise, bias = systematic)");
            System.out.printf("%-11s %-13s | %9s %9s %9s | %10s %10s %10s %9s | %10s%n",
                    "statistic", "arm", "rel(trim)", "rel(med)", "rel(p95)", "MAE", "RMSE", "bias", "sd", "win-vs-uni");
            printSeries("edge count", tr.cnt);
            if (root.msgValue.size() > 0) printSeries("msg sum", tr.sum);
            printSeries("birthYear", tr.yr);
            System.out.printf("runtime per release pass (ms): selective=%.3f  uniform=%.3f  osdp=%.3f%n",
                    tr.ms[0] / trials, tr.ms[1] / trials, tr.ms[2] / trials);
            System.out.printf("RELERROR (DP4SQL protocol: |y-yhat|/max(1,y) over %d samples; reference = always output 0 -> RelError %.1f)%n",
                    trials, DPMetrics.ALWAYS_ZERO_REL_ERROR);
            System.out.printf("%-11s %-13s | %8s %8s %8s %8s %8s %8s | %s%n", "statistic", "arm", "min", "Q1", "median", "Q3", "max", "mean", "vs always-0");
            printRelError("edge count", tr.cnt);
            if (root.msgValue.size() > 0) printRelError("msg sum", tr.sum);

            LedgerSummary ls = summarize(tr.firstSel.ledger);
            LedgerSummary lu = summarize(tr.firstUni.ledger);
            LedgerSummary lo = summarize(tr.firstOsdp.ledger);
            long selProtYears = tr.firstSel.years.values().stream().filter(y -> y[2] == 1.0).count();
            double selEdgeShare = root.edges.isEmpty() ? 0 : sensRootEdges / (double) root.edges.size();
            double selMsgShare = root.msgValue.isEmpty() ? 0 : root.sensitiveMsgIds.size() / (double) root.msgValue.size();
            double selPersShare = tr.firstSel.years.isEmpty() ? 0 : selProtYears / (double) tr.firstSel.years.size();
            System.out.println("PRIVACY accounting   (composition: sum of epsilon over distinct releases covering a record)");
            System.out.printf("%-13s | %8s | %12s %12s | %s%n", "arm", "releases", "max eps/rec", "mean eps/rec",
                    "share of the FINAL result that is protected (edges / messages / persons)");
            System.out.printf("%-13s | %8d | %12.2f %12.2f | %5.1f%% / %5.1f%% / %5.1f%%   (the rest is released EXACTLY, eps = infinity)%n",
                    "selective-op", ls.releases(), ls.maxRecordEps(), ls.meanRecordEps(),
                    100 * selEdgeShare, 100 * selMsgShare, 100 * selPersShare);
            System.out.printf("%-13s | %8d | %12.2f %12.2f | %5.1f%% / %5.1f%% / %5.1f%%%n",
                    "uniform-post", lu.releases(), lu.maxRecordEps(), lu.meanRecordEps(), 100.0, 100.0, 100.0);
            System.out.printf("%-13s | %8d | %12.2f %12.2f | %5.1f%% /   n/a  /   n/a    (sensitive edges dropped; non-sensitive ones have no guarantee)%n",
                    "osdp-style", lo.releases(), lo.maxRecordEps(), lo.meanRecordEps(), 100 * selEdgeShare);

            if (eps == focus) {
                System.out.printf("%nSELECTIVE per-operator edge-count error at eps=%.2f (mean over trials; each operator is a release):%n", eps);
                System.out.printf("%-3s %-26s %8s %10s %12s%n", "lvl", "operator", "edges", "sens", "rel err");
                for (int i = 0; i < nOps; i++) {
                    OpData op = prep.ops.get(i);
                    long sens = op.edges.stream().filter(e -> Boolean.TRUE.equals(prep.edgeSensitive.get(e.getId()))).count();
                    System.out.printf("%-3d %-26s %8d %10d %12.4f%n", op.depth, op.label, op.edges.size(), sens, DPMetrics.mean(tr.opRel[i]));
                }
            }

            auditRootEdges(prep, eps, auditTrials, seed);
        }
        System.out.println();
        System.out.println("How to read: 'win-vs-uni' is the share of trials where the arm had smaller error than uniform-post. Selective is expected to");
        System.out.println("win on utility; the privacy tables show the cost: exact exposure, composition over operators, and (audit) whether the exact");
        System.out.println("part leaks sensitive records through the join. osdp-style trades the dropped sensitive part (bias) for very low noise.\n");
    }

    /** One row per arm: the boxplot numbers of RelError (Series.rel is |y-yhat|/max(1,y) for y >= 0) and the comparison with answering 0. */
    private static void printRelError(final String stat, final Series[] s) {
        for (int k = 0; k < s.length; k++) {
            DPMetrics.BoxStats b = DPMetrics.boxStats(s[k].rel);
            String verdict = DPMetrics.verdictVsZero(b.median());
            System.out.printf("%-11s %-13s | %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f | %s%n", stat, ARM_NAMES[k],
                    b.min(), b.q1(), b.median(), b.q3(), b.max(), b.mean(), verdict);
        }
    }

    /** Share of paired trials where a has smaller relative error than b (ties count half). */
    private static double winShare(final Series a, final Series b) {
        int n = a.rel.length;
        double w = 0;
        for (int t = 0; t < n; t++) w += a.rel[t] < b.rel[t] ? 1.0 : (a.rel[t] == b.rel[t] ? 0.5 : 0.0);
        return w / n;
    }

    /** Says, once, what each arm of the AGGREGATE comparison does with the sensitive part and what the error is computed over. */
    private static void printAggregateNoiseMap() {
        System.out.println("WHERE IS THE NOISE? (aggregates: edge count, message-length sum, birthYear of the persons in the result)");
        System.out.printf("  %-14s | %-38s | %-24s | %s%n", "arm", "SENSITIVE part", "NON-SENSITIVE part", "noise");
        System.out.printf("  %-14s | %-38s | %-24s | %s%n", "selective-op", "INCLUDED, clipped; noise on the sub-aggregate", "exact",
                "Lap(1/eps) on the sensitive edge count, Lap(clip/eps) on the sensitive sum, Lap(range/eps) on each protected person's year; one cached draw per target");
        System.out.printf("  %-14s | %-38s | %-24s | %s%n", "uniform-post", "included", "included (treated alike)",
                "the same Laplace noise on the WHOLE final aggregate and on every person");
        System.out.printf("  %-14s | %-38s | %-24s | %s%n", "osdp-style", "REMOVED (not counted)", "counted",
                "one-sided noise on the count of the non-sensitive edges only; the sensitive edges cost accuracy only because they are missing");
        System.out.println("  ERROR IS MEASURED over the whole final result, not over the retrieved values only: the edge count and the message sum are compared with");
        System.out.println("  the TRUE total (sensitive part included); birthYear is averaged over ALL persons of the final result (unprotected persons contribute 0).");
        System.out.println("  Metrics restricted to the protected part alone (e.g. 'MAE on perturbed cells only') are shown only for contrast.\n");
    }

    private static void printSeries(final String stat, final Series[] s) {
        for (int k = 0; k < s.length; k++) {
            String win = "";
            if (k != 1) win = String.format("%.0f%%", 100.0 * winShare(s[k], s[1]));   // each arm vs uniform-post (index 1)
            System.out.printf("%-11s %-13s | %9.4f %9.4f %9.4f | %10.3f %10.3f %10.3f %9.3f | %10s%n",
                    stat, ARM_NAMES[k],
                    DPMetrics.trimmedMean(s[k].rel, 0.2), DPMetrics.median(s[k].rel), DPMetrics.percentile(s[k].rel, 0.95),
                    DPMetrics.mean(s[k].abs), Math.sqrt(DPMetrics.mean(s[k].sq)),
                    DPMetrics.mean(s[k].signed), DPMetrics.std(s[k].signed), win);
        }
    }

    // ----------------------------------------------------------------------------------------------
    //  Sweep: how much does selectivity help as the share of sensitive data changes?
    // ----------------------------------------------------------------------------------------------
    /** Re-labels which edges are sensitive without re-running the query (messages: sensitive if any edge on a path is). */
    private static void reclassify(final Prepared prep, final java.util.function.Predicate<Edge> sensitive) {
        prep.edgeSensitive.clear();
        for (Edge e : prep.allEdges.values()) prep.edgeSensitive.put(e.getId(), sensitive.test(e));
        for (OpData op : prep.ops) {
            op.sensitiveMsgIds.clear();
            for (int i = 0; i < op.pathEdges.size(); i++) {
                String msg = op.pathMsg.get(i);
                if (msg == null) continue;
                for (String id : op.pathEdges.get(i)) {
                    if (Boolean.TRUE.equals(prep.edgeSensitive.get(id))) { op.sensitiveMsgIds.add(msg); break; }
                }
            }
        }
    }

    /**
     * Sweep the share of sensitive edges (OSDP's non-sensitive ratio) under two synthetic policies:
     *  RANDOM : sensitive edges chosen uniformly at random            (sensitivity unrelated to structure: OSDP "Close")
     *  HUB    : sensitive edges are those touching the highest-degree nodes (sensitivity tied to structure: OSDP "Far")
     * At fraction 1.0 selective must equal uniform (sanity check). The crossover where selective stops winning
     * on utility is the number to report.
     */
    public static void sweepSensitiveFraction(final String query, final SensitivePatternPolicy pattern,
                                              final double[] fractions, final double eps, final double clipBound,
                                              final int trials, final long seed) throws Exception {
        System.out.println("=== SWEEP of the sensitive share (eps = " + eps + ", " + trials + " trials per cell) ===");
        System.out.println("query: " + query + "\n");
        Prepared prep = prepare(query, pattern);
        OpData root = prep.ops.get(prep.ops.size() - 1);
        Map<String, Node> rootPersonNodes = rootPersonNodes(root);

        Map<String, Integer> deg = new HashMap<>();
        for (Edge e : prep.allEdges.values()) {
            deg.merge(e.getSource().getId(), 1, Integer::sum);
            deg.merge(e.getTarget().getId(), 1, Integer::sum);
        }
        List<String> randomOrder = new ArrayList<>(prep.allEdges.keySet());
        Collections.sort(randomOrder);
        SplittableRandom rng = new SplittableRandom(DPMetrics.mix(seed, 777, 0));
        for (int i = randomOrder.size() - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            String tmp = randomOrder.get(i); randomOrder.set(i, randomOrder.get(j)); randomOrder.set(j, tmp);
        }
        List<String> hubOrder = new ArrayList<>(prep.allEdges.keySet());
        hubOrder.sort((x, y) -> {
            Edge ex = prep.allEdges.get(x), ey = prep.allEdges.get(y);
            int sx = Math.max(deg.get(ex.getSource().getId()), deg.get(ex.getTarget().getId()));
            int sy = Math.max(deg.get(ey.getSource().getId()), deg.get(ey.getTarget().getId()));
            return sx != sy ? Integer.compare(sy, sx) : x.compareTo(y);
        });

        System.out.printf("%-7s %6s %10s | %-29s | %-19s | %-19s | %s%n", "policy", "sens%", "root-sens%",
                "edge count rel: sel/uni/osdp", "msg sum rel: sel/uni", "birthYear rel: sel/uni", "selective wins (>=65% of trials)");
        for (String mode : new String[]{"RANDOM", "HUB"}) {
            List<String> order = mode.equals("RANDOM") ? randomOrder : hubOrder;
            for (double f : fractions) {
                int k = (int) Math.ceil(f * order.size());
                Set<String> sens = new HashSet<>(order.subList(0, Math.min(k, order.size())));
                reclassify(prep, e -> sens.contains(e.getId()));
                TrialResults tr = runTrials(prep, rootPersonNodes, eps, clipBound, trials, seed);
                double rootShare = root.edges.isEmpty() ? 0 : sensitiveRootEdges(prep, root) / (double) root.edges.size();
                double cs = DPMetrics.trimmedMean(tr.cnt[0].rel, 0.2), cu = DPMetrics.trimmedMean(tr.cnt[1].rel, 0.2),
                        co = DPMetrics.trimmedMean(tr.cnt[2].rel, 0.2);
                double ss = DPMetrics.trimmedMean(tr.sum[0].rel, 0.2), su = DPMetrics.trimmedMean(tr.sum[1].rel, 0.2);
                double ys = DPMetrics.trimmedMean(tr.yr[0].rel, 0.2), yu = DPMetrics.trimmedMean(tr.yr[1].rel, 0.2);
                StringBuilder wins = new StringBuilder();
                final double need = 0.65;                // paired win-rate; ~3 sd above chance at 100 trials
                if (winShare(tr.cnt[0], tr.cnt[1]) >= need) wins.append("count ");
                if (root.msgValue.size() > 0 && winShare(tr.sum[0], tr.sum[1]) >= need) wins.append("sum ");
                if (winShare(tr.yr[0], tr.yr[1]) >= need) wins.append("year");
                System.out.printf("%-7s %5.0f%% %9.1f%% | %9.4f /%9.4f /%9.4f | %8.4f /%8.4f | %8.4f /%8.4f | %s%n",
                        mode, 100 * f, 100 * rootShare, cs, cu, co,
                        root.msgValue.size() > 0 ? ss : Double.NaN, root.msgValue.size() > 0 ? su : Double.NaN, ys, yu,
                        wins.length() == 0 ? "-" : wins.toString().trim());
            }
        }
        System.out.println();
        System.out.println("At 100% selective equals uniform. Below that, the gap is the utility bought by leaving data unprotected;");
        System.out.println("compare RANDOM with HUB: when sensitivity sits on the high-influence nodes, the benefit shrinks and the exact part leaks more.\n");
    }

    // ----------------------------------------------------------------------------------------------
    //  Empirical audit of the final edge-count release
    // ----------------------------------------------------------------------------------------------
    /**
     * Distinguishing game on the two most influential edges of the final result: one sensitive, one not.
     * World D is the real final result; world D-e is the final result recomputed WITHOUT the paths that use e
     * (deleting edge e from the graph removes exactly those paths). The attacker sees only the released edge count.
     * The measured lower bound on epsilon is compared with the nominal epsilon of the release.
     */
    private static void auditRootEdges(final Prepared prep, final double eps, final int trials, final long seed) {
        OpData root = prep.ops.get(prep.ops.size() - 1);
        System.out.printf("EMPIRICAL AUDIT of the final edge-count release  (nominal eps = %.2f, %d samples per world, 95%% lower bound)%n", eps, trials);
        if (root.pathEdges.isEmpty()) {
            System.out.println("  skipped: the final result has no paths.");
            return;
        }
        Map<String, Integer> infl = new HashMap<>();
        for (String[] ids : root.pathEdges) {
            for (String id : new HashSet<>(Arrays.asList(ids))) infl.merge(id, 1, Integer::sum);
        }
        String bestSens = null, bestNon = null;
        for (Map.Entry<String, Integer> en : infl.entrySet()) {
            boolean sens = Boolean.TRUE.equals(prep.edgeSensitive.get(en.getKey()));
            if (sens) {
                if (bestSens == null || en.getValue() > infl.get(bestSens)
                        || (en.getValue().equals(infl.get(bestSens)) && en.getKey().compareTo(bestSens) < 0)) bestSens = en.getKey();
            } else {
                if (bestNon == null || en.getValue() > infl.get(bestNon)
                        || (en.getValue().equals(infl.get(bestNon)) && en.getKey().compareTo(bestNon) < 0)) bestNon = en.getKey();
            }
        }
        System.out.printf("  %-13s %-22s %6s | %9s %8s | %8s | %s%n", "arm", "target edge", "paths", "d(noised)", "d(exact)", "eps-hat", "verdict");
        if (bestSens == null) {
            System.out.println("  no sensitive edge appears in the final result: selective-op protects nothing here (everything is exact).");
        }
        String[] targets = {bestSens, bestNon};
        String[] kinds = {"sensitive", "non-sensitive"};
        for (int ti = 0; ti < 2; ti++) {
            String target = targets[ti];
            if (target == null) continue;
            Set<String> full = new HashSet<>(infl.keySet());
            Set<String> reduced = new HashSet<>();
            for (String[] ids : root.pathEdges) {
                boolean uses = false;
                for (String id : ids) if (id.equals(target)) { uses = true; break; }
                if (!uses) reduced.addAll(Arrays.asList(ids));
            }
            int protF = 0, protR = 0;
            for (String id : full) if (Boolean.TRUE.equals(prep.edgeSensitive.get(id))) protF++;
            for (String id : reduced) if (Boolean.TRUE.equals(prep.edgeSensitive.get(id))) protR++;
            int nonF = full.size() - protF, nonR = reduced.size() - protR;

            for (int arm = 0; arm < 2; arm++) {
                SplittableRandom rng = new SplittableRandom(DPMetrics.mix(seed, 100 + arm * 10 + ti, Double.doubleToLongBits(eps)));
                double[] w1 = new double[trials], w0 = new double[trials];
                double scale = 1.0 / eps;
                for (int i = 0; i < trials; i++) {
                    if (arm == 0) {        // selective: exact non-sensitive part + noisy sensitive part
                        w1[i] = nonF + Math.max(0, Math.round(protF + DPMetrics.laplace(rng, scale)));
                        w0[i] = nonR + Math.max(0, Math.round(protR + DPMetrics.laplace(rng, scale)));
                    } else {               // uniform: whole count noised
                        w1[i] = Math.max(0, Math.round(full.size() + DPMetrics.laplace(rng, scale)));
                        w0[i] = Math.max(0, Math.round(reduced.size() + DPMetrics.laplace(rng, scale)));
                    }
                }
                DPMetrics.AuditResult ar = DPMetrics.audit(w1, w0, 0.05);
                int dNoised = arm == 0 ? protF - protR : full.size() - reduced.size();
                int dExact = arm == 0 ? nonF - nonR : 0;
                String verdict;
                double cap = Math.log(ar.nTest() / 3.7);
                if (ar.epsLower() <= eps + 0.15) {
                    verdict = "consistent with nominal eps";
                } else if (arm == 0 && dExact > 0) {
                    verdict = ar.epsLower() >= cap - 0.5 ? "LEAK: exact part moves (>= trial cap)" : "LEAK: exact part moves";
                } else {
                    verdict = String.format("VIOLATION: statistic moves by %d, sensitivity 1 assumed", dNoised);
                }
                System.out.printf("  %-13s %-22s %6d | %9d %8d | %8.2f | %s%n", arm == 0 ? "selective-op" : "uniform-post",
                        kinds[ti] + " " + target, infl.get(target), dNoised, dExact, ar.epsLower(), verdict);
            }
        }
        System.out.println("  (eps-hat is a lower bound on the true epsilon; the largest value measurable with this many samples is about "
                + String.format("%.1f", Math.log(trials / 2 / 3.7)) + ")");
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