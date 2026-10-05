package com.gdblab.execution;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.function.ToDoubleFunction;
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
 * Operator-level evaluation for an EDGE-pattern-sensitive query. Load a graph first
 * with "/load -n ... -e ...", then run this against any query.
 *
 * Two aggregates, each using the sensitivity notion appropriate to its own
 * granularity:
 *  - EDGE COUNT: how many "knows"-style edges are sensitive vs not - classified by
 *    SensitivePatternPolicy.isSensitiveEdge (edge-level: Bob->David stays exact even
 *    in a path that also contains Alice->President).
 *  - Message.length SUM/AVG: classified by SensitivePatternPolicy.isSensitivePath
 *    (path-level: a message's length is only meaningfully "reached through
 *    President" via the whole knows-chain leading to it, not any single edge).
 *
 * Both follow the same mechanism shape: exact sum/count over the NON-sensitive
 * contributors (no clipping, no noise - they never needed protection), clipped
 * sum/count over the SENSITIVE contributors, ONE Laplace draw added to the
 * sensitive sub-aggregate, combined via post-processing.
 *
 * NOISE IS CACHED, KEYED BY THE CANONICAL SET OF CONTRIBUTING SENSITIVE RECORD IDS:
 * if two operators (e.g. a recursive join and the union above it) protect the exact
 * same underlying sensitive contributors, they get the EXACT SAME noisy answer -
 * reused, not redrawn - because propagating the SAME protected release to an
 * ancestor operator is not a new disclosure. If an operator's sensitive set grows
 * (recursion finds another sensitive edge, say), that is a genuinely different
 * target and gets its own fresh draw. Every row below is annotated NEW or REUSED so
 * this is directly visible, not just asserted.
 */
public final class Evaluator {

    private static final class OpRow {
        String label;
        int depth;
        int totalPaths;

        // edge-count aggregate
        int uniqueEdges;
        long sensitiveEdgeCount;
        long nonSensitiveEdgeCount;
        double releasedEdgeCount;
        boolean edgeNoiseIsNew;

        // value (SUM/AVG) aggregate
        boolean hasTargetValue;
        int sensitiveMessages;
        int nonSensitiveMessages;
        double trueSensitiveSum;
        double trueNonSensitiveSum;
        double releasedSum;
        double releasedAvg;
        boolean valueNoiseIsNew;
    }

    public static void main(String[] args) throws Exception {
        Tools.loadDefaultGraph();
        SensitivePatternPolicy pattern = new SensitivePatternPolicy();
        pattern.addSensitiveNodeName("President");
        run("MATCH p = (x)-[knows+.posted]->(msg) RETURN x.name, msg.length;", pattern, 1.0, 600.0);
    }

    /**
     * @param query     any PathDB query
     * @param pattern   externally configured edge-sensitivity policy (/sdp-sensitive-edge)
     * @param epsilon   epsilon used for every release (diagnostic and real) in this run
     * @param clipBound clip bound for Message.length contributions to the sensitive sub-sum
     */
    public static void run(final String query, final SensitivePatternPolicy pattern,
                            final double epsilon, final double clipBound) throws Exception {
        Predicate<Path> isSensitivePath = pattern::isSensitivePath;
        ToDoubleFunction<Path> messageLength = p -> {
            Node last = p.last();
            String v = (last.getProperties() != null) ? last.getProperties().get("length") : null;
            return (v == null || v.isEmpty()) ? 0.0 : Double.parseDouble(v);
        };

        System.out.println("=== Sensitive-edge-pattern operator-level evaluation ===");
        System.out.println("query: " + query);
        System.out.println("sensitive edges: label in " + pattern.getSensitiveEdgeLabels()
                + ", or touching a node with id in " + pattern.getSensitiveNodeIds()
                + ", name in " + pattern.getSensitiveNodeNames()
                + ", or status matching " + pattern.getSensitiveStatusValues() + "\n");

        LogicalOperator root = IntermediateResultsExplainer.parseToLogicalRoot(query);
        List<OpRow> rows = new ArrayList<>();
        NoiseCache noiseCache = new NoiseCache(); // ONE cache for the whole run - shared across every operator

        // walk() materializes every operator (including root) exactly once; its
        // return value IS root's path list - reused below for the edge listing, no
        // second materialize() call.
        List<Path> finalPaths = walk(root, 0, messageLength, isSensitivePath, pattern, epsilon, clipBound, rows, noiseCache);

        Map<String, Edge> rootEdgesMap = new LinkedHashMap<>();
        for (Path p : finalPaths) {
            for (GraphObject go : p.getSequence()) {
                if (go instanceof Edge e) rootEdgesMap.putIfAbsent(e.getId(), e);
            }
        }
        List<Edge> rootEdgesHolder = new ArrayList<>(rootEdgesMap.values());

        System.out.println("--- EDGE COUNT, per operator (classified at EACH operator; noise cached by sensitive-edge-set) ---");
        System.out.printf("%-3s %-26s %7s %8s %10s %10s %12s %-8s%n",
                "lvl", "operator", "paths", "edges", "sensitive", "non-sens", "released", "noise");
        for (OpRow r : rows) {
            System.out.printf("%-3d %-26s %7d %8d %10d %10d %12.2f %-8s%n",
                    r.depth, r.label, r.totalPaths, r.uniqueEdges, r.sensitiveEdgeCount, r.nonSensitiveEdgeCount,
                    r.releasedEdgeCount, r.edgeNoiseIsNew ? "NEW" : "REUSED");
        }
        System.out.println();

        boolean anyValueReached = rows.stream().anyMatch(r -> r.hasTargetValue);
        if (anyValueReached) {
            System.out.println("--- Message.length SUM/AVG, per operator (noise cached by sensitive-message-set) ---");
            System.out.printf("%-3s %-26s %10s %10s %14s %16s %12s %10s %-8s%n",
                    "lvl", "operator", "sens-msgs", "nonsens", "true-sens-sum", "true-nonsens-sum", "released-sum", "avg", "noise");
            for (OpRow r : rows) {
                if (r.hasTargetValue) {
                    System.out.printf("%-3d %-26s %10d %10d %14.1f %16.1f %12.2f %10.2f %-8s%n",
                            r.depth, r.label, r.sensitiveMessages, r.nonSensitiveMessages,
                            r.trueSensitiveSum, r.trueNonSensitiveSum, r.releasedSum, r.releasedAvg,
                            r.valueNoiseIsNew ? "NEW" : "REUSED");
                }
            }
            System.out.println();
        } else {
            System.out.println("--- Message.length SUM/AVG: this query never reaches a Message node - nothing to sum ---\n");
        }

        OpRow rootRow = rows.get(rows.size() - 1);

        System.out.println("--- edge-by-edge verdicts at the FINAL (root) operator (capped at 30) ---");
        System.out.printf("%-6s %-40s %-10s%n", "id", "edge", "verdict");
        int shown = 0;
        for (Edge e : rootEdgesHolder) {
            if (shown++ >= 30) {
                System.out.println("  ... (" + (rootEdgesHolder.size() - 30) + " more, truncated for display)");
                break;
            }
            boolean sensitive = pattern.isSensitiveEdge(e);
            System.out.printf("%-6s (%s:%s)-[%s]->(%s:%s) %-10s%n",
                    e.getId(), e.getSource().getId(), safeName(e.getSource()), e.getLabel(),
                    e.getTarget().getId(), safeName(e.getTarget()), sensitive ? "SENSITIVE" : "not sensitive");
        }

        System.out.printf("%n=== Real releases (root operator), eps=%.2f each ===%n", epsilon);
        System.out.printf("Edge count:          released=%.2f (true=%d sensitive + %d non-sensitive = %d), noise %s%n",
                rootRow.releasedEdgeCount, rootRow.sensitiveEdgeCount, rootRow.nonSensitiveEdgeCount,
                rootRow.sensitiveEdgeCount + rootRow.nonSensitiveEdgeCount,
                rootRow.edgeNoiseIsNew ? "freshly drawn here" : "reused from an earlier operator");
        if (rootRow.hasTargetValue) {
            System.out.printf("Message length SUM:  released=%.2f (true=%.1f), AVG=%.2f, noise %s%n",
                    rootRow.releasedSum, rootRow.trueSensitiveSum + rootRow.trueNonSensitiveSum, rootRow.releasedAvg,
                    rootRow.valueNoiseIsNew ? "freshly drawn here" : "reused from an earlier operator");
        }
        System.out.printf("Total distinct noise draws this run (edge-count cache + value-sum cache combined): %d%n%n",
                noiseCache.size());

        Tools.resetContext();
    }

    private static String safeName(final Node n) {
        String name = (n.getProperties() != null) ? n.getProperties().get("name") : null;
        return name != null ? name : "";
    }

    private static String canonicalKey(final String prefix, final java.util.Collection<String> ids) {
        return prefix + ":" + ids.stream().sorted().collect(Collectors.joining(","));
    }

    private static double clip(final double v, final double bound) {
        return Math.max(-bound, Math.min(bound, v));
    }

    private static List<Path> walk(final LogicalOperator node, final int depth,
                                    final ToDoubleFunction<Path> messageLength,
                                    final Predicate<Path> isSensitivePath,
                                    final SensitivePatternPolicy pattern,
                                    final double epsilon,
                                    final double clipBound,
                                    final List<OpRow> rows,
                                    final NoiseCache noiseCache) {
        if (node instanceof UnaryLogicalOperator u) {
            walk(u.getChild(), depth + 1, messageLength, isSensitivePath, pattern, epsilon, clipBound, rows, noiseCache);
        } else if (node instanceof BinaryLogicalOperator b) {
            walk(b.getLeftChild(), depth + 1, messageLength, isSensitivePath, pattern, epsilon, clipBound, rows, noiseCache);
            walk(b.getRightChild(), depth + 1, messageLength, isSensitivePath, pattern, epsilon, clipBound, rows, noiseCache);
        } else if (node instanceof NullaryLogicalOperator) {
            // leaf
        }

        List<Path> paths = IntermediateResultsExplainer.materialize(node);
        OpRow row = new OpRow();
        row.label = node.getClass().getSimpleName();
        row.depth = depth;
        row.totalPaths = paths.size();

        // ---- EDGE COUNT aggregate ----
        Map<String, Edge> uniqueEdgesHere = new LinkedHashMap<>();
        for (Path p : paths) {
            for (GraphObject go : p.getSequence()) {
                if (go instanceof Edge e) uniqueEdgesHere.putIfAbsent(e.getId(), e);
            }
        }
        List<Edge> edgesHere = new ArrayList<>(uniqueEdgesHere.values());
        List<String> sensitiveEdgeIds = new ArrayList<>();
        long nonSensitiveEdgeCount = 0;
        for (Edge e : edgesHere) {
            if (pattern.isSensitiveEdge(e)) sensitiveEdgeIds.add(e.getId());
            else nonSensitiveEdgeCount++;
        }
        String edgeKey = canonicalKey("edgecount", sensitiveEdgeIds);
        boolean edgeNoiseIsNew = !noiseCache.isCached(edgeKey);
        double edgeNoise = noiseCache.laplaceNoiseFor(edgeKey, 1.0 / Math.max(epsilon, 1e-9)); // sensitivity=1 per edge
        double releasedEdgeCount = (sensitiveEdgeIds.size() + edgeNoise) + nonSensitiveEdgeCount;

        row.uniqueEdges = edgesHere.size();
        row.sensitiveEdgeCount = sensitiveEdgeIds.size();
        row.nonSensitiveEdgeCount = nonSensitiveEdgeCount;
        row.releasedEdgeCount = releasedEdgeCount;
        row.edgeNoiseIsNew = edgeNoiseIsNew;

        // ---- Message.length SUM/AVG aggregate ----
        // dedupe by Message node id: a recursive query can reach the SAME message via
        // more than one path (different hop counts), which would otherwise double-
        // count its length.
        Map<String, Double> sensitiveMsgLengths = new LinkedHashMap<>();
        Map<String, Double> nonSensitiveMsgLengths = new LinkedHashMap<>();
        for (Path p : paths) {
            Node last = p.last();
            Map<String, String> props = last.getProperties();
            boolean hasValue = props != null && props.get("length") != null && !props.get("length").isEmpty();
            if (!hasValue) continue;
            row.hasTargetValue = true;
            double len = messageLength.applyAsDouble(p);
            if (isSensitivePath.test(p)) sensitiveMsgLengths.putIfAbsent(last.getId(), len);
            else nonSensitiveMsgLengths.putIfAbsent(last.getId(), len);
        }
        if (row.hasTargetValue) {
            double trueSensitiveSum = sensitiveMsgLengths.values().stream().mapToDouble(Double::doubleValue).sum();
            double trueNonSensitiveSum = nonSensitiveMsgLengths.values().stream().mapToDouble(Double::doubleValue).sum();
            double clippedSensitiveSum = sensitiveMsgLengths.values().stream().mapToDouble(v -> clip(v, clipBound)).sum();

            String valueKey = canonicalKey("valuesum", sensitiveMsgLengths.keySet());
            boolean valueNoiseIsNew = !noiseCache.isCached(valueKey);
            double valueNoise = noiseCache.laplaceNoiseFor(valueKey, clipBound / Math.max(epsilon, 1e-9));
            double releasedSum = (clippedSensitiveSum + valueNoise) + trueNonSensitiveSum; // exact public part + clipped+noised sensitive part

            int totalMsgs = sensitiveMsgLengths.size() + nonSensitiveMsgLengths.size();

            row.sensitiveMessages = sensitiveMsgLengths.size();
            row.nonSensitiveMessages = nonSensitiveMsgLengths.size();
            row.trueSensitiveSum = trueSensitiveSum;
            row.trueNonSensitiveSum = trueNonSensitiveSum;
            row.releasedSum = releasedSum;
            row.releasedAvg = releasedSum / Math.max(1, totalMsgs);
            row.valueNoiseIsNew = valueNoiseIsNew;
        }

        rows.add(row);
        return paths;
    }
}