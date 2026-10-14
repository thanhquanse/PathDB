package com.gdblab.execution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Function;

import com.gdblab.algebra.queryplan.logical.BinaryLogicalOperator;
import com.gdblab.algebra.queryplan.logical.LogicalOperator;
import com.gdblab.algebra.queryplan.logical.UnaryLogicalOperator;
import com.gdblab.graph.schema.Edge;
import com.gdblab.graph.schema.GraphObject;
import com.gdblab.graph.schema.Node;
import com.gdblab.graph.schema.Path;
import com.gdblab.privacy.DPMetrics;
import com.gdblab.privacy.RecordReleaseTracker;
import com.gdblab.privacy.SensitivePatternPolicy;

/**
 * TRUE-DATA RELEASE evaluation, following the "releasing true data" experiments of one-sided DP
 * (Doudalis et al., VLDB 2017, Sections 4 and 6.3.2).
 *
 * RECORD       : one result path of an operator (a row PathDB returns). Sensitive iff pattern.isSensitivePath(path).
 * DOWNSTREAM   : the n-gram histogram of the released paths (n consecutive nodes, counted once per path, the way
 *                OSDP counts distinct users per n-gram). Nodes are named by nodeKey (default: node id).
 *
 * ARMS (all compared with the GROUND TRUTH = the normal, non-private execution of the query)
 *   all-NS exact          : release every non-sensitive result path exactly, drop the sensitive ones. NOT private:
 *                           the absence of a record reveals that it is sensitive (an exclusion attack). Shown as the
 *                           reference that OSDP calls "All NS", i.e. what exact selective release would give.
 *   selective-op RR       : YOUR APPROACH made safe. At EVERY operator the exact output is released through OsdpRR
 *                           (sensitive: never; non-sensitive: with probability 1 - e^-eps) and a RecordReleaseTracker
 *                           makes each record's coin a one-time decision shared by all operators. The final result
 *                           is the root operator's released set. Reported raw and rescaled by 1/(1 - e^-eps)
 *                           (the paper does not say whether it rescales, so both are shown).
 *   DP-post LM T_k        : execute exactly, then release only a Laplace-noised n-gram histogram of the final result,
 *                           all records protected. Each path contributes at most k n-grams (truncation), so the
 *                           L1 sensitivity is 2k and the noise is Lap(2k/eps) per bin (OSDP's "LM T1").
 *                           True records cannot be released under standard DP.
 *   DP-post LM T*         : the best k by oracle (NOT private; it looks at the data), as in the paper.
 *
 * METRICS (OSDP, Section 6.2): MRE = (1/d) sum_i |x_i - est_i| / max(x_i, 1) over the d possible n-grams
 * (d = V^n, V = number of distinct node keys); zero bins of the Laplace arm are added analytically (expected
 * |Lap(b)| = b), exactly as the paper does. Because d is huge, we also report MRE over the ACTIVE bins and its median
 * and 95th percentile (Rel50 / Rel95 of the paper, restricted to active bins). Values are the trimmed mean over
 * trials (20% dropped on each side, the R2T / DP-S4S protocol).
 *
 * OPERATOR-LEVEL SEMANTICS (assumption, as in your Evaluator): every operator's EXACT output is materialised once
 * and released separately; released outputs are not fed into the parent operator (the engine cannot re-run an
 * operator on released inputs from outside). Hence, at the root the utility of tracked operator-level RR equals RR
 * applied to the final result only; what operator-level adds is the intermediate releases, and what tracking
 * guarantees is that a record emitted by k operators still costs eps once instead of k * eps (ledger below).
 */
public final class DataReleaseEvaluator {

    private DataReleaseEvaluator() {}

    public static final Function<Node, String> NODE_ID = Node::getId;

    // =====================================================================================
    //  Snapshot of the plan (the query is executed ONCE; everything after is in memory)
    // =====================================================================================
    private static final class Rec {
        final String id;
        boolean sensitive;
        final String[] nodeKeys;
        final String[] edgeIds;
        final String[] nodeNames;        // "id(name)", display only
        final String[] nodeIds;          // node ids, used for coverage
        final String[] edgeLabels;       // display only
        final boolean[] edgeSensitive;   // pattern.isSensitiveEdge per edge, display only
        private final Map<Integer, List<String>> gramCache = new HashMap<>();

        Rec(final String id, final boolean sensitive, final String[] nodeKeys, final String[] edgeIds,
            final String[] nodeNames, final String[] edgeLabels, final boolean[] edgeSensitive, final String[] nodeIds) {
            this.nodeIds = nodeIds;
            this.id = id;
            this.sensitive = sensitive;
            this.nodeKeys = nodeKeys;
            this.edgeIds = edgeIds;
            this.nodeNames = nodeNames;
            this.edgeLabels = edgeLabels;
            this.edgeSensitive = edgeSensitive;
        }

        /** Distinct n-grams of this path, in path order. */
        List<String> grams(final int n) {
            return gramCache.computeIfAbsent(n, k -> {
                LinkedHashSet<String> s = new LinkedHashSet<>();
                for (int i = 0; i + n <= nodeKeys.length; i++) {
                    s.add(String.join(">", Arrays.copyOfRange(nodeKeys, i, i + n)));
                }
                return new ArrayList<>(s);
            });
        }
    }

    private static final class Op {
        String label;
        int depth;
        final List<Integer> children = new ArrayList<>();      // indexes of child operators in the plan list
        final Map<String, Rec> recs = new LinkedHashMap<>();
    }

    private static List<Op> snapshot(final String query, final SensitivePatternPolicy pattern,
                                     final Function<Node, String> nodeKey) throws Exception {
        try {
            LogicalOperator root = IntermediateResultsExplainer.parseToLogicalRoot(query);
            List<Op> ops = new ArrayList<>();
            collect(root, 0, pattern, nodeKey, ops);
            return ops;
        } finally {
            Tools.resetContext();
        }
    }

    private static int collect(final LogicalOperator node, final int depth, final SensitivePatternPolicy pattern,
                               final Function<Node, String> nodeKey, final List<Op> out) {
        List<Integer> kids = new ArrayList<>();
        if (node instanceof UnaryLogicalOperator u) {
            kids.add(collect(u.getChild(), depth + 1, pattern, nodeKey, out));
        } else if (node instanceof BinaryLogicalOperator b) {
            kids.add(collect(b.getLeftChild(), depth + 1, pattern, nodeKey, out));
            kids.add(collect(b.getRightChild(), depth + 1, pattern, nodeKey, out));
        }
        Op op = new Op();
        op.label = node.getClass().getSimpleName();
        op.depth = depth;
        op.children.addAll(kids);
        for (Path p : IntermediateResultsExplainer.materialize(node)) {
            StringBuilder id = new StringBuilder();
            List<String> keys = new ArrayList<>();
            List<String> names = new ArrayList<>();
            List<String> nids = new ArrayList<>();
            List<String> edges = new ArrayList<>();
            List<String> elabels = new ArrayList<>();
            List<Boolean> esens = new ArrayList<>();
            for (GraphObject go : p.getSequence()) {
                if (go instanceof Edge e) {
                    id.append("E:").append(e.getId()).append('|');
                    edges.add(e.getId());
                    elabels.add(String.valueOf(e.getLabel()));
                    esens.add(pattern.isSensitiveEdge(e));
                } else if (go instanceof Node nd) {
                    id.append("N:").append(nd.getId()).append('|');
                    keys.add(nodeKey.apply(nd));
                    nids.add(nd.getId());
                    String nm = (nd.getProperties() == null) ? null : nd.getProperties().get("name");
                    names.add(nd.getId() + (nm == null ? "" : "(" + nm + ")"));
                }
            }
            boolean[] es = new boolean[esens.size()];
            for (int i = 0; i < es.length; i++) es[i] = esens.get(i);
            String rid = id.toString();
            op.recs.putIfAbsent(rid, new Rec(rid, pattern.isSensitivePath(p), keys.toArray(new String[0]),
                    edges.toArray(new String[0]), names.toArray(new String[0]), elabels.toArray(new String[0]), es,
                    nids.toArray(new String[0])));
        }
        out.add(op);   // post-order: children first, root LAST
        return out.size() - 1;
    }

    // =====================================================================================
    //  Histograms and metrics
    // =====================================================================================
    /** n-gram -> number of distinct paths containing it; trunc > 0 keeps only the first `trunc` n-grams per path. */
    private static Map<String, Integer> hist(final Collection<Rec> recs, final int n, final int trunc) {
        Map<String, Integer> h = new HashMap<>();
        for (Rec r : recs) {
            List<String> g = r.grams(n);
            int m = trunc > 0 ? Math.min(trunc, g.size()) : g.size();
            for (int i = 0; i < m; i++) h.merge(g.get(i), 1, Integer::sum);
        }
        return h;
    }

    private static Map<String, Double> asDouble(final Map<String, Integer> h, final double factor) {
        Map<String, Double> d = new HashMap<>();
        for (Map.Entry<String, Integer> e : h.entrySet()) d.put(e.getKey(), e.getValue() * factor);
        return d;
    }

    /**
     * {MRE over the whole domain, MRE over active bins, Rel50, Rel95}. truth = exact histogram (active bins only);
     * est = released values (missing = 0); zeroMean = expected |noise| on a bin that is zero in the truth
     * (b for the Laplace arm, 0 for the arms that never invent counts).
     */
    private static double[] histMetrics(final Map<String, Integer> truth, final Map<String, Double> est,
                                        final double domain, final double zeroMean) {
        int nb = truth.size();
        if (nb == 0) return new double[]{Double.NaN, Double.NaN, Double.NaN, Double.NaN};
        double[] rel = new double[nb];
        double sum = 0;
        int i = 0;
        for (Map.Entry<String, Integer> e : truth.entrySet()) {
            double x = e.getValue();
            double y = est.getOrDefault(e.getKey(), 0.0);
            double r = Math.abs(x - y) / Math.max(x, 1.0);
            rel[i++] = r;
            sum += r;
        }
        double d = Math.max(domain, nb);
        double mreDomain = (sum + Math.max(0.0, d - nb) * zeroMean) / d;
        return new double[]{mreDomain, sum / nb, DPMetrics.median(rel), DPMetrics.percentile(rel, 0.95)};
    }

    private static double distinctNodeKeys(final List<Op> ops) {
        Set<String> keys = new HashSet<>();
        for (Op op : ops) for (Rec r : op.recs.values()) keys.addAll(Arrays.asList(r.nodeKeys));
        return keys.size();
    }

    // =====================================================================================
    //  One (epsilon, n) evaluation over many trials
    // =====================================================================================
    private static final class NgramEval {
        int n;
        double domain;
        int truthBins, truthRecords, rootNonSensitive, maxBinCount, nsBins, truthEdges, truthNodes;
        double allNsEdgeCov, allNsNodeCov, rrRecall, rrRecallMedian, rrEdgeCov, rrNodeCov;
        final List<String> armNames = new ArrayList<>();
        final List<double[]> metrics = new ArrayList<>();   // per arm: {MREdomain, MREactive, Rel50, Rel95}
        final List<Double> releasedPct = new ArrayList<>(); // per arm, NaN when records are not released
        final List<double[]> perTrialRelError = new ArrayList<>(); // per arm: mean RelError over the active bins, one value per trial
        int kStar;
        RecordReleaseTracker.LedgerStats sensLedger;
        int distinct, multi, maxReq;
        long requests, reused;
        double[] opReleased, opNew, opReused, opMreActive;
    }

    private static double tm(final double[] a) { return DPMetrics.trimmedMean(a, 0.2); }

    private static NgramEval evaluate(final List<Op> ops, final double eps, final int n, final int trials,
                                      final long seed, final long domainNodes, final int maxTrunc) {
        final Op root = ops.get(ops.size() - 1);
        final int nOps = ops.size();
        NgramEval ev = new NgramEval();
        ev.n = n;
        double v = domainNodes > 0 ? domainNodes : distinctNodeKeys(ops);
        ev.domain = Math.pow(Math.max(v, 1), n);
        double p = RecordReleaseTracker.releaseProbability(eps);

        // ground truth (non-private normal execution)
        Map<String, Integer> gt = hist(root.recs.values(), n, 0);
        ev.truthBins = gt.size();
        ev.maxBinCount = gt.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        ev.truthRecords = root.recs.size();

        // all-NS exact (deterministic)
        List<Rec> nsRoot = new ArrayList<>();
        for (Rec r : root.recs.values()) if (!r.sensitive) nsRoot.add(r);
        ev.rootNonSensitive = nsRoot.size();
        ev.nsBins = hist(nsRoot, n, 0).size();

        // coverage of the ground-truth edges / nodes by the released set of paths
        Set<String> tEdges = new HashSet<>(), tNodes = new HashSet<>(), nsEdges = new HashSet<>(), nsNodes = new HashSet<>();
        for (Rec r : root.recs.values()) { tEdges.addAll(Arrays.asList(r.edgeIds)); tNodes.addAll(Arrays.asList(r.nodeIds)); }
        for (Rec r : nsRoot) { nsEdges.addAll(Arrays.asList(r.edgeIds)); nsNodes.addAll(Arrays.asList(r.nodeIds)); }
        ev.truthEdges = tEdges.size();
        ev.truthNodes = tNodes.size();
        ev.allNsEdgeCov = tEdges.isEmpty() ? Double.NaN : nsEdges.size() / (double) tEdges.size();
        ev.allNsNodeCov = tNodes.isEmpty() ? Double.NaN : nsNodes.size() / (double) tNodes.size();
        double[] rrRecall = new double[trials], rrEdgeCov = new double[trials], rrNodeCov = new double[trials];
        double[] mAllNs = histMetrics(gt, asDouble(hist(nsRoot, n, 0), 1.0), ev.domain, 0.0);

        // truncated TRUE histograms for the Laplace arm
        int kmax = 1;
        for (Rec r : root.recs.values()) kmax = Math.max(kmax, r.grams(n).size());
        final int K = Math.min(kmax, Math.max(1, maxTrunc));
        List<Map<String, Integer>> trunc = new ArrayList<>();
        for (int k = 1; k <= K; k++) trunc.add(hist(root.recs.values(), n, k));

        // exact histogram of every operator (for the per-operator view)
        List<Map<String, Integer>> opTruth = new ArrayList<>();
        for (Op op : ops) opTruth.add(hist(op.recs.values(), n, 0));

        double[][] rr = new double[4][trials], rrS = new double[4][trials];
        double[] relPct = new double[trials];
        double[][][] lm = new double[K][4][trials];
        ev.opReleased = new double[nOps];
        ev.opNew = new double[nOps];
        ev.opReused = new double[nOps];
        ev.opMreActive = new double[nOps];

        for (int t = 0; t < trials; t++) {
            // ---- selective operator-level OsdpRR, ONE shared tracker for the whole plan ----
            RecordReleaseTracker tracker = new RecordReleaseTracker(DPMetrics.mix(seed, t, Double.doubleToLongBits(eps)));
            List<Rec> rootReleased = new ArrayList<>();
            for (int oi = 0; oi < nOps; oi++) {
                List<Rec> rel = new ArrayList<>();
                int fresh = 0, again = 0;
                for (Rec r : ops.get(oi).recs.values()) {
                    RecordReleaseTracker.Outcome o = tracker.release(r.id, r.sensitive, eps);
                    if (o.isNew()) fresh++; else again++;
                    if (o.released()) rel.add(r);
                }
                ev.opReleased[oi] += rel.size() / (double) trials;
                ev.opNew[oi] += fresh / (double) trials;
                ev.opReused[oi] += again / (double) trials;
                double[] om = histMetrics(opTruth.get(oi), asDouble(hist(rel, n, 0), 1.0), ev.domain, 0.0);
                if (!Double.isNaN(om[1])) ev.opMreActive[oi] += om[1] / trials;
                if (oi == nOps - 1) rootReleased = rel;
            }
            Map<String, Integer> sample = hist(rootReleased, n, 0);
            double[] a = histMetrics(gt, asDouble(sample, 1.0), ev.domain, 0.0);
            double[] b = histMetrics(gt, asDouble(sample, 1.0 / p), ev.domain, 0.0);
            for (int m = 0; m < 4; m++) { rr[m][t] = a[m]; rrS[m][t] = b[m]; }
            relPct[t] = root.recs.isEmpty() ? Double.NaN : 100.0 * rootReleased.size() / root.recs.size();
            Set<String> re = new HashSet<>(), rn = new HashSet<>();
            for (Rec r : rootReleased) { re.addAll(Arrays.asList(r.edgeIds)); rn.addAll(Arrays.asList(r.nodeIds)); }
            rrRecall[t] = root.recs.isEmpty() ? Double.NaN : rootReleased.size() / (double) root.recs.size();
            rrEdgeCov[t] = tEdges.isEmpty() ? Double.NaN : re.size() / (double) tEdges.size();
            rrNodeCov[t] = tNodes.isEmpty() ? Double.NaN : rn.size() / (double) tNodes.size();
            if (t == 0) {
                ev.sensLedger = tracker.ledger(true);
                ev.distinct = tracker.distinctRecords();
                ev.requests = tracker.totalRequests();
                ev.reused = tracker.reusedRequests();
                ev.multi = tracker.recordsRequestedMoreThanOnce();
                ev.maxReq = tracker.maxRequestsPerRecord();
            }

            // ---- DP post-processing of the final result only: Laplace n-gram histogram, truncation k ----
            SplittableRandom rng = new SplittableRandom(DPMetrics.mix(seed, t, 7000 + n));
            for (int k = 1; k <= K; k++) {
                double scale = 2.0 * k / eps;                       // L1 sensitivity of the truncated histogram = 2k
                Map<String, Integer> th = trunc.get(k - 1);
                Map<String, Double> noisy = new HashMap<>();
                for (String bin : gt.keySet()) {
                    noisy.put(bin, th.getOrDefault(bin, 0) + DPMetrics.laplace(rng, scale));
                }
                double[] c = histMetrics(gt, noisy, ev.domain, scale);
                for (int m = 0; m < 4; m++) lm[k - 1][m][t] = c[m];
            }
        }

        ev.rrRecall = tm(rrRecall);
        ev.rrRecallMedian = DPMetrics.median(rrRecall);
        ev.rrEdgeCov = tm(rrEdgeCov);
        ev.rrNodeCov = tm(rrNodeCov);

        // oracle k for LM T* (by MRE over the domain, the paper's metric)
        int best = 1;
        double bestVal = Double.POSITIVE_INFINITY;
        for (int k = 1; k <= K; k++) {
            double val = tm(lm[k - 1][0]);
            if (val < bestVal) { bestVal = val; best = k; }
        }
        ev.kStar = best;

        ev.armNames.add("all-NS exact [not private]");
        ev.metrics.add(mAllNs);
        ev.releasedPct.add(root.recs.isEmpty() ? Double.NaN : 100.0 * nsRoot.size() / root.recs.size());
        double[] allNsTrials = new double[trials];
        Arrays.fill(allNsTrials, mAllNs[1]);                 // deterministic arm: the same value in every trial
        ev.perTrialRelError.add(allNsTrials);

        ev.armNames.add("selective-op RR (raw)");
        ev.metrics.add(new double[]{tm(rr[0]), tm(rr[1]), tm(rr[2]), tm(rr[3])});
        ev.releasedPct.add(tm(relPct));
        ev.perTrialRelError.add(rr[1]);

        ev.armNames.add("selective-op RR (x 1/p)");
        ev.metrics.add(new double[]{tm(rrS[0]), tm(rrS[1]), tm(rrS[2]), tm(rrS[3])});
        ev.releasedPct.add(tm(relPct));
        ev.perTrialRelError.add(rrS[1]);

        ev.armNames.add("DP-post LM T1");
        ev.metrics.add(new double[]{tm(lm[0][0]), tm(lm[0][1]), tm(lm[0][2]), tm(lm[0][3])});
        ev.releasedPct.add(Double.NaN);
        ev.perTrialRelError.add(lm[0][1]);

        ev.armNames.add("DP-post LM T*(k=" + best + ") [oracle]");
        ev.metrics.add(new double[]{tm(lm[best - 1][0]), tm(lm[best - 1][1]), tm(lm[best - 1][2]), tm(lm[best - 1][3])});
        ev.releasedPct.add(Double.NaN);
        ev.perTrialRelError.add(lm[best - 1][1]);
        return ev;
    }

    // =====================================================================================
    //  Public API
    // =====================================================================================
    /** Defaults: nodes named by id, domain size derived from the plan, truncation search up to k = 10. */
    public static void compare(final String query, final SensitivePatternPolicy pattern, final double[] epsilons,
                               final int[] ngramSizes, final int trials, final long seed) throws Exception {
        compare(query, pattern, epsilons, ngramSizes, trials, seed, NODE_ID, 0, 10);
    }

    /**
     * Releasing-true-data comparison: selective operator-level OSDP-RR (with tracking) vs DP post-processing of the
     * final result vs ground truth.
     *
     * @param nodeKey     how a node is named inside an n-gram (e.g. n -> n.getProperties().get("name"))
     * @param domainNodes number of distinct node keys in the WHOLE graph (domain size = V^n); 0 = derive from the plan
     * @param maxTrunc    largest truncation k searched by the oracle LM T*
     */
    public static void compare(final String query, final SensitivePatternPolicy pattern, final double[] epsilons,
                               final int[] ngramSizes, final int trials, final long seed,
                               final Function<Node, String> nodeKey, final long domainNodes, final int maxTrunc) throws Exception {
        System.out.println("=== TRUE-DATA RELEASE: selective operator-level OSDP-RR (tracked) vs DP post-processing vs ground truth ===");
        System.out.println("query: " + query + "\n");
        List<Op> ops = snapshot(query, pattern, nodeKey);
        Op root = ops.get(ops.size() - 1);
        long sensRoot = root.recs.values().stream().filter(r -> r.sensitive).count();
        System.out.printf("ground truth (non-private execution): %d result paths at the root, %d sensitive; %d operators in the plan%n",
                root.recs.size(), sensRoot, ops.size());
        planOverview(ops);
        int longest = 0;
        for (Rec r : root.recs.values()) longest = Math.max(longest, r.nodeKeys.length);
        printArmLegend();
        System.out.println("Exact release of the non-sensitive part (all-NS) reveals sensitivity by absence (unbounded odds gain);");
        System.out.println("OsdpRR bounds the odds gain by e^eps (OSDP Thm 3.1, independent records assumed).\n");

        double focus = epsilons[0];
        for (double e : epsilons) if (Math.abs(e - 1.0) < Math.abs(focus - 1.0)) focus = e;

        for (double eps : epsilons) {
            System.out.printf("%n################  epsilon = %.3f   release probability of a non-sensitive record p = %.3f  (%d trials)  ################%n",
                    eps, RecordReleaseTracker.releaseProbability(eps), trials);
            printNoiseMap(eps);
            NgramEval first = null;
            for (int n : ngramSizes) {
                if (n > longest) {
                    System.out.printf("%n-- n = %d : SKIPPED. The longest result path has only %d nodes, so no path contains a %d-gram.%n"
                            + "   (choose a query that returns longer paths, e.g. a recursive pattern, or a smaller n.)%n", n, longest, n);
                    continue;
                }
                NgramEval ev = evaluate(ops, eps, n, trials, seed, domainNodes, maxTrunc);
                if (first == null) first = ev;
                System.out.printf("%n-- n = %d : domain d = %.3e possible n-grams, %d active bins (n-grams that occur) from %d records (%d non-sensitive) --%n",
                        n, ev.domain, ev.truthBins, ev.truthRecords, ev.rootNonSensitive);
                System.out.printf("%-34s %10s | %12s %12s %9s %9s%n", "arm", "releases", "MRE(domain)", "MRE(active)", "Rel50", "Rel95");
                System.out.printf("%-34s %10s | %12.4f %12.4f %9.4f %9.4f%n", "ground truth (non-private)", "100.0% rec", 0.0, 0.0, 0.0, 0.0);
                for (int i = 0; i < ev.armNames.size(); i++) {
                    double[] m = ev.metrics.get(i);
                    double rp = ev.releasedPct.get(i);
                    System.out.printf("%-34s %10s | %12.4f %12.4f %9.4f %9.4f%n", ev.armNames.get(i),
                            Double.isNaN(rp) ? "histogram" : String.format("%.1f%% rec", rp), m[0], m[1], m[2], m[3]);
                }
                System.out.printf("   RELERROR (DP4SQL: |y-yhat|/max(1,y), per bin; value of each trial = mean over active bins; %d trials; always-0 baseline = %.1f)%n",
                        trials, DPMetrics.ALWAYS_ZERO_REL_ERROR);
                System.out.printf("   %-34s %8s %8s %8s %8s %8s %9s | %s%n", "arm", "min", "Q1", "median", "Q3", "max", "mean", "vs always-0");
                for (int i = 0; i < ev.armNames.size(); i++) {
                    DPMetrics.BoxStats bx = DPMetrics.boxStats(ev.perTrialRelError.get(i));
                    System.out.printf("   %-34s %8.4f %8.4f %8.4f %8.4f %8.4f %9.4f | %s%n", ev.armNames.get(i), bx.min(), bx.q1(), bx.median(), bx.q3(), bx.max(), bx.mean(),
                            DPMetrics.verdictVsZero(bx.median()));
                }
                if (ev.rootNonSensitive == 0) {
                    System.out.println("   WARNING: every record of the final result is sensitive, so no one-sided arm can release anything (0% released,");
                    System.out.println("   error = always-0). The query is probably anchored at a sensitive entity; use a query whose result mixes sensitive");
                    System.out.println("   and non-sensitive paths, otherwise this experiment cannot show any benefit of selective release.");
                }
                printExpected(ev, eps);
                if (ev.maxBinCount <= 1) {
                    System.out.println("   WARNING: every active bin has count 1, so the histogram IS the record set and the errors only measure the missing");
                    System.out.println("   records (RR raw MRE(active) = 1 - released share). Use longer paths and/or a coarser nodeKey (e.g. a property bucket)");
                    System.out.println("   so that many paths share an n-gram; then the histogram test is informative.");
                }
            }
            if (first == null) {
                System.out.println("\n   No n-gram size could be evaluated for this query.");
            }
            printSetDistance(first, eps);
            printTracking(first, eps);
            if (eps == focus && first != null) printOperators(ops, first, eps);
        }
        System.out.println();
        System.out.println("Reading guide: MRE(domain) is the paper's metric and is dominated by the huge domain (zero bins); MRE(active) and Rel95");
        System.out.println("look only at n-grams that really occur. The DP-post arms release a histogram, never records; all-NS is the non-private");
        System.out.println("upper bound of any one-sided mechanism; 'x 1/p' removes the sampling bias of OsdpRR, sensitive records stay missing.");
        System.out.println("HOW THE ERROR IS MEASURED: always against the FULL ground truth (all paths, sensitive ones included). A path or n-gram that is");
        System.out.println("not released counts as an error; nothing is judged only on the records that were released.\n");
    }

    private static void printArmLegend() {
        System.out.println("WHERE THE NOISE / RANDOMNESS IS, and what happens to the SENSITIVE part");
        System.out.println("  arm              sensitive paths        non-sensitive paths                                additive noise");
        System.out.println("  all-NS exact     removed                all released exactly                               none. NOT private: a missing path reveals it is sensitive");
        System.out.println("  selective-op RR  removed (no coin)      each released only if its coin u < p = 1 - e^-eps  none on the records. The randomness is the COIN (one per path,");
        System.out.println("                                          (one tracked coin per path, shared by operators)   shared by all operators): a missing path may be sensitive OR unlucky");
        System.out.println("  DP-post LM       counted in the bins    counted in the bins                                Laplace(0, 2k/eps) on EVERY histogram bin; no paths released");
        System.out.println("  aggregates (OSDP) not counted           counted, plus one-sided Laplace noise on the COUNT noise goes on the count of non-sensitive records, never on removed ones");
        System.out.println("  Nothing is added to a removed sensitive path: there is nothing left to add noise to. What hides it is that non-sensitive paths");
        System.out.println("  are ALSO dropped at random, so absence alone proves nothing (odds gain <= e^eps instead of unbounded).\n");
    }

    private static void printSetDistance(final NgramEval ev, final double eps) {
        if (ev == null) return;
        final double p = RecordReleaseTracker.releaseProbability(eps);
        final int N = ev.truthRecords, NS = ev.rootNonSensitive, S = N - NS;
        if (N == 0) return;
        System.out.println("   RESULT-SET DISTANCE (paths): how far the released set of paths is from the ground-truth set. Every ground-truth path counts;");
        System.out.println("   a path that is not released is an error. Neither arm ever releases a path that is not in the ground truth (precision = 1),");
        System.out.println("   so Jaccard distance = 1 - |released| / |ground truth| = 1 - recall.");
        System.out.printf("   %-26s %9s %8s %13s | %12s %12s | %9s %9s%n", "arm", "released", "recall", "Jaccard dist", "missing:sens", "missing:coin", "edges cov", "nodes cov");
        double rN = ev.rrRecall;
        System.out.printf("   %-26s %9d %8.4f %13.4f | %12.4f %12.4f | %9.4f %9.4f%n", "all-NS exact [not private]", NS, (double) NS / N, 1.0 - (double) NS / N,
                (double) S / N, 0.0, ev.allNsEdgeCov, ev.allNsNodeCov);
        System.out.printf("   %-26s %9.2f %8.4f %13.4f | %12.4f %12.4f | %9.4f %9.4f%n", "selective-op RR", rN * N, rN, 1.0 - rN,
                (double) S / N, Math.max(0.0, 1.0 - rN - (double) S / N), ev.rrEdgeCov, ev.rrNodeCov);
        System.out.printf("   %-26s %9s%n", "DP-post LM", "no paths (histogram only): the path-set distance does not apply");
        System.out.printf("   formulas: recall = |released|/N;  missing:sens = S/N = %d/%d = %.4f (always lost);  missing:coin = (1-p)*NS/N = %.4f;%n",
                S, N, S / (double) N, (1.0 - p) * NS / N);
        System.out.printf("             E[recall] = p*NS/N = %.5f * %d / %d = %.4f   (observed trimmed mean %.4f, median %.4f over the trials)%n",
                p, NS, N, p * NS / N, ev.rrRecall, ev.rrRecallMedian);
        System.out.printf("             edge coverage = share of the %d distinct ground-truth edges that appear in some released path; node coverage: %d nodes%n",
                ev.truthEdges, ev.truthNodes);
    }

    private static void printNoiseMap(final double eps) {
        double p = RecordReleaseTracker.releaseProbability(eps);
        System.out.printf("WHERE IS THE NOISE?  What each arm does with the sensitive part and with the rest   (p = 1 - e^-eps = %.4f)%n", p);
        System.out.printf("  %-18s | %-26s | %-33s | %s%n", "arm", "SENSITIVE paths", "NON-SENSITIVE paths", "where the randomness / noise is");
        System.out.printf("  %-18s | %-26s | %-33s | %s%n", "all-NS exact", "removed", "all released, exactly", "nowhere: NOT private, a missing path reveals that it was sensitive");
        System.out.printf("  %-18s | %-26s | %-33s | %s%n", "selective-op RR", "removed, never released", "each released only with prob. p",
                "in WHICH non-sensitive paths are dropped; nothing is added to sensitive data (it is not released)");
        System.out.printf("  %-18s | %-26s | %-33s | %s%n", "DP-post LM T1", "counted in the histogram", "counted in the histogram",
                "Laplace(2k/eps) on every histogram count; no path is released at all");
        System.out.println("  Why drop non-sensitive paths at random: if all of them were released, every path missing from the result would have to be sensitive,");
        System.out.println("  so the removal itself would reveal it (exclusion attack). With the coin a missing path may be sensitive or just unlucky, and the");
        System.out.println("  odds that a given missing path is sensitive change by at most a factor e^eps (OSDP Thm 3.1, independent records assumed).");
        System.out.println("  The price is utility, and it is measured against the FULL ground truth: the result loses (a) the sensitive paths (policy, 'missing:sens')");
        System.out.println("  and (b) a share 1-p of the non-sensitive ones (privacy, 'missing:coin'). Both appear in the RESULT-SET DISTANCE table below.\n");
    }

    private static void printExpected(final NgramEval ev, final double eps) {
        final double p = RecordReleaseTracker.releaseProbability(eps);
        final int N = ev.truthRecords, NS = ev.rootNonSensitive, B = ev.truthBins;
        final double b = 2.0 / eps;
        System.out.println("   HOW THESE NUMBERS ARISE (formulas evaluated for this run; explain(...) gives the record-by-record trace)");
        System.out.printf("     N = %d root records = %d sensitive + %d non-sensitive;  p = 1 - e^-eps = %.5f;  B = %d active bins;  d = %.3e%n",
                N, N - NS, NS, p, B, ev.domain);
        System.out.printf("     all-NS exact    : releases NS/N = %d/%d = %.1f%% of the records; the %d sensitive records are missing%n",
                NS, N, N == 0 ? 0.0 : 100.0 * NS / N, N - NS);
        System.out.printf("     selective-op RR : a non-sensitive record is released with probability p, so E[released] = p*NS = %.5f * %d = %.4f records = %.2f%% of N%n",
                p, NS, p * NS, N == 0 ? 0.0 : 100.0 * p * NS / N);
        if (ev.maxBinCount <= 1 && B > 0) {
            double allNs = 1.0 - (double) ev.nsBins / B;
            double raw = 1.0 - p * ev.nsBins / B;
            double scaled = 1.0 + ev.nsBins * (1.0 - 2.0 * p) / B;
            System.out.printf("       (every bin holds one record; %d of the B = %d bins come from non-sensitive records)%n", ev.nsBins, B);
            System.out.printf("       all-NS  : MRE(active) = 1 - NS_bins/B = %.4f%n", allNs);
            System.out.printf("       RR raw  : E[MRE(active)] = 1 - p*NS_bins/B = %.4f   (every missing bin has RelError 1)%n", raw);
            System.out.printf("       RR x1/p : E[MRE(active)] = 1 + NS_bins*(1-2p)/B = %.4f   (a released bin is estimated 1/p = %.2f, RelError |1-1/p| = %.2f)%n",
                    scaled, 1.0 / p, Math.abs(1.0 - 1.0 / p));
            System.out.println("                 this is a MEAN over trials and is dominated by rare trials in which a record gets through; the MEDIAN is exactly 1");
            System.out.println("                 whenever usually nothing is released (p*NS < 1), which is why mean and median differ in the table above.");
        } else {
            System.out.println("       per bin g: E[H_RR(g)] = p * H_NS(g); RR raw is biased low by (1-p)*H_NS(g) plus the missing sensitive part H(g)-H_NS(g); x1/p removes only the first bias.");
        }
        System.out.printf("     DP-post LM T1   : noise Laplace(0, b), b = 2k/eps = 2*1/%.4f = %.2f on EVERY bin of the domain, so MRE(domain) ~ b = %.2f (zero bins dominate)%n", eps, b, b);
        System.out.printf("       for a bin with H = 1: median RelError = b ln 2 = %.2f, 95th percentile = b ln 20 = %.2f, mean = b = %.2f%n", b * Math.log(2), b * Math.log(20), b);
        System.out.println("     always-0 reference: RelError = 1 per bin; an arm whose median RelError is >= 1 is no better than answering 0.");
    }

    private static void printTracking(final NgramEval ev, final double eps) {
        if (ev == null) return;
        RecordReleaseTracker.LedgerStats s = ev.sensLedger;
        System.out.printf("%nTRACKING (selective operator-level RR, eps = %.3f)%n", eps);
        System.out.printf("  distinct records tracked: %d   release requests: %d   reused decisions: %d (%.1f%%)%n",
                ev.distinct, ev.requests, ev.reused, ev.requests == 0 ? 0.0 : 100.0 * ev.reused / ev.requests);
        System.out.printf("  records emitted by more than one operator: %d   (max operators per record: %d)%n", ev.multi, ev.maxReq);
        System.out.printf("  composed epsilon per SENSITIVE record  [%d records]%n", s.records());
        System.out.printf("    with tracking   : max %.3f  mean %.3f   (one decision per record)%n", s.maxTracked(), s.meanTracked());
        System.out.printf("    without tracking: max %.3f  mean %.3f   (a fresh coin per operator)%n", s.maxUntracked(), s.meanUntracked());
    }

    private static void printOperators(final List<Op> ops, final NgramEval ev, final double eps) {
        System.out.printf("%nPER-OPERATOR view at eps = %.3f, n = %d (mean over trials; each operator releases its own exact output)%n", eps, ev.n);
        System.out.printf("%-3s %-26s %8s %10s %10s %8s %8s %8s %12s%n", "lvl", "operator", "records", "sensitive", "released", "recall", "NEW", "REUSED", "MRE(active)");
        for (int i = 0; i < ops.size(); i++) {
            Op op = ops.get(i);
            long sens = op.recs.values().stream().filter(r -> r.sensitive).count();
            System.out.printf("%-3d %-26s %8d %10d %10.1f %8.4f %8.1f %8.1f %12.4f%n", op.depth, op.label, op.recs.size(), sens,
                    ev.opReleased[i], op.recs.isEmpty() ? Double.NaN : ev.opReleased[i] / op.recs.size(), ev.opNew[i], ev.opReused[i], ev.opMreActive[i]);
        }
    }

    // =====================================================================================
    //  EXPLAIN: where selective DP intervenes, record by record
    // =====================================================================================
    private static void planOverview(final List<Op> ops) {
        System.out.println("PLAN (execution order, children first). EVERY operator output is a release point where selective DP intervenes:");
        System.out.printf("  %-4s %-30s %-6s %-10s %9s %10s%n", "idx", "operator", "level", "children", "records", "sensitive");
        for (int i = 0; i < ops.size(); i++) {
            Op op = ops.get(i);
            long sens = op.recs.values().stream().filter(r -> r.sensitive).count();
            String kids = op.children.isEmpty() ? "-" : op.children.toString();
            System.out.printf("  %-4d %-30s %-6d %-10s %9d %10d%s%n", i, op.label, op.depth, kids, op.recs.size(), sens,
                    i == ops.size() - 1 ? "   <- ROOT (final result)" : "");
        }
        if (ops.size() == 1) {
            System.out.println("  NOTE: the plan has a single operator, so there are no intermediate operators: tracking has nothing to reuse and");
            System.out.println("        every record is released exactly once. Use a query with joins / recursion to see intermediate releases.");
        }
        System.out.println();
    }

    private static String pathText(final Rec r) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < r.nodeNames.length; i++) {
            sb.append(r.nodeNames[i]);
            if (i < r.edgeLabels.length) sb.append("-[").append(r.edgeLabels[i]).append("]->");
        }
        return sb.toString();
    }

    private static String edgeFlags(final Rec r) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < r.edgeSensitive.length; i++) sb.append(i == 0 ? "" : ",").append(r.edgeSensitive[i] ? "S" : "-");
        return sb.toString();
    }

    private static String clip(final String s, final int w) { return s.length() <= w ? s : s.substring(0, w - 3) + "..."; }

    private static String describeOperator(final String label) {
        switch (label) {
            case "LogicalOpSelectionByLabel":
                return "selection by label: returns the edges that carry the requested label (a base scan; each edge is a 1-edge path)";
            case "LogicalOpSelection":
                return "selection: keeps the paths of its child that satisfy a condition (its records are a subset of the child's records)";
            case "LogicalOpNodeJoin":
                return "node join: concatenates a path of the left child with a path of the right child that meet at the same node";
            default:
                return "(no description registered for this operator class)";
        }
    }

    private static String whySensitive(final Rec r) {
        List<String> hits = new ArrayList<>();
        for (int i = 0; i < r.edgeSensitive.length; i++) {
            if (r.edgeSensitive[i]) {
                String ends = (i + 1 < r.nodeNames.length) ? r.nodeNames[i] + " - " + r.nodeNames[i + 1] : "";
                hits.add("edge " + r.edgeIds[i] + " [" + r.edgeLabels[i] + "] " + ends);
            }
        }
        return hits.isEmpty() ? "path-level rule (isSensitivePath = true, no single edge flagged)" : String.join("; ", hits);
    }

    private static final class Info {
        final Rec r;
        final double u;
        final boolean released;
        final boolean isNew;
        final int decidedAt;
        Info(final Rec r, final double u, final boolean released, final boolean isNew, final int decidedAt) {
            this.r = r; this.u = u; this.released = released; this.isNew = isNew; this.decidedAt = decidedAt;
        }
    }

    private static final int FINAL_ROWS_CAP = 200;
    private static final int BIN_ROWS_CAP = 15;

    /** explain with up to 25 records printed per category and operator. */
    public static void explain(final String query, final SensitivePatternPolicy pattern, final double eps, final int n,
                               final long seed, final Function<Node, String> nodeKey) throws Exception {
        explain(query, pattern, eps, n, 25, seed, nodeKey);
    }

    /**
     * One trial, shown completely: what is sensitive and why, the formulas, every operator with its records split into
     * SENSITIVE / NON-SENSITIVE and the decision taken for each, the final query result, and the n-gram values with the
     * per-bin arithmetic behind every metric.
     *
     * @param rowsPerGroup records printed per category (sensitive / non-sensitive) per operator; the rest is counted
     */
    public static void explain(final String query, final SensitivePatternPolicy pattern, final double eps, final int n,
                               final int rowsPerGroup, final long seed, final Function<Node, String> nodeKey) throws Exception {
        List<Op> ops = snapshot(query, pattern, nodeKey);
        final double p = RecordReleaseTracker.releaseProbability(eps);
        final long tseed = DPMetrics.mix(seed, 0, Double.doubleToLongBits(eps));
        RecordReleaseTracker tracker = new RecordReleaseTracker(tseed);
        final int last = ops.size() - 1;
        Op root = ops.get(last);

        int longest = 0;
        for (Rec r : root.recs.values()) longest = Math.max(longest, r.nodeKeys.length);
        final int nn = Math.min(n, Math.max(longest, 1));
        final double V = distinctNodeKeys(ops);
        final double d = Math.pow(Math.max(V, 1), nn);
        final int N = root.recs.size();
        final int NS = (int) root.recs.values().stream().filter(r -> !r.sensitive).count();
        final Map<String, Integer> gt = hist(root.recs.values(), nn, 0);
        final int B = gt.size();

        System.out.printf("=== EXPLAIN (eps = %.4f, n = %d) ===%n", eps, nn);
        System.out.println("query: " + query + "\n");

        System.out.println("1. WHAT IS SENSITIVE, AND WHAT HAPPENS TO IT");
        System.out.println("   record      : one result path of an operator (what the operator returns).");
        System.out.println("   SENSITIVE   : pattern.isSensitivePath(path) is true. 'edges' marks each edge with isSensitiveEdge (S = sensitive, - = not).");
        System.out.println("   NON-SENSITIVE: everything else.");
        System.out.println("   INTERVENTION: after an operator computes its EXACT output, each record passes through OsdpRR:");
        System.out.println("                 sensitive      -> never released (SUPPRESSED, no coin is tossed);");
        System.out.println("                 non-sensitive  -> released only if its coin u < p (otherwise SUPPRESSED by the coin).");
        System.out.println("   TRACKING    : the coin u of a record is fixed once (by seed and record id). If another operator emits the same record it gets");
        System.out.println("                 the same decision (REUSED), so the record costs eps once, however many operators emit it.");
        System.out.println("   FINAL RESULT: the root operator's released records. Sensitive records are always missing from it.");
        System.out.println("   WHERE IS THE NOISE: nothing is added to a removed sensitive path (there is nothing left to noise). The randomness is the coin that");
        System.out.println("                 also drops non-sensitive paths at random, so a missing path may be sensitive OR unlucky. Additive Laplace noise only");
        System.out.println("                 appears in the DP-post arm (on histogram bins) and in OSDP's aggregate answers (on the count of non-sensitive records).");
        System.out.println("   HOW ERROR IS MEASURED: against the FULL ground truth; every path or n-gram that is not released counts as an error.\n");
        printNoiseMap(eps);

        System.out.println("2. FORMULAS USED (evaluated with this run's numbers)");
        System.out.printf("   p = 1 - e^(-eps) = 1 - e^(-%.4f) = %.5f            release probability of a NON-sensitive record%n", eps, p);
        System.out.println("   decision(r) = SUPPRESSED if r is sensitive; otherwise RELEASED iff u(r) < p, with u(r) = hash(seed, id(r)) in [0,1)");
        System.out.println("   E[#released at an operator] = p * (#non-sensitive records at that operator)");
        System.out.println("   n-gram of a path = n consecutive node keys;  H(g) = number of result paths that contain the n-gram g (once per path)");
        System.out.println("   ground truth H           : H over all root records");
        System.out.println("   all-NS exact (not DP)    : H_NS(g) = H over the non-sensitive root records only");
        System.out.println("   selective-op RR (raw)    : H_RR(g) = H over the released root records");
        System.out.printf ("   selective-op RR (x 1/p)  : H_RR(g) / p = H_RR(g) / %.5f   (inverse-probability scaling)%n", p);
        System.out.printf ("   DP-post LM T1            : H_1(g) + Laplace(0, b), b = 2k/eps = 2*1/%.4f = %.2f   (H_1 keeps 1 n-gram per path, L1 sensitivity 2k)%n", eps, 2.0 / eps);
        System.out.println("   RelError(g)              : |H(g) - Hhat(g)| / max(1, H(g))                       (DP4SQL)");
        System.out.println("   MRE(active)              : (1/B) * sum over the B active bins of RelError(g)");
        System.out.printf ("   MRE(domain)              : [ sum_active RelError + (d - B) * E|noise on a zero bin| ] / d,   d = V^n = %.0f^%d = %.3e%n", V, nn, d);
        System.out.println("                              E|noise| = b for LM, 0 for the other arms (they never invent counts)");
        System.out.println("   always-0 reference       : RelError = 1 for every bin with H(g) >= 1");
        System.out.println("   composed epsilon         : eps x (number of independent coins a record gets); tracked = eps, untracked = (#operators emitting it) x eps");
        System.out.printf ("   this run                 : N = %d root records = %d sensitive + %d non-sensitive; B = %d active bins%n%n", N, N - NS, NS, B);

        System.out.println("3. PLAN");
        planOverview(ops);
        for (int i = 0; i < ops.size(); i++) System.out.printf("   [%d] %s : %s%n", i, ops.get(i).label, describeOperator(ops.get(i).label));

        System.out.println("\n4. OPERATOR BY OPERATOR (the release point after each operator)");
        List<List<Info>> infos = new ArrayList<>();
        Map<String, Integer> firstSeen = new HashMap<>();
        for (int oi = 0; oi < ops.size(); oi++) {
            Op op = ops.get(oi);
            List<Info> all = new ArrayList<>();
            for (Rec r : op.recs.values()) {
                RecordReleaseTracker.Outcome o = tracker.release(r.id, r.sensitive, eps);
                firstSeen.putIfAbsent(r.id, oi);
                all.add(new Info(r, RecordReleaseTracker.unit(tseed, r.id), o.released(), o.isNew(), firstSeen.get(r.id)));
            }
            infos.add(all);
            List<Info> sensL = new ArrayList<>(), nsL = new ArrayList<>();
            for (Info in : all) (in.r.sensitive ? sensL : nsL).add(in);
            long released = nsL.stream().filter(in -> in.released).count();
            long coinOut = nsL.size() - released;
            long fresh = all.stream().filter(in -> in.isNew).count();

            System.out.println("\n" + "=".repeat(112));
            System.out.printf("OPERATOR [%d] %s (level %d)%s%n", oi, op.label, op.depth, oi == last ? "   <- ROOT, its released set is the FINAL RESULT" : "");
            System.out.println("  meaning : " + describeOperator(op.label));
            for (int c : op.children) {
                long shared = op.recs.keySet().stream().filter(id -> ops.get(c).recs.containsKey(id)).count();
                System.out.printf("  input   : child [%d] %s emitted %d records; %d of this operator's %d records are the SAME records (decisions reused)%n",
                        c, ops.get(c).label, ops.get(c).recs.size(), shared, op.recs.size());
            }
            if (op.children.isEmpty()) System.out.println("  input   : none (base operator)");
            System.out.printf("  EXACT OUTPUT : %d records = %d SENSITIVE + %d NON-SENSITIVE%n", all.size(), sensL.size(), nsL.size());

            System.out.printf("%n  (A) SENSITIVE records (%d): rule = SUPPRESSED, no coin%n", sensL.size());
            if (!sensL.isEmpty()) System.out.printf("      %-4s %-46s %-9s %-52s %s%n", "#", "path", "edges", "why it is sensitive", "tracker");
            for (int i = 0; i < Math.min(rowsPerGroup, sensL.size()); i++) {
                Info in = sensL.get(i);
                System.out.printf("      %-4d %-46s %-9s %-52s %s%n", i + 1, clip(pathText(in.r), 46), edgeFlags(in.r),
                        clip(whySensitive(in.r), 52), in.isNew ? "NEW" : "REUSED (decided at [" + in.decidedAt + "])");
            }
            if (sensL.size() > rowsPerGroup) System.out.printf("      ... %d more sensitive records not shown (raise rowsPerGroup)%n", sensL.size() - rowsPerGroup);

            System.out.printf("%n  (B) NON-SENSITIVE records (%d): rule = RELEASED iff u < p = %.5f   (sorted by u)%n", nsL.size(), p);
            nsL.sort((x, y) -> Double.compare(x.u, y.u));
            if (!nsL.isEmpty()) System.out.printf("      %-4s %-46s %-9s %8s %-6s %-22s %s%n", "#", "path", "edges", "u", "u<p?", "decision", "tracker");
            for (int i = 0; i < Math.min(rowsPerGroup, nsL.size()); i++) {
                Info in = nsL.get(i);
                System.out.printf("      %-4d %-46s %-9s %8.5f %-6s %-22s %s%n", i + 1, clip(pathText(in.r), 46), edgeFlags(in.r), in.u,
                        in.u < p ? "yes" : "no", in.released ? "RELEASED" : "SUPPRESSED (coin)",
                        in.isNew ? "NEW" : "REUSED (decided at [" + in.decidedAt + "])");
            }
            if (nsL.size() > rowsPerGroup) System.out.printf("      ... %d more non-sensitive records not shown%n", nsL.size() - rowsPerGroup);

            System.out.printf("%n  RESULT OF THE RELEASE AT THIS OPERATOR%n");
            System.out.printf("      released = #{non-sensitive r : u(r) < p} = %d      suppressed: %d sensitive + %d by the coin = %d%n",
                    released, sensL.size(), coinOut, sensL.size() + coinOut);
            System.out.printf("      expected released E = p * NS = %.5f * %d = %.3f      observed = %d%n", p, nsL.size(), p * nsL.size(), released);
            System.out.printf("      recall at this operator = released / records = %d / %d = %.4f   (missing: %d sensitive + %d by the coin)%n",
                    released, all.size(), all.isEmpty() ? Double.NaN : released / (double) all.size(), sensL.size(), coinOut);
            System.out.printf("      tracker: %d decisions NEW, %d REUSED%n", fresh, all.size() - fresh);
        }

        // ---------------- final result ----------------
        List<Info> rootInfos = infos.get(last);
        System.out.println("\n" + "=".repeat(112));
        System.out.println("5. FINAL QUERY RESULT (root operator [" + last + "])");
        System.out.printf("   ground truth = %d result paths (normal, non-private execution).%n", N);
        System.out.printf("   %-4s %-11s %-9s %-26s %s%n", "#", "sensitive", "all-NS", "selective-op RR", "result path (printed in full)");
        int shownRows = 0;
        for (Info in : rootInfos) {
            if (shownRows++ >= FINAL_ROWS_CAP) break;
            String rr = in.r.sensitive ? "SUPPRESSED (sensitive)" : (in.released ? "RELEASED (u=" + String.format("%.4f", in.u) + ")" : "suppressed (u=" + String.format("%.4f", in.u) + ")");
            System.out.printf("   %-4d %-11s %-9s %-26s %s%n", shownRows, in.r.sensitive ? "YES" : "no",
                    in.r.sensitive ? "dropped" : "released", rr, pathText(in.r));
        }
        if (rootInfos.size() > FINAL_ROWS_CAP) System.out.printf("   ... %d more result paths not shown%n", rootInfos.size() - FINAL_ROWS_CAP);
        long rrReleased = rootInfos.stream().filter(in -> in.released).count();
        System.out.printf("   what each arm outputs as the final result:%n");
        System.out.printf("     ground truth        : %d paths%n", N);
        System.out.printf("     all-NS exact        : %d paths (all %d non-sensitive ones; the %d sensitive ones are dropped)  [not private]%n", NS, NS, N - NS);
        System.out.printf("     selective-op RR     : %d paths (E = p * NS = %.3f); sensitive never; non-sensitive only if u < p%n", rrReleased, p * NS);
        System.out.println("     DP-post LM          : no paths at all, only the noisy n-gram histogram printed below");

        System.out.println("\n   5b. DISTANCE OF THE FINAL RESULT TO THE GROUND TRUTH (set of paths; every ground-truth path counts)");
        if (N > 0) {
            int coinMissing = NS - (int) rrReleased;
            Set<String> tE = new HashSet<>(), tN = new HashSet<>(), rE = new HashSet<>(), rN = new HashSet<>(), nE = new HashSet<>(), nN = new HashSet<>();
            for (Info in : rootInfos) {
                tE.addAll(Arrays.asList(in.r.edgeIds)); tN.addAll(Arrays.asList(in.r.nodeIds));
                if (in.released) { rE.addAll(Arrays.asList(in.r.edgeIds)); rN.addAll(Arrays.asList(in.r.nodeIds)); }
                if (!in.r.sensitive) { nE.addAll(Arrays.asList(in.r.edgeIds)); nN.addAll(Arrays.asList(in.r.nodeIds)); }
            }
            System.out.printf("     selective-op RR : recall = released / N = %d / %d = %.4f;  Jaccard distance = 1 - recall = %.4f%n", rrReleased, N, rrReleased / (double) N, 1.0 - rrReleased / (double) N);
            System.out.printf("                       missing = %d sensitive + %d by the coin = %d paths = %.4f + %.4f of the ground truth%n", N - NS, coinMissing, N - rrReleased,
                    (N - NS) / (double) N, coinMissing / (double) N);
            System.out.printf("                       edge coverage %d / %d = %.4f, node coverage %d / %d = %.4f%n", rE.size(), tE.size(), rE.size() / (double) Math.max(1, tE.size()),
                    rN.size(), tN.size(), rN.size() / (double) Math.max(1, tN.size()));
            System.out.printf("     all-NS exact    : recall = NS / N = %d / %d = %.4f;  Jaccard distance = %.4f;  edge coverage %.4f, node coverage %.4f%n", NS, N, NS / (double) N,
                    1.0 - NS / (double) N, nE.size() / (double) Math.max(1, tE.size()), nN.size() / (double) Math.max(1, tN.size()));
            System.out.printf("     expected recall of selective-op RR = p * NS / N = %.5f * %d / %d = %.4f%n", p, NS, N, p * NS / N);
        }

        // ---------------- values and metrics ----------------
        System.out.printf("%n6. %d-GRAM VALUES AND THE ARITHMETIC BEHIND THE METRICS%n", nn);
        if (nn < n) System.out.printf("   (n reduced from %d to %d: the longest result path has only %d nodes)%n", n, nn, longest);
        List<Rec> nsRoot = new ArrayList<>(), relRoot = new ArrayList<>();
        for (Info in : rootInfos) { if (!in.r.sensitive) nsRoot.add(in.r); if (in.released) relRoot.add(in.r); }
        Map<String, Integer> hNs = hist(nsRoot, nn, 0), hRr = hist(relRoot, nn, 0), h1 = hist(root.recs.values(), nn, 1);
        List<String> bins = new ArrayList<>(gt.keySet());
        bins.sort((x, y) -> gt.get(x).equals(gt.get(y)) ? x.compareTo(y) : Integer.compare(gt.get(y), gt.get(x)));
        if (B == 0) { System.out.println("   no n-grams: nothing to compute."); return; }

        double sumNs = 0, sumRaw = 0, sumScaled = 0, sumLm = 0;
        final double b = 2.0 / eps;
        SplittableRandom rng = new SplittableRandom(DPMetrics.mix(seed, 0, 7000 + nn));
        double[] relLm = new double[B];
        double[] noiseDraw = new double[B], lmOut = new double[B];
        for (int i = 0; i < B; i++) {
            String g = bins.get(i);
            double x = gt.get(g);
            sumNs += Math.abs(x - hNs.getOrDefault(g, 0)) / Math.max(1, x);
            sumRaw += Math.abs(x - hRr.getOrDefault(g, 0)) / Math.max(1, x);
            sumScaled += Math.abs(x - hRr.getOrDefault(g, 0) / p) / Math.max(1, x);
            noiseDraw[i] = DPMetrics.laplace(rng, b);
            lmOut[i] = h1.getOrDefault(g, 0) + noiseDraw[i];
            relLm[i] = Math.abs(x - lmOut[i]) / Math.max(1, x);
            sumLm += relLm[i];
        }
        System.out.printf("   (a) selective-op RR and all-NS: per-bin counts for the %d most frequent of the B = %d active bins%n", Math.min(BIN_ROWS_CAP, B), B);
        System.out.printf("       %-34s %7s %7s %7s %9s | %10s %10s%n", "n-gram", "H", "H_NS", "H_RR", "H_RR/p", "RelErr raw", "RelErr /p");
        for (int i = 0; i < Math.min(BIN_ROWS_CAP, B); i++) {
            String g = bins.get(i);
            double x = gt.get(g);
            System.out.printf("       %-34s %7d %7d %7d %9.2f | %10.4f %10.4f%n", clip(g, 34), gt.get(g), hNs.getOrDefault(g, 0), hRr.getOrDefault(g, 0),
                    hRr.getOrDefault(g, 0) / p, Math.abs(x - hRr.getOrDefault(g, 0)) / Math.max(1, x), Math.abs(x - hRr.getOrDefault(g, 0) / p) / Math.max(1, x));
        }
        System.out.printf("       totals over ALL %d bins: sum RelError  all-NS = %.4f   RR raw = %.4f   RR x1/p = %.4f%n", B, sumNs, sumRaw, sumScaled);
        System.out.printf("       MRE(active) = sum / B :  all-NS = %.4f / %d = %.4f   RR raw = %.4f / %d = %.4f   RR x1/p = %.4f / %d = %.4f%n",
                sumNs, B, sumNs / B, sumRaw, B, sumRaw / B, sumScaled, B, sumScaled / B);
        System.out.printf("       MRE(domain) = sum / d :  all-NS = %.3e   RR raw = %.3e   RR x1/p = %.3e   (d = %.3e, zero bins add 0)%n",
                sumNs / d, sumRaw / d, sumScaled / d, d);

        System.out.printf("%n   (b) DP-post LM T1: noisy histogram, ONE illustrative draw, b = 2k/eps = %.2f%n", b);
        System.out.printf("       %-34s %7s %7s %12s %12s | %10s%n", "n-gram", "H", "H_1", "noise", "output", "RelError");
        for (int i = 0; i < Math.min(BIN_ROWS_CAP, B); i++) {
            String g = bins.get(i);
            System.out.printf("       %-34s %7d %7d %12.3f %12.3f | %10.4f%n", clip(g, 34), gt.get(g), h1.getOrDefault(g, 0), noiseDraw[i], lmOut[i], relLm[i]);
        }
        double zeroPart = Math.max(0.0, d - B) * b;
        System.out.printf("       MRE(active) = sum_active RelError / B = %.4f / %d = %.4f%n", sumLm, B, sumLm / B);
        System.out.printf("       MRE(domain) = [sum_active + (d - B) * b] / d = [%.4f + (%.3e - %d) * %.2f] / %.3e = %.4f%n", sumLm, d, B, b, d, (sumLm + zeroPart) / d);
        System.out.printf("       every one of the d - B zero bins also receives Laplace noise (E|noise| = b = %.2f); they are not materialised, their expectation is added.%n", b);
        System.out.printf("       reference: for a bin with H = 1, median RelError = b ln 2 = %.2f, 95th percentile = b ln 20 = %.2f%n", b * Math.log(2), b * Math.log(20));
        double[] rel = relLm.clone();
        System.out.printf("       this draw: Rel50 = %.3f, Rel95 = %.3f over the %d active bins%n", DPMetrics.median(rel), DPMetrics.percentile(rel, 0.95), B);

        int maxBin = gt.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        if (maxBin <= 1) {
            System.out.println("\n   NOTE: every active bin has count 1, so each bin is a single record. Then RR raw MRE(active) = 1 - (released bins)/B and the");
            System.out.println("   histogram only measures which records got through. Use longer paths and/or a coarser nodeKey for an informative test.");
        }
        if (NS == 0) System.out.println("\n   NOTE: every record of the final result is sensitive: nothing can be released as true data.");

        // ---------------- tracker check ----------------
        System.out.println("\n7. TRACKER CHECK: ask again for records that were already decided (what a parent operator does)");
        int shown = 0;
        for (Rec r : root.recs.values()) {
            if (shown++ >= 3) break;
            RecordReleaseTracker.Outcome again = tracker.release(r.id, r.sensitive, eps);
            System.out.printf("   %-62s second request -> %s, decision unchanged (released=%b)%n", clip(pathText(r), 62), again.isNew() ? "NEW" : "REUSED", again.released());
        }
        RecordReleaseTracker.LedgerStats ls = tracker.ledger(true);
        System.out.printf("   records tracked: %d, requests: %d, reused: %d%n", tracker.distinctRecords(), tracker.totalRequests(), tracker.reusedRequests());
        System.out.printf("   composed epsilon per SENSITIVE record: tracked max %.3f / untracked max %.3f (untracked = #operators emitting the record x eps)%n%n",
                ls.maxTracked(), ls.maxUntracked());
    }

    // =====================================================================================
    //  Sweep of the sensitive share (OSDP's non-sensitive ratio P99 ... P1)
    // =====================================================================================
    /**
     * Re-labels sensitivity at EDGE level (edge sensitive iff a seeded coin < fraction) and calls a path sensitive
     * when any of its edges is, so a record's label is consistent in every operator. Prints, for each fraction,
     * the release rate and the n-gram errors of the arms, to show where selective release stops paying off.
     */
    public static void sweepSensitiveEdges(final String query, final SensitivePatternPolicy pattern,
                                           final double[] fractions, final double eps, final int n,
                                           final int trials, final long seed,
                                           final Function<Node, String> nodeKey, final long domainNodes,
                                           final int maxTrunc) throws Exception {
        System.out.println("=== SWEEP of the sensitive-edge share (eps = " + eps + ", n = " + n + ", " + trials + " trials) ===");
        System.out.println("query: " + query + "\n");
        List<Op> ops = snapshot(query, pattern, nodeKey);
        Op root = ops.get(ops.size() - 1);
        System.out.printf("%6s %12s %9s | %-44s | %-44s%n", "sens%", "root non-sens%", "released%",
                "MRE(domain): all-NS / RR raw / RR x1/p / LM T1", "MRE(active): all-NS / RR raw / RR x1/p / LM T1");
        for (double f : fractions) {
            for (Op op : ops) {
                for (Rec r : op.recs.values()) {
                    boolean s = false;
                    for (String id : r.edgeIds) {
                        if (RecordReleaseTracker.unit(DPMetrics.mix(seed, 31, 0), id) < f) { s = true; break; }
                    }
                    r.sensitive = s;
                }
            }
            long ns = root.recs.values().stream().filter(r -> !r.sensitive).count();
            NgramEval ev = evaluate(ops, eps, n, trials, seed, domainNodes, maxTrunc);
            double[] a = ev.metrics.get(0), b = ev.metrics.get(1), c = ev.metrics.get(2), d = ev.metrics.get(3);
            System.out.printf("%5.0f%% %13.1f%% %8.1f%% | %10.4f %10.4f %10.4f %10.4f | %10.4f %10.4f %10.4f %10.4f%n",
                    100 * f, root.recs.isEmpty() ? 0.0 : 100.0 * ns / root.recs.size(), ev.releasedPct.get(1),
                    a[0], b[0], c[0], d[0], a[1], b[1], c[1], d[1]);
        }
        System.out.println();
        System.out.println("At 100% sensitive nothing can be released as true data (RR error = the whole histogram); the DP-post histogram does");
        System.out.println("not depend on the share. The crossover where RR (x 1/p) falls behind LM T1 on the active bins is the number to report.\n");
    }

    // =====================================================================================
    //  Example
    // =====================================================================================
    public static void main(String[] args) throws Exception {
        SensitivePatternPolicy pattern = new SensitivePatternPolicy();
        pattern.addSensitiveNodeName("President");

        // Use a query that returns paths of several hops, so the plan has intermediate operators and the n-grams are not trivial.
        String query = "MATCH p = (x)-[knows+]{..4}->(y) WHERE x.name = \"Alice\" RETURN p LIMIT 100000;";

        // A categorical node key makes many paths share an n-gram (a histogram over decades instead of over unique ids).
        Function<Node, String> decade = nd -> {
            String y = (nd.getProperties() == null) ? null : nd.getProperties().get("birthYear");
            return (y != null && y.length() >= 4) ? y.substring(0, 3) + "0s" : String.valueOf(nd.getLabel());
        };

        Tools.loadDefaultGraph();
        explain(query, pattern, 1.0, 2, 3, 42L, decade);      // read this first: sensitive vs not, operator by operator

        Tools.loadDefaultGraph();
        compare(query, pattern, new double[]{0.01, 0.1, 0.5, 1.0}, new int[]{2, 3}, 100, 42L, decade, 0, 10);

        Tools.loadDefaultGraph();
        sweepSensitiveEdges(query, pattern, new double[]{0.01, 0.10, 0.25, 0.50, 0.75, 1.0}, 1.0, 2, 100, 42L, decade, 0, 10);
    }
}