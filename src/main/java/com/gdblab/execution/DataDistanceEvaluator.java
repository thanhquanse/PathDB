package com.gdblab.execution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Predicate;

import com.gdblab.algebra.queryplan.logical.BinaryLogicalOperator;
import com.gdblab.algebra.queryplan.logical.LogicalOperator;
import com.gdblab.algebra.queryplan.logical.UnaryLogicalOperator;
import com.gdblab.graph.schema.Edge;
import com.gdblab.graph.schema.GraphObject;
import com.gdblab.graph.schema.Node;
import com.gdblab.graph.schema.Path;
import com.gdblab.privacy.AttributeReleaseTracker;
import com.gdblab.privacy.DPMetrics;
import com.gdblab.privacy.RecordReleaseTracker;
import com.gdblab.privacy.SensitivePatternPolicy;

/**
 * DATA RELEASE STUDY: what the data consumer (for example an ML training job) receives from the FINAL RESULT of a query
 * that selects numeric fields, under four release approaches, measured against the GROUND TRUTH (normal execution).
 *
 * THE RESULT TABLE. One row per result path. The query's selected fields are described by Field(label, Attr, pos, sensitive):
 * the numeric property `attr` of the node at path position `pos` (0 = first node, -1 = last node, -2 = second last, ...).
 * A path is a SENSITIVE ROW when pattern.isSensitivePath(path). A node OWNS a sensitive value when `sensitiveOwner`
 * selects it (null = every endpoint of a sensitive edge); only fields flagged `sensitive` are ever perturbed.
 *
 * ARMS
 *   all-NS exact              : sensitive rows removed, every other row released exactly. NOT private (the absence of a row
 *                               reveals that it was sensitive); the non-private reference for any one-sided method.
 *   selective-op RR           : sensitive rows removed; every non-sensitive row released only with probability
 *                               p = 1 - e^-eps (OsdpRR), at EVERY operator, through a RecordReleaseTracker so a row gets ONE
 *                               coin however many operators emit it. The noise is the random dropping of non-sensitive rows.
 *   selective-op VALUE noise  : every row is kept; the sensitive fields of the OWNER nodes get Laplace(0, Delta/eps) noise,
 *                               ONCE per (node, property) through an AttributeReleaseTracker, in every operator and in the
 *                               final result. Everything else is exact. Protects values only; the structure stays public.
 *   DP-post uniform VALUE noise: execute exactly, then perturb the sensitive fields of EVERY node in the final result, once.
 * A standard DP release cannot hand out true rows, so uniform post-processing perturbs values.
 *
 * METRICS (final result against ground truth, per trial; reported as trimmed mean over trials, boxplots for RelError)
 *   D_table  = (1/N) sum_rows d_i,  d_i = 1 if the row is missing, else mean over fields of |v - vhat| / Delta_f.
 *              It splits exactly into D_missing = (N - R)/N (rows lost) + D_noise (value error on the rows that are present).
 *   RelError = |y - yhat| / max(1, y)   (DP4SQL), for COUNT, SUM and AVG. For selective-op RR, COUNT and SUM are rescaled
 *              by 1/p (inverse-probability estimator); AVG is the mean of the released rows. osdp-count: the one-sided
 *              Laplace count over the non-sensitive rows only (OsdpLaplaceL1).
 *   ML       = train-on-released, test-on-ground-truth: ordinary least squares of the last field on the first field fitted
 *              on the released table; slope error |b - b0| and RMSE ratio = RMSE(model on released) / RMSE(model on truth),
 *              both evaluated on the ground-truth rows. 1.00 means the released table trains as well as the real one.
 *
 * LIMITS. (1) selective-op RR protects rows (structure and values together); selective-op VALUE noise protects values only.
 * (2) Predicates that filter or join ON a sensitive value must run on the released value; this study perturbs after exact
 * execution. (3) RR's guarantee assumes independent records (OSDP); overlapping paths are an open case. (4) With a wide
 * public range the Laplace noise is large; look at the decomposition D_missing / D_noise to see where the error comes from.
 */
public final class DataDistanceEvaluator {

    private DataDistanceEvaluator() {}

    public record Attr(String name, double min, double max, boolean integer) {
        double range() { return max - min; }
    }

    /** A selected numeric field: attr of the node at position pos (0 first, -1 last, -2 second last, ...). */
    public record Field(String label, Attr attr, int pos, boolean sensitive) {}

    public static final Attr BIRTH_YEAR = new Attr("birthYear", 1980, 2000, true);
    public static final Attr MESSAGE_LENGTH = new Attr("length", 0, 1000, true);

    public static final class Config {
        public final List<Field> fields = new ArrayList<>();
        public Predicate<Node> sensitiveOwner = null;
        public Config field(final Field f) { fields.add(f); return this; }
        public Config owner(final Predicate<Node> p) { sensitiveOwner = p; return this; }
    }

    private static final String[] ARMS = {
            "all-NS exact [not private]",
            "selective-op RR (rows)",
            "selective-op VALUE noise",
            "DP-post uniform VALUE noise"};

    // =====================================================================================
    //  Snapshot of the plan (the query is executed ONCE)
    // =====================================================================================
    private static final class Row {
        final String id;
        final boolean sensitive;
        final String[] nodeIds, nodeNames, edgeIds, edgeLabels;
        final boolean[] edgeSens;
        Row(final String id, final boolean sensitive, final String[] nodeIds, final String[] nodeNames,
            final String[] edgeIds, final String[] edgeLabels, final boolean[] edgeSens) {
            this.id = id; this.sensitive = sensitive; this.nodeIds = nodeIds; this.nodeNames = nodeNames;
            this.edgeIds = edgeIds; this.edgeLabels = edgeLabels; this.edgeSens = edgeSens;
        }
    }

    private static final class Op {
        String label;
        int depth;
        final List<Integer> children = new ArrayList<>();
        final Map<String, Row> rows = new LinkedHashMap<>();
    }

    private static final class Snap {
        final List<Op> ops = new ArrayList<>();
        final Map<String, Map<String, String>> props = new HashMap<>();
        final Set<String> sensEdgeEndpoints = new HashSet<>();
        final Set<String> ownerNodes = new HashSet<>();
        Set<String> sensitiveNodes;
    }

    private static Snap snapshot(final String query, final SensitivePatternPolicy pattern,
                                 final Predicate<Node> owner) throws Exception {
        try {
            LogicalOperator root = IntermediateResultsExplainer.parseToLogicalRoot(query);
            Snap s = new Snap();
            collect(root, 0, pattern, owner, s);
            s.sensitiveNodes = (owner == null) ? s.sensEdgeEndpoints : s.ownerNodes;
            return s;
        } finally {
            Tools.resetContext();
        }
    }

    private static int collect(final LogicalOperator node, final int depth, final SensitivePatternPolicy pattern,
                               final Predicate<Node> owner, final Snap s) {
        List<Integer> kids = new ArrayList<>();
        if (node instanceof UnaryLogicalOperator u) {
            kids.add(collect(u.getChild(), depth + 1, pattern, owner, s));
        } else if (node instanceof BinaryLogicalOperator b) {
            kids.add(collect(b.getLeftChild(), depth + 1, pattern, owner, s));
            kids.add(collect(b.getRightChild(), depth + 1, pattern, owner, s));
        }
        Op op = new Op();
        op.label = node.getClass().getSimpleName();
        op.depth = depth;
        op.children.addAll(kids);
        for (Path p : IntermediateResultsExplainer.materialize(node)) {
            StringBuilder id = new StringBuilder();
            List<String> nids = new ArrayList<>(), names = new ArrayList<>(), eids = new ArrayList<>(), elabels = new ArrayList<>();
            List<Boolean> esens = new ArrayList<>();
            for (GraphObject go : p.getSequence()) {
                if (go instanceof Edge e) {
                    id.append("E:").append(e.getId()).append('|');
                    eids.add(e.getId());
                    elabels.add(String.valueOf(e.getLabel()));
                    esens.add(pattern.isSensitiveEdge(e));
                } else if (go instanceof Node nd) {
                    id.append("N:").append(nd.getId()).append('|');
                    nids.add(nd.getId());
                    String nm = (nd.getProperties() == null) ? null : nd.getProperties().get("name");
                    names.add(nd.getId() + (nm == null ? "" : "(" + nm + ")"));
                    s.props.putIfAbsent(nd.getId(), nd.getProperties() == null ? new HashMap<>() : new HashMap<>(nd.getProperties()));
                    if (owner != null && owner.test(nd)) s.ownerNodes.add(nd.getId());
                }
            }
            boolean[] es = new boolean[esens.size()];
            for (int i = 0; i < es.length; i++) {
                es[i] = esens.get(i);
                if (es[i] && i + 1 < nids.size()) {                 // edge i joins node i and node i+1
                    s.sensEdgeEndpoints.add(nids.get(i));
                    s.sensEdgeEndpoints.add(nids.get(i + 1));
                }
            }
            String rid = id.toString();
            op.rows.putIfAbsent(rid, new Row(rid, pattern.isSensitivePath(p), nids.toArray(new String[0]),
                    names.toArray(new String[0]), eids.toArray(new String[0]), elabels.toArray(new String[0]), es));
        }
        s.ops.add(op);   // post-order: children first, root LAST
        return s.ops.size() - 1;
    }

    // =====================================================================================
    //  The final result as a table of selected fields
    // =====================================================================================
    private static final class Tab {
        final List<Row> rows = new ArrayList<>();
        final List<double[]> truthRows = new ArrayList<>();
        final List<String[]> nodeRows = new ArrayList<>();
        double[][] truth;
        String[][] node;
        int n, k;
    }

    private static double propValue(final Snap s, final String nodeId, final String prop) {
        Map<String, String> pr = s.props.get(nodeId);
        String raw = (pr == null) ? null : pr.get(prop);
        if (raw == null) return Double.NaN;
        try { return Double.parseDouble(raw.trim()); } catch (NumberFormatException ex) { return Double.NaN; }
    }

    private static Tab table(final Snap s, final Config cfg) {
        Op root = s.ops.get(s.ops.size() - 1);
        Tab t = new Tab();
        t.k = cfg.fields.size();
        for (Row r : root.rows.values()) {
            double[] v = new double[t.k];
            String[] nd = new String[t.k];
            boolean ok = true;
            for (int f = 0; f < t.k && ok; f++) {
                Field fd = cfg.fields.get(f);
                int idx = fd.pos() >= 0 ? fd.pos() : r.nodeIds.length + fd.pos();
                if (idx < 0 || idx >= r.nodeIds.length) { ok = false; break; }
                double x = propValue(s, r.nodeIds[idx], fd.attr().name());
                if (Double.isNaN(x)) { ok = false; break; }
                v[f] = x;
                nd[f] = r.nodeIds[idx];
            }
            if (ok) { t.rows.add(r); t.truthRows.add(v); t.nodeRows.add(nd); }
        }
        t.n = t.rows.size();
        t.truth = t.truthRows.toArray(new double[0][]);
        t.node = t.nodeRows.toArray(new String[0][]);
        return t;
    }

    // =====================================================================================
    //  The four arms for one trial
    // =====================================================================================
    private static final class Arm {
        final boolean[] present;
        final double[][] val;
        final double scale;     // multiplies COUNT and SUM (1/p for the RR estimator)
        Arm(final boolean[] present, final double[][] val, final double scale) { this.present = present; this.val = val; this.scale = scale; }
    }

    private static Arm[] arms(final Tab tab, final Snap s, final Config cfg, final double eps, final long seed, final int t) {
        final int N = tab.n, K = tab.k;
        final double p = RecordReleaseTracker.releaseProbability(eps);
        boolean[] pNs = new boolean[N], pRr = new boolean[N], pAll = new boolean[N];
        Arrays.fill(pAll, true);
        RecordReleaseTracker rt = new RecordReleaseTracker(DPMetrics.mix(seed, t, Double.doubleToLongBits(eps)));
        for (int i = 0; i < N; i++) {
            Row r = tab.rows.get(i);
            pNs[i] = !r.sensitive;
            pRr[i] = rt.release(r.id, r.sensitive, eps).released();
        }
        AttributeReleaseTracker t1 = new AttributeReleaseTracker(DPMetrics.mix(seed, t, 21));
        AttributeReleaseTracker t2 = new AttributeReleaseTracker(DPMetrics.mix(seed, t, 22));
        double[][] vSel = new double[N][K], vUni = new double[N][K];
        for (int i = 0; i < N; i++) {
            for (int f = 0; f < K; f++) {
                Field fd = cfg.fields.get(f);
                Attr a = fd.attr();
                double tr = tab.truth[i][f];
                String nid = tab.node[i][f];
                vSel[i][f] = (fd.sensitive() && s.sensitiveNodes.contains(nid))
                        ? t1.release(nid, a.name(), tr, a.min(), a.max(), a.integer(), eps).value() : tr;
                vUni[i][f] = fd.sensitive() ? t2.release(nid, a.name(), tr, a.min(), a.max(), a.integer(), eps).value() : tr;
            }
        }
        return new Arm[]{new Arm(pNs, tab.truth, 1.0), new Arm(pRr, tab.truth, 1.0 / p),
                new Arm(pAll, vSel, 1.0), new Arm(pAll, vUni, 1.0)};
    }

    /** OsdpLaplaceL1 count over the NON-sensitive rows, RelError against the true row count. */
    private static double osdpCountError(final Tab tab, final double eps, final long seed, final int t) {
        int ns = 0;
        for (Row r : tab.rows) if (!r.sensitive) ns++;
        SplittableRandom rng = new SplittableRandom(DPMetrics.mix(seed, t, 31));
        double y = ns - Math.abs(DPMetrics.laplace(rng, 1.0 / eps));    // one-sided (non-positive) noise
        y = y < 0 ? 0 : y + Math.log(2) / eps;                          // clamp, add back the noise median
        return DPMetrics.relError(tab.n, Math.round(y));
    }

    // =====================================================================================
    //  Metrics
    // =====================================================================================
    private static final int M_ROWS = 0, M_DTAB = 1, M_MISS = 2, M_NOISE = 3, M_COUNT = 4;
    private static int mSum(final int f) { return 5 + 3 * f; }
    private static int mAvg(final int f) { return 6 + 3 * f; }
    private static int mAvgN(final int f) { return 7 + 3 * f; }
    private static int mSlope(final int k) { return 5 + 3 * k; }
    private static int mRmse(final int k) { return 6 + 3 * k; }
    private static int mLen(final int k) { return 7 + 3 * k; }

    private static String[] metricNames(final Config cfg) {
        int K = cfg.fields.size();
        String[] n = new String[mLen(K)];
        n[M_ROWS] = "rows released (share of N)";
        n[M_DTAB] = "D_table (whole-table distance)";
        n[M_MISS] = "   of which D_missing (rows lost)";
        n[M_NOISE] = "   of which D_noise (value error)";
        n[M_COUNT] = "COUNT  RelError";
        for (int f = 0; f < K; f++) {
            String l = cfg.fields.get(f).label();
            n[mSum(f)] = "SUM(" + l + ")  RelError";
            n[mAvg(f)] = "AVG(" + l + ")  RelError";
            n[mAvgN(f)] = "AVG(" + l + ")  error / range";
        }
        n[mSlope(K)] = "ML slope error |b - b0|";
        n[mRmse(K)] = "ML RMSE ratio (1.00 = as good as truth)";
        return n;
    }

    /** {slope a, b} of y ~ a + b x, or null if it cannot be fitted. */
    private static double[] ols(final double[] x, final double[] y, final int n) {
        if (n < 3) return null;
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
        mx /= n; my /= n;
        double sxx = 0, sxy = 0;
        for (int i = 0; i < n; i++) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); }
        if (sxx < 1e-12) return null;
        double b = sxy / sxx;
        return new double[]{my - b * mx, b};
    }

    private static double rmseOnTruth(final double[] m, final Tab tab, final int fx, final int fy) {
        double s = 0;
        for (int i = 0; i < tab.n; i++) {
            double e = tab.truth[i][fy] - (m[0] + m[1] * tab.truth[i][fx]);
            s += e * e;
        }
        return Math.sqrt(s / tab.n);
    }

    private static double[] metrics(final Tab tab, final Config cfg, final Arm a) {
        final int N = tab.n, K = tab.k;
        double[] m = new double[mLen(K)];
        Arrays.fill(m, Double.NaN);
        if (N == 0) return m;
        int R = 0;
        double dsum = 0;
        for (int i = 0; i < N; i++) {
            if (!a.present[i]) { dsum += 1.0; continue; }
            R++;
            double d = 0;
            for (int f = 0; f < K; f++) d += Math.min(1.0, Math.abs(tab.truth[i][f] - a.val[i][f]) / cfg.fields.get(f).attr().range());
            dsum += d / K;
        }
        m[M_ROWS] = R / (double) N;
        m[M_DTAB] = dsum / N;
        m[M_MISS] = (N - R) / (double) N;
        m[M_NOISE] = m[M_DTAB] - m[M_MISS];
        m[M_COUNT] = DPMetrics.relError(N, R * a.scale);
        for (int f = 0; f < K; f++) {
            double ySum = 0, hSum = 0;
            for (int i = 0; i < N; i++) { ySum += tab.truth[i][f]; if (a.present[i]) hSum += a.val[i][f]; }
            double yAvg = ySum / N, hAvg = R > 0 ? hSum / R : 0.0;
            m[mSum(f)] = DPMetrics.relError(ySum, hSum * a.scale);
            m[mAvg(f)] = DPMetrics.relError(yAvg, hAvg);
            m[mAvgN(f)] = Math.abs(yAvg - hAvg) / cfg.fields.get(f).attr().range();
        }
        if (K >= 2) {
            int fx = 0, fy = K - 1;
            double[] xt = new double[N], yt = new double[N], xr = new double[N], yr = new double[N];
            int c = 0;
            for (int i = 0; i < N; i++) {
                xt[i] = tab.truth[i][fx]; yt[i] = tab.truth[i][fy];
                if (a.present[i]) { xr[c] = a.val[i][fx]; yr[c] = a.val[i][fy]; c++; }
            }
            double[] m0 = ols(xt, yt, N), mr = ols(xr, yr, c);
            if (m0 != null && mr != null) {
                m[mSlope(K)] = Math.abs(mr[1] - m0[1]);
                double base = rmseOnTruth(m0, tab, fx, fy);
                m[mRmse(K)] = base < 1e-12 ? Double.NaN : rmseOnTruth(mr, tab, fx, fy) / base;
            }
        }
        return m;
    }

    private static double tm(final double[] a) {
        double[] c = Arrays.stream(a).filter(x -> !Double.isNaN(x)).toArray();
        return c.length == 0 ? Double.NaN : DPMetrics.trimmedMean(c, 0.2);
    }

    // =====================================================================================
    //  Plan, per-operator statistics
    // =====================================================================================
    private static String describe(final String label) {
        switch (label) {
            case "LogicalOpSelectionByLabel": return "selection by label: returns the edges that carry the requested label (a base scan)";
            case "LogicalOpSelection": return "selection: keeps the paths of its child that satisfy a condition (a subset of the child's rows)";
            case "LogicalOpNodeJoin": return "node join: concatenates a path of the left child with a path of the right child that meet at a node";
            default: return "(no description registered for this operator class)";
        }
    }

    private static String clip(final String s, final int w) { return s.length() <= w ? s : s.substring(0, w - 3) + "..."; }

    private static String pathText(final Row r) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < r.nodeNames.length; i++) {
            sb.append(r.nodeNames[i]);
            if (i < r.edgeLabels.length) sb.append("-[").append(r.edgeLabels[i]).append("]->");
        }
        return sb.toString();
    }

    private static String edgeFlags(final Row r) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < r.edgeSens.length; i++) sb.append(i == 0 ? "" : ",").append(r.edgeSens[i] ? "S" : "-");
        return sb.toString();
    }

    private static String whySensitive(final Row r) {
        List<String> hits = new ArrayList<>();
        for (int i = 0; i < r.edgeSens.length; i++) {
            if (r.edgeSens[i]) {
                String ends = (i + 1 < r.nodeNames.length) ? r.nodeNames[i] + " - " + r.nodeNames[i + 1] : "";
                hits.add("edge " + r.edgeIds[i] + " [" + r.edgeLabels[i] + "] " + ends);
            }
        }
        return hits.isEmpty() ? "path-level rule (isSensitivePath = true, no single edge flagged)" : String.join("; ", hits);
    }

    private static Set<Attr> sensitiveAttrs(final Config cfg) {
        Set<Attr> a = new java.util.LinkedHashSet<>();
        for (Field f : cfg.fields) if (f.sensitive()) a.add(f.attr());
        return a;
    }

    private static final class OpStat {
        final int[] rows, sens, released, coin, newDec, reusedDec, ownerVals, newVals, reusedVals;
        RecordReleaseTracker.LedgerStats rowLedger;
        AttributeReleaseTracker.LedgerStats valLedger;
        OpStat(final int n) {
            rows = new int[n]; sens = new int[n]; released = new int[n]; coin = new int[n]; newDec = new int[n];
            reusedDec = new int[n]; ownerVals = new int[n]; newVals = new int[n]; reusedVals = new int[n];
        }
    }

    private static OpStat opStats(final Snap s, final Config cfg, final double eps, final long seed) {
        OpStat st = new OpStat(s.ops.size());
        RecordReleaseTracker rt = new RecordReleaseTracker(DPMetrics.mix(seed, 0, Double.doubleToLongBits(eps)));
        AttributeReleaseTracker vt = new AttributeReleaseTracker(DPMetrics.mix(seed, 0, 21));
        Set<Attr> sa = sensitiveAttrs(cfg);
        for (int oi = 0; oi < s.ops.size(); oi++) {
            Set<String> seen = new HashSet<>();
            for (Row r : s.ops.get(oi).rows.values()) {
                st.rows[oi]++;
                RecordReleaseTracker.Outcome o = rt.release(r.id, r.sensitive, eps);
                if (r.sensitive) st.sens[oi]++; else if (o.released()) st.released[oi]++; else st.coin[oi]++;
                if (o.isNew()) st.newDec[oi]++; else st.reusedDec[oi]++;
                for (String nid : r.nodeIds) {
                    if (!s.sensitiveNodes.contains(nid)) continue;
                    for (Attr a : sa) {
                        double v = propValue(s, nid, a.name());
                        if (Double.isNaN(v) || !seen.add(nid + "|" + a.name())) continue;
                        AttributeReleaseTracker.Outcome vo = vt.release(nid, a.name(), v, a.min(), a.max(), a.integer(), eps);
                        st.ownerVals[oi]++;
                        if (vo.isNew()) st.newVals[oi]++; else st.reusedVals[oi]++;
                    }
                }
            }
        }
        st.rowLedger = rt.ledger(true);
        st.valLedger = vt.ledger();
        return st;
    }

    private static void planOverview(final Snap s) {
        System.out.println("PLAN (execution order, children first). EVERY operator output is a release point where selective DP intervenes:");
        System.out.printf("  %-4s %-30s %-6s %-9s %8s %10s%n", "idx", "operator", "level", "children", "rows", "sensitive");
        for (int i = 0; i < s.ops.size(); i++) {
            Op op = s.ops.get(i);
            long sens = op.rows.values().stream().filter(r -> r.sensitive).count();
            System.out.printf("  %-4d %-30s %-6d %-9s %8d %10d%s%n", i, op.label, op.depth, op.children.isEmpty() ? "-" : op.children.toString(),
                    op.rows.size(), sens, i == s.ops.size() - 1 ? "   <- ROOT (final result)" : "");
        }
        if (s.ops.size() == 1) System.out.println("  NOTE: a single operator: there are no intermediate operators, so nothing can be reused by a parent.");
        System.out.println();
    }

    private static void noiseMap(final double eps) {
        double p = RecordReleaseTracker.releaseProbability(eps);
        System.out.printf("WHERE IS THE NOISE?  (p = 1 - e^-eps = %.4f)%n", p);
        System.out.printf("  %-28s | %-26s | %-34s | %s%n", "arm", "SENSITIVE part", "NON-SENSITIVE part", "noise");
        System.out.printf("  %-28s | %-26s | %-34s | %s%n", "all-NS exact", "rows removed", "rows released, exactly", "none: NOT private (a missing row reveals it was sensitive)");
        System.out.printf("  %-28s | %-26s | %-34s | %s%n", "selective-op RR", "rows removed, never released", "each row released with prob. p",
                "WHICH non-sensitive rows are dropped; nothing is added to the removed rows");
        System.out.printf("  %-28s | %-26s | %-34s | %s%n", "selective-op VALUE noise", "row kept, value noised", "row and values exact",
                "Laplace(0, Delta/eps) on the owner's sensitive value, once per (node, property)");
        System.out.printf("  %-28s | %-26s | %-34s | %s%n", "DP-post uniform VALUE noise", "row kept, value noised", "row kept, value noised",
                "the same Laplace noise on EVERY node's value, once, in the final result only");
        System.out.println("  osdp-count (COUNT only): the sensitive rows are not counted; one-sided Laplace(1/eps) noise is added to the count of the non-sensitive ones.");
        System.out.println("  Random dropping is needed because releasing every non-sensitive row exactly would make each missing row visibly sensitive.\n");
    }

    // =====================================================================================
    //  compare
    // =====================================================================================
    public static void compare(final String query, final SensitivePatternPolicy pattern, final double[] epsilons,
                               final Config cfg, final int trials, final long seed) throws Exception {
        System.out.println("=== DATA RELEASE STUDY: ground truth vs all-NS vs selective-op (rows / values) vs DP post-processing ===");
        System.out.println("query: " + query);
        System.out.println("selected fields: " + cfg.fields.stream().map(f -> f.label() + (f.sensitive() ? " [sensitive]" : "")).toList() + "\n");
        Snap s = snapshot(query, pattern, cfg.sensitiveOwner);
        Tab tab = table(s, cfg);
        planOverview(s);
        final int N = tab.n, K = tab.k;
        long sensRows = tab.rows.stream().filter(r -> r.sensitive).count();
        long ownerCells = 0;
        for (int i = 0; i < N; i++) for (int f = 0; f < K; f++) if (cfg.fields.get(f).sensitive() && s.sensitiveNodes.contains(tab.node[i][f])) ownerCells++;
        System.out.printf("ground truth final table: %d rows x %d fields = %d cells; %d sensitive rows; %d cells belong to sensitive owners (%.1f%% of the cells)%n%n",
                N, K, N * K, sensRows, ownerCells, N * K == 0 ? 0.0 : 100.0 * ownerCells / (N * K));
        if (N == 0) { System.out.println("The final result has no row with all selected fields: nothing to compare."); return; }

        final String[] names = metricNames(cfg);
        final int len = mLen(K);
        double focus = epsilons[0];
        for (double e : epsilons) if (Math.abs(e - 1.0) < Math.abs(focus - 1.0)) focus = e;

        for (double eps : epsilons) {
            double[][][] M = new double[4][len][trials];
            double[] osdp = new double[trials];
            for (int t = 0; t < trials; t++) {
                Arm[] a = arms(tab, s, cfg, eps, seed, t);
                for (int k = 0; k < 4; k++) {
                    double[] m = metrics(tab, cfg, a[k]);
                    for (int j = 0; j < len; j++) M[k][j][t] = m[j];
                }
                osdp[t] = osdpCountError(tab, eps, seed, t);
            }
            System.out.printf("%n################  epsilon = %.3f  (%d trials, p = %.4f)  ################%n%n", eps, trials, RecordReleaseTracker.releaseProbability(eps));
            noiseMap(eps);

            System.out.println("1. DISTANCE AND ERROR OF THE FINAL RESULT vs THE GROUND TRUTH (trimmed mean over trials)");
            System.out.printf("   %-42s | %12s %12s %12s %12s%n", "metric", "all-NS", "selective RR", "sel. VALUE", "DP-post");
            double[][] mean = new double[4][len];
            for (int j = 0; j < len; j++) {
                for (int k = 0; k < 4; k++) mean[k][j] = tm(M[k][j]);
                System.out.printf("   %-42s | %12s %12s %12s %12s%n", names[j], f4(mean[0][j]), f4(mean[1][j]), f4(mean[2][j]), f4(mean[3][j]));
            }
            System.out.printf("   %-42s | %12s %12s %12s %12.4f   (one-sided noise on the count of non-sensitive rows)%n", "COUNT  RelError, osdp-count", "-", "-", "-", tm(osdp));

            System.out.println("\n2. DP4SQL RelError = |y - yhat| / max(1, y): boxplot numbers over the trials; reference line = always output 0 (RelError 1.0)");
            System.out.printf("   %-34s %-28s | %8s %8s %8s %8s %8s | %s%n", "statistic", "arm", "min", "Q1", "median", "Q3", "max", "vs always-0");
            List<Integer> relIdx = new ArrayList<>();
            relIdx.add(M_COUNT);
            for (int f = 0; f < K; f++) { relIdx.add(mSum(f)); relIdx.add(mAvg(f)); }
            for (int j : relIdx) {
                for (int k = 0; k < 4; k++) printBox(names[j], ARMS[k], M[k][j]);
                if (j == M_COUNT) printBox(names[j] + " (osdp-count)", "all-NS rows, one-sided noise", osdp);
            }

            System.out.println();
            reading(mean, K, sensRows, N, ownerCells, eps);

            OpStat st = opStats(s, cfg, eps, seed);
            System.out.println("\n3. PRIVACY ACCOUNTING (one decision per row, one release per (node, property))");
            System.out.printf("   rows   : %d decisions, %d requests, %d reused; composed epsilon per sensitive row: tracked max %.2f, untracked max %.2f%n",
                    st.rowLedger.records(), Arrays.stream(st.rows).sum(), Arrays.stream(st.reusedDec).sum(), st.rowLedger.maxTracked(), st.rowLedger.maxUntracked());
            System.out.printf("   values : %d released, %d requests, %d reused; composed epsilon per node: tracked max %.2f, untracked max %.2f%n",
                    st.valLedger.cells(), st.valLedger.requests(), st.valLedger.reused(), st.valLedger.maxNodeTracked(), st.valLedger.maxNodeUntracked());
            System.out.printf("   exposure: selective-op RR removes %d of %d final rows (%.1f%%) and drops non-sensitive ones at random; selective-op VALUE noise perturbs %.1f%% of the cells and keeps every row (structure public);%n",
                    sensRows, N, 100.0 * sensRows / N, 100.0 * ownerCells / (N * K));
            System.out.println("            DP-post uniform VALUE noise perturbs 100% of the sensitive-flagged cells. Fewer perturbed or removed items means less protection, so always read the error together with this line.");

            if (eps == focus) {
                System.out.printf("%n4. OPERATOR BY OPERATOR at eps = %.3f (each operator's exact output is a release point)%n", eps);
                System.out.printf("   %-3s %-28s %7s %9s %9s %9s | %6s %7s | %10s %6s %7s%n", "lvl", "operator", "rows", "sensitive", "released", "coin-drop", "NEW", "REUSED", "owner vals", "NEW", "REUSED");
                for (int oi = 0; oi < s.ops.size(); oi++) {
                    System.out.printf("   %-3d %-28s %7d %9d %9d %9d | %6d %7d | %10d %6d %7d%n", s.ops.get(oi).depth, s.ops.get(oi).label, st.rows[oi], st.sens[oi],
                            st.released[oi], st.coin[oi], st.newDec[oi], st.reusedDec[oi], st.ownerVals[oi], st.newVals[oi], st.reusedVals[oi]);
                }
                System.out.println("   sensitive = removed rows (no noise); released / coin-drop = what the coin did to the non-sensitive rows; owner vals = sensitive values replaced");
                System.out.println("   by their released value at that operator. NEW = first retrieval (decision/noise created); REUSED = same decision or value reused.");
            }
        }
        System.out.println("\nRun explain(...) for the record-by-record trace of one draw (every operator, the final result, and the arithmetic of each number).\n");
    }

    private static String f4(final double v) { return Double.isNaN(v) ? "n/a" : String.format("%.4f", v); }

    private static void printBox(final String stat, final String arm, final double[] a) {
        double[] c = Arrays.stream(a).filter(x -> !Double.isNaN(x)).toArray();
        if (c.length == 0) { System.out.printf("   %-34s %-28s | %8s%n", clip(stat, 34), arm, "n/a"); return; }
        DPMetrics.BoxStats b = DPMetrics.boxStats(c);
        System.out.printf("   %-34s %-28s | %8.4f %8.4f %8.4f %8.4f %8.4f | %s%n", clip(stat, 34), arm, b.min(), b.q1(), b.median(), b.q3(), b.max(),
                DPMetrics.verdictVsZero(b.median()));
    }

    /** Plain-language reading of the table above, generated from the measured numbers. */
    private static void reading(final double[][] m, final int K, final long sensRows, final int N, final long ownerCells, final double eps) {
        double p = RecordReleaseTracker.releaseProbability(eps);
        System.out.println("READING THE RESULT (what is good, and why)");
        System.out.printf("   D_table = 0 means identical to the ground truth, 1 means nothing usable. D_table = D_missing (rows lost) + D_noise (value error).%n");
        System.out.printf("   all-NS exact     : D_table %.4f = missing %.4f (the %d sensitive rows) + noise %.4f. Only the sensitive rows are missing and nothing is noised: the best any row-removing method can do, but NOT private.%n",
                m[0][M_DTAB], m[0][M_MISS], sensRows, m[0][M_NOISE]);
        System.out.printf("   selective-op RR  : D_table %.4f = missing %.4f (sensitive rows %.4f + rows dropped by the coin %.4f) + noise %.4f. Released rows are exact;%n",
                m[1][M_DTAB], m[1][M_MISS], sensRows / (double) N, Math.max(0.0, m[1][M_MISS] - sensRows / (double) N), m[1][M_NOISE]);
        System.out.printf("                      the loss is rows only: policy %.4f + expected coin loss (1-p)*NS/N = %.4f; expected released share p*NS/N = %.4f.%n",
                sensRows / (double) N, (1 - p) * (N - sensRows) / N, p * (N - sensRows) / N);
        System.out.printf("   selective VALUE  : D_table %.4f = missing 0 + noise %.4f. No row is lost; only %.1f%% of the cells (the sensitive owners') carry noise.%n",
                m[2][M_DTAB], m[2][M_NOISE], 100.0 * ownerCells / Math.max(1, N * K));
        System.out.printf("   DP-post VALUE    : D_table %.4f = missing 0 + noise %.4f. Every sensitive-flagged value is noised, so the noise reaches (almost) all cells.%n",
                m[3][M_DTAB], m[3][M_NOISE]);
        int best = 1;
        for (int k = 2; k <= 3; k++) if (m[k][M_DTAB] < m[best][M_DTAB]) best = k;
        System.out.printf("   => lowest D_table among the PRIVATE arms: %s (%.4f).%n", ARMS[best], m[best][M_DTAB]);
        if (m[2][M_DTAB] < m[3][M_DTAB]) {
            System.out.printf("   => selective VALUE noise is %.1f times closer to the ground truth than DP-post, because it perturbs only the sensitive owners' values.%n",
                    m[3][M_DTAB] / Math.max(1e-12, m[2][M_DTAB]));
        } else {
            System.out.println("   => selective VALUE noise is NOT closer than DP-post here: the sensitive owners cover most of the cells, so both perturb about the same share.");
        }
        System.out.printf("   SUM / AVG: AVG error is small when the noise is zero-mean and averaged over many rows; the sensitive owner's single draw is shared by every row that%n");
        System.out.printf("   contains it, so its error does not average out. RR needs the 1/p rescaling for COUNT and SUM (here 1/p = %.2f) and still has sampling variance.%n", 1.0 / p);
        System.out.println("   Note: RelError of an AVG of a value near 1950 divides by the value itself, so it is tiny for every arm; compare arms with 'error / range'.");
        System.out.println("         COUNT and SUM RelError are the informative ones for the row-removing arms.");
        if (K >= 2) {
            System.out.printf("   ML (train on the released table, test on the truth), RMSE ratio: RR %s, VALUE %s, DP-post %s  (1.00 = as good as training on the real table).%n",
                    f4(m[1][mRmse(K)]), f4(m[2][mRmse(K)]), f4(m[3][mRmse(K)]));
            double rmax = 0;
            for (int k = 1; k <= 3; k++) if (!Double.isNaN(m[k][mRmse(K)])) rmax = Math.max(rmax, m[k][mRmse(K)]);
            if (rmax > 0 && rmax < 1.05) {
                System.out.println("   The RMSE ratios are all close to 1: the last field is only weakly predicted by the first one in this data, so this ML test cannot tell the arms apart;");
                System.out.println("   the slope error still shows the distortion. Use fields with a real relationship (for example salary and age) for a sharper ML comparison.");
            }
            System.out.println("   Uniform noise on the training features biases the fitted slope toward 0 (attenuation); removing rows keeps the values exact but shrinks and may bias the sample.");
        }
        System.out.println("   Caution: lower error here is paid for with less protection (see the exposure line). RR and VALUE noise protect different things (rows versus values).");
    }

    // =====================================================================================
    //  explain: one draw, completely shown
    // =====================================================================================
    public static void explain(final String query, final SensitivePatternPolicy pattern, final double eps,
                               final Config cfg, final int rowsShown, final long seed) throws Exception {
        Snap s = snapshot(query, pattern, cfg.sensitiveOwner);
        Tab tab = table(s, cfg);
        final int nOps = s.ops.size(), last = nOps - 1;
        final double p = RecordReleaseTracker.releaseProbability(eps);
        final long rseed = DPMetrics.mix(seed, 0, Double.doubleToLongBits(eps));
        RecordReleaseTracker rt = new RecordReleaseTracker(rseed);
        AttributeReleaseTracker vt = new AttributeReleaseTracker(DPMetrics.mix(seed, 0, 21));
        Set<Attr> sa = sensitiveAttrs(cfg);

        System.out.printf("=== EXPLAIN data release (eps = %.4f, p = 1 - e^-eps = %.5f) ===%n", eps, p);
        System.out.println("query: " + query);
        System.out.println("selected fields: " + cfg.fields.stream().map(f -> f.label() + " = " + f.attr().name() + "@pos" + f.pos() + (f.sensitive() ? " [sensitive]" : "")).toList() + "\n");
        System.out.println("1. WHAT IS SENSITIVE AND WHAT HAPPENS TO IT");
        System.out.println("   row        : one result path. SENSITIVE row = pattern.isSensitivePath(path). 'edges' marks isSensitiveEdge per edge (S / -).");
        System.out.println("   owner      : a node whose sensitive field values are protected: " + (cfg.sensitiveOwner == null ? "every endpoint of a sensitive edge" : "the nodes chosen by the given predicate (e.g. the President)"));
        System.out.println("   RR         : sensitive row -> removed (no coin); non-sensitive row -> released iff its coin u < p. The coin is fixed once per row (tracker).");
        System.out.println("   VALUE noise: an owner's sensitive value -> released = clamp(round(clamp(v) + Lap(0, Delta/eps))), computed once per (node, property) and reused.");
        System.out.println("   All of this happens at every operator output; the final result is the root operator's.\n");
        noiseMap(eps);
        System.out.println("2. FORMULAS");
        System.out.printf("   p = 1 - e^(-%.4f) = %.5f;  E[rows released at an operator] = p * (#non-sensitive rows)%n", eps, p);
        for (Attr a : sa) System.out.printf("   %s: Delta = %.0f - %.0f = %.0f;  noise scale = Delta/eps = %.3f;  E|noise| = scale (before clamping)%n", a.name(), a.max(), a.min(), a.range(), a.range() / eps);
        System.out.println("   D_table = (1/N) sum_i d_i,  d_i = 1 (row missing) or mean_f |v - vhat|/Delta_f;   RelError = |y - yhat| / max(1, y);");
        System.out.println("   COUNT/SUM of RR are multiplied by 1/p; AVG(RR) = mean of the released rows; ML = OLS of the last field on the first, trained on the released table, tested on the truth.\n");
        System.out.println("3. PLAN");
        planOverview(s);
        for (int i = 0; i < nOps; i++) System.out.printf("   [%d] %s : %s%n", i, s.ops.get(i).label, describe(s.ops.get(i).label));

        System.out.println("\n4. OPERATOR BY OPERATOR (the release point after each operator)");
        Map<String, Integer> firstSeen = new HashMap<>();
        for (int oi = 0; oi < nOps; oi++) {
            Op op = s.ops.get(oi);
            List<Object[]> sensL = new ArrayList<>(), nsL = new ArrayList<>();   // {Row, u, released, isNew, decidedAt}
            for (Row r : op.rows.values()) {
                RecordReleaseTracker.Outcome o = rt.release(r.id, r.sensitive, eps);
                firstSeen.putIfAbsent(r.id, oi);
                Object[] rec = {r, RecordReleaseTracker.unit(rseed, r.id), o.released(), o.isNew(), firstSeen.get(r.id)};
                (r.sensitive ? sensL : nsL).add(rec);
            }
            long released = nsL.stream().filter(x -> (Boolean) x[2]).count();
            long coin = nsL.size() - released;
            long fresh = sensL.stream().filter(x -> (Boolean) x[3]).count() + nsL.stream().filter(x -> (Boolean) x[3]).count();
            System.out.println("\n" + "=".repeat(112));
            System.out.printf("OPERATOR [%d] %s (level %d)%s%n", oi, op.label, op.depth, oi == last ? "   <- ROOT, its released set is the FINAL RESULT" : "");
            System.out.println("  meaning : " + describe(op.label));
            for (int c : op.children) {
                long shared = op.rows.keySet().stream().filter(id -> s.ops.get(c).rows.containsKey(id)).count();
                System.out.printf("  input   : child [%d] %s emitted %d rows; %d of this operator's %d rows are the SAME rows (decisions reused)%n", c, s.ops.get(c).label, s.ops.get(c).rows.size(), shared, op.rows.size());
            }
            if (op.children.isEmpty()) System.out.println("  input   : none (base operator)");
            System.out.printf("  EXACT OUTPUT : %d rows = %d SENSITIVE + %d NON-SENSITIVE%n", op.rows.size(), sensL.size(), nsL.size());

            System.out.printf("%n  (A) SENSITIVE rows (%d): removed, no coin%n", sensL.size());
            if (!sensL.isEmpty()) System.out.printf("      %-4s %-44s %-9s %-50s %s%n", "#", "row (path)", "edges", "why it is sensitive", "tracker");
            for (int i = 0; i < Math.min(rowsShown, sensL.size()); i++) {
                Row r = (Row) sensL.get(i)[0];
                System.out.printf("      %-4d %-44s %-9s %-50s %s%n", i + 1, clip(pathText(r), 44), edgeFlags(r), clip(whySensitive(r), 50),
                        (Boolean) sensL.get(i)[3] ? "NEW" : "REUSED (decided at [" + sensL.get(i)[4] + "])");
            }
            if (sensL.size() > rowsShown) System.out.printf("      ... %d more sensitive rows not shown%n", sensL.size() - rowsShown);

            System.out.printf("%n  (B) NON-SENSITIVE rows (%d): released iff u < p = %.5f   (sorted by u)%n", nsL.size(), p);
            nsL.sort((x, y) -> Double.compare((Double) x[1], (Double) y[1]));
            if (!nsL.isEmpty()) System.out.printf("      %-4s %-44s %-9s %8s %-6s %-18s %s%n", "#", "row (path)", "edges", "u", "u<p?", "decision", "tracker");
            for (int i = 0; i < Math.min(rowsShown, nsL.size()); i++) {
                Object[] x = nsL.get(i);
                Row r = (Row) x[0];
                System.out.printf("      %-4d %-44s %-9s %8.5f %-6s %-18s %s%n", i + 1, clip(pathText(r), 44), edgeFlags(r), (Double) x[1],
                        (Double) x[1] < p ? "yes" : "no", (Boolean) x[2] ? "RELEASED" : "dropped (coin)", (Boolean) x[3] ? "NEW" : "REUSED (decided at [" + x[4] + "])");
            }
            if (nsL.size() > rowsShown) System.out.printf("      ... %d more non-sensitive rows not shown%n", nsL.size() - rowsShown);
            System.out.printf("%n  COUNTS AT THIS OPERATOR: released = #{non-sensitive : u < p} = %d;  removed (sensitive) = %d;  dropped by the coin = %d%n", released, sensL.size(), coin);
            System.out.printf("      expected released E = p * NS = %.5f * %d = %.3f   observed = %d      decisions NEW = %d, REUSED = %d%n", p, nsL.size(), p * nsL.size(), released, fresh, op.rows.size() - fresh);

            // value noise at this operator
            LinkedHashMap<String, Object[]> vals = new LinkedHashMap<>();       // key -> {node id, Attr, true value, count}
            for (Row r : op.rows.values()) {
                for (String nid : r.nodeIds) {
                    if (!s.sensitiveNodes.contains(nid)) continue;
                    for (Attr a : sa) {
                        double v = propValue(s, nid, a.name());
                        if (Double.isNaN(v)) continue;
                        Object[] e = vals.computeIfAbsent(nid + "|" + a.name(), k -> new Object[]{nid, a, v, 0});
                        e[3] = (Integer) e[3] + 1;
                    }
                }
            }
            System.out.printf("%n  (C) SENSITIVE VALUES at this operator (%d): replaced by their released value (noise once per node and property)%n", vals.size());
            if (!vals.isEmpty()) System.out.printf("      %-26s %-10s %8s %10s %10s %6s  %s%n", "node", "property", "true", "noise", "released", "rows", "tracker");
            int shownV = 0;
            for (Object[] e : vals.values()) {
                Attr a = (Attr) e[1];
                AttributeReleaseTracker.Outcome o = vt.release((String) e[0], a.name(), (Double) e[2], a.min(), a.max(), a.integer(), eps);
                if (shownV++ < rowsShown) {
                    Map<String, String> pr = s.props.get((String) e[0]);
                    String nm = pr == null ? "" : pr.get("name");
                    System.out.printf("      %-26s %-10s %8.0f %10.2f %10.0f %6d  %s%n", e[0] + (nm == null ? "" : "(" + nm + ")"), a.name(), (Double) e[2], o.noise(), o.value(), (Integer) e[3], o.isNew() ? "NEW" : "REUSED");
                }
            }
            if (vals.size() > rowsShown) System.out.printf("      ... %d more sensitive values not shown%n", vals.size() - rowsShown);
        }

        // ---------------- final result ----------------
        System.out.println("\n" + "=".repeat(112));
        System.out.println("5. FINAL QUERY RESULT (root operator [" + last + "]): the ground-truth table and what each arm releases");
        final int N = tab.n, K = tab.k;
        Arm[] a = arms(tab, s, cfg, eps, seed, 0);
        System.out.printf("   %d rows with all selected fields (%d sensitive).  Values are [%s]; * marks a value that differs from the truth.%n", N,
                tab.rows.stream().filter(r -> r.sensitive).count(), String.join(", ", cfg.fields.stream().map(Field::label).toList()));
        System.out.printf("   %-4s %-9s %-20s | %-22s | %-22s | %-22s | %s%n", "#", "sensitive", "RR decision", "ground truth", "selective VALUE noise", "DP-post VALUE noise", "row (path)");
        for (int i = 0; i < Math.min(N, 200); i++) {
            Row r = tab.rows.get(i);
            String dec = r.sensitive ? "removed (sensitive)" : (a[1].present[i] ? "RELEASED" : "dropped (coin)");
            System.out.printf("   %-4d %-9s %-20s | %-22s | %-22s | %-22s | %s%n", i + 1, r.sensitive ? "YES" : "no", dec, vec(tab.truth[i], null), vec(a[2].val[i], tab.truth[i]), vec(a[3].val[i], tab.truth[i]), pathText(r));
        }
        if (N > 200) System.out.printf("   ... %d more rows not shown%n", N - 200);
        int rr = 0, ns = 0;
        for (int i = 0; i < N; i++) { if (a[1].present[i]) rr++; if (a[0].present[i]) ns++; }
        System.out.printf("   what each arm outputs: ground truth %d rows | all-NS %d rows (exact) | selective RR %d rows (E = p*NS = %.2f, exact values) | selective VALUE %d rows | DP-post %d rows%n",
                N, ns, rr, p * ns, N, N);

        System.out.println("\n6. DISTANCE AND ERROR, WORKED OUT FOR THIS DRAW");
        String[] names = metricNames(cfg);
        double[][] m = new double[4][];
        for (int k = 0; k < 4; k++) m[k] = metrics(tab, cfg, a[k]);
        System.out.printf("   %-42s | %12s %12s %12s %12s%n", "metric", "all-NS", "selective RR", "sel. VALUE", "DP-post");
        for (int j = 0; j < names.length; j++) System.out.printf("   %-42s | %12s %12s %12s %12s%n", names[j], f4(m[0][j]), f4(m[1][j]), f4(m[2][j]), f4(m[3][j]));
        System.out.println("\n   Arithmetic:");
        for (int k = 0; k < 4; k++) {
            int R = 0;
            double noiseSum = 0;
            for (int i = 0; i < N; i++) {
                if (!a[k].present[i]) continue;
                R++;
                double d = 0;
                for (int f = 0; f < K; f++) d += Math.min(1.0, Math.abs(tab.truth[i][f] - a[k].val[i][f]) / cfg.fields.get(f).attr().range());
                noiseSum += d / K;
            }
            System.out.printf("   %-30s D_table = (missing %d + value error %.3f) / %d = %.4f   [D_missing = %d/%d = %.4f, D_noise = %.3f/%d = %.4f]%n", ARMS[k], N - R, noiseSum, N,
                    ((N - R) + noiseSum) / N, N - R, N, (N - R) / (double) N, noiseSum, N, noiseSum / N);
        }
        for (int f = 0; f < K; f++) {
            double ySum = 0;
            for (int i = 0; i < N; i++) ySum += tab.truth[i][f];
            System.out.printf("   SUM(%s): truth %.0f, AVG truth %.3f%n", cfg.fields.get(f).label(), ySum, ySum / N);
            for (int k = 0; k < 4; k++) {
                int R = 0;
                double h = 0;
                for (int i = 0; i < N; i++) if (a[k].present[i]) { R++; h += a[k].val[i][f]; }
                System.out.printf("      %-30s SUM_hat = %.0f x %.2f = %.0f -> RelError %.4f | AVG_hat = %.3f -> RelError %.4f%n", ARMS[k], h, a[k].scale, h * a[k].scale,
                        DPMetrics.relError(ySum, h * a[k].scale), R > 0 ? h / R : 0.0, DPMetrics.relError(ySum / N, R > 0 ? h / R : 0.0));
            }
        }
        System.out.printf("   COUNT: truth %d;  RR: %d / p = %.2f -> RelError %.4f;  all-NS: %d -> %.4f;  value arms: %d -> 0 (rows are public)%n", N, rr, rr / p,
                DPMetrics.relError(N, rr / p), ns, DPMetrics.relError(N, ns), N);

        System.out.println();
        double[][] mean = m;   // one draw: the 'trimmed mean' is the value itself
        reading(mean, K, tab.rows.stream().filter(r -> r.sensitive).count(), N,
                java.util.stream.IntStream.range(0, N).mapToLong(i -> java.util.stream.IntStream.range(0, K)
                        .filter(f -> cfg.fields.get(f).sensitive() && s.sensitiveNodes.contains(tab.node[i][f])).count()).sum(), eps);
        System.out.println();
    }

    private static String vec(final double[] v, final double[] truth) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.0f", v[i]));
            if (truth != null && Math.abs(v[i] - truth[i]) > 1e-9) sb.append('*');
        }
        return sb.append(']').toString();
    }

    // =====================================================================================
    //  Example
    // =====================================================================================
    public static void main(String[] args) throws Exception {
        SensitivePatternPolicy pattern = new SensitivePatternPolicy();
        pattern.addSensitiveNodeName("President");
        String query = "MATCH p = (x)-[knows+.created]->(msg) RETURN x.birthYear, x.name, NODE(2).name, msg.length LIMIT 1000;";

        // Selected fields: the creator's birthYear (second last node; sensitive) and the message length (last node; not sensitive).
        // Add Field("salary", new Attr("salary", 0, 200000, true), -2, true) later for a salary column.
        Config cfg = new Config()
                .field(new Field("birthYear", BIRTH_YEAR, -2, true))
                .field(new Field("length", MESSAGE_LENGTH, -1, false))
                .owner(nd -> nd.getProperties() != null && "President".equals(nd.getProperties().get("name")));

        Tools.loadDefaultGraph();
        explain(query, pattern, 1.0, cfg, 5, 42L);                       // read this first

        Tools.loadDefaultGraph();
        compare(query, pattern, new double[]{0.5, 1.0, 2.0, 5.0}, cfg, 100, 42L);
    }
}