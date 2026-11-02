package com.gdblab.execution;

import java.util.ArrayList;
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
 * PRIVACY ASSESSMENT of the selective operator-level release (OsdpRR on rows, one-sided / Laplace noise on histograms,
 * Laplace noise on sensitive values).
 *
 * WHAT THIS IS, AND WHAT IT IS NOT. A differential-privacy guarantee is PROVED, not measured; sweeping epsilon only
 * shows the utility trade-off. The OSDP paper (Doudalis et al., VLDB 2017) proves its mechanisms (Theorems 3.1, 3.3, 4.1, 5.2) and
 * uses its experiments for utility only. This class therefore does three different things:
 *   1. STATES the privacy unit, the neighbour relation, the theorem behind each release and the assumptions (section 1-2);
 *   2. CHECKS that the implementation and the policy satisfy the conditions of those theorems (invariants, coin calibration,
 *      budget accounting) and that no violation is detectable (audits with a one-direction test and negative controls);
 *   3. QUANTIFIES where the theorems' assumptions are at risk (overlapping paths, the open case of the paper) and gives
 *      a provable alternative (OsdpRR on the base edges, then run the query on the released sample).
 * PASS means "no violation detected". It is never a proof. The FAIL / WARN lines are the ones to act on.
 *
 * AUDIT DIRECTION. One-sided DP requires only P[E | D] <= e^eps P[E | D'] with D' = D with a sensitive record replaced by any record.
 * The audits use that direction (DPMetrics.audit(..., oneSided = true)). For symmetric mechanisms (Laplace on a value inside a public
 * range) both directions are tested. The ratio eps-hat / eps reads like DP4SQL's under- and over-protection check:
 * > 1 means a violation (under-protection), about 1 is tight, well below 1 is conservative (over-protection, noise larger than needed).
 */
public final class PrivacyAssessment {

    private PrivacyAssessment() {}

    public enum Status { PASS, FAIL, WARN, INFO }

    private record Finding(String id, Status status, String what, String detail) {}

    /** The sensitive numeric property with its PUBLIC range and the nodes that own sensitive values (null = endpoints of sensitive edges). */
    public record ValueSpec(String property, double min, double max, boolean integer, Predicate<Node> owner) {}

    // =====================================================================================
    //  Snapshot
    // =====================================================================================
    private static final class Row {
        final String id;
        final boolean sensitive;
        final String[] nodeIds, edgeIds;
        final boolean[] edgeSens;
        Row(final String id, final boolean sensitive, final String[] nodeIds, final String[] edgeIds, final boolean[] edgeSens) {
            this.id = id; this.sensitive = sensitive; this.nodeIds = nodeIds; this.edgeIds = edgeIds; this.edgeSens = edgeSens;
        }
    }

    private static final class Op {
        String label;
        int depth;
        final Map<String, Row> rows = new LinkedHashMap<>();
    }

    private static final class Snap {
        final List<Op> ops = new ArrayList<>();
        final Map<String, Map<String, String>> props = new HashMap<>();
        final Set<String> sensEdgeEndpoints = new HashSet<>();
        final Set<String> ownerNodes = new HashSet<>();
        Set<String> owners;
    }

    private static Snap snapshot(final String query, final SensitivePatternPolicy pattern, final Predicate<Node> owner) throws Exception {
        try {
            LogicalOperator root = IntermediateResultsExplainer.parseToLogicalRoot(query);
            Snap s = new Snap();
            collect(root, 0, pattern, owner, s);
            s.owners = (owner == null) ? s.sensEdgeEndpoints : s.ownerNodes;
            return s;
        } finally {
            Tools.resetContext();
        }
    }

    private static void collect(final LogicalOperator node, final int depth, final SensitivePatternPolicy pattern,
                                final Predicate<Node> owner, final Snap s) {
        if (node instanceof UnaryLogicalOperator u) {
            collect(u.getChild(), depth + 1, pattern, owner, s);
        } else if (node instanceof BinaryLogicalOperator b) {
            collect(b.getLeftChild(), depth + 1, pattern, owner, s);
            collect(b.getRightChild(), depth + 1, pattern, owner, s);
        }
        Op op = new Op();
        op.label = node.getClass().getSimpleName();
        op.depth = depth;
        for (Path p : IntermediateResultsExplainer.materialize(node)) {
            StringBuilder id = new StringBuilder();
            List<String> nids = new ArrayList<>(), eids = new ArrayList<>();
            List<Boolean> esens = new ArrayList<>();
            for (GraphObject go : p.getSequence()) {
                if (go instanceof Edge e) {
                    id.append("E:").append(e.getId()).append('|');
                    eids.add(e.getId());
                    esens.add(pattern.isSensitiveEdge(e));
                } else if (go instanceof Node nd) {
                    id.append("N:").append(nd.getId()).append('|');
                    nids.add(nd.getId());
                    s.props.putIfAbsent(nd.getId(), nd.getProperties() == null ? new HashMap<>() : new HashMap<>(nd.getProperties()));
                    if (owner != null && owner.test(nd)) s.ownerNodes.add(nd.getId());
                }
            }
            boolean[] es = new boolean[esens.size()];
            for (int i = 0; i < es.length; i++) {
                es[i] = esens.get(i);
                if (es[i] && i + 1 < nids.size()) { s.sensEdgeEndpoints.add(nids.get(i)); s.sensEdgeEndpoints.add(nids.get(i + 1)); }
            }
            String rid = id.toString();
            op.rows.putIfAbsent(rid, new Row(rid, pattern.isSensitivePath(p), nids.toArray(new String[0]), eids.toArray(new String[0]), es));
        }
        s.ops.add(op);   // post-order: children first, root LAST
    }

    // =====================================================================================
    //  Entry point
    // =====================================================================================
    /**
     * @param value             the sensitive numeric property and its public range (null = skip the value checks)
     * @param histogramReleases how many noisy histograms / counts you release from the same data (for the budget statement)
     * @param auditSamples      samples per world in every audit (>= 20000 recommended)
     */
    public static void run(final String query, final SensitivePatternPolicy pattern, final double[] epsilons, final ValueSpec value,
                           final int histogramReleases, final int auditSamples, final long seed) throws Exception {
        System.out.println("=== PRIVACY ASSESSMENT of the selective operator-level release ===");
        System.out.println("query: " + query + "\n");
        Snap s = snapshot(query, pattern, value == null ? null : value.owner());
        List<Finding> F = new ArrayList<>();
        statement(value);

        System.out.println("3. CHECKS\n");
        policyConsistency(s, F);
        invariants(s, epsilons[0], seed, F);
        for (double eps : epsilons) {
            System.out.printf("%n--- epsilon = %.3f ---%n", eps);
            coinCalibration(eps, auditSamples, seed, F);
            accounting(s, value, eps, histogramReleases, seed, F);
            audits(value, eps, auditSamples, seed, F);
        }
        System.out.println();
        baseEdge(s, epsilons, 200, seed, F);
        dependency(s, F);
        verdict(F);
    }

    // =====================================================================================
    //  1-2: statement of the guarantee
    // =====================================================================================
    private static void statement(final ValueSpec v) {
        System.out.println("1. PRIVACY UNIT, POLICY AND NEIGHBOURS (what the guarantee is about)");
        System.out.println("   records     : the result paths of every operator (path-level), the unit OsdpRR is applied to in this code.");
        System.out.println("   policy P    : P(r) = sensitive iff pattern.isSensitivePath(r). The fact that a record is sensitive must stay hidden.");
        System.out.println("   neighbours  : D' = D with ONE sensitive record r replaced by ANY record r' (one-sided, OSDP Definition 3.2).");
        System.out.println("   guarantee   : P[M(D) in O] <= e^eps * P[M(D') in O] (OSDP Definition 3.3), one direction only.");
        System.out.println("   ASSUMPTION A1 (independence): records are independent. Result paths share edges and nodes, so A1 is an ASSUMPTION here, not a fact;");
        System.out.println("                 the OSDP paper lists correlated records (a non-sensitive item reachable only through sensitive ones) as open.");
        System.out.println("   PROVABLE ANCHOR: apply OsdpRR to the BASE EDGES, then run the query on the released edge sample. The query is post-processing, so");
        System.out.println("                 an edge-level (P, eps)-OSDP guarantee follows from Theorem 4.1 with no independence assumption on paths. Cost: a path of");
        System.out.println("                 L edges survives with probability p^L (section 3, base-edge RR).");
        if (v != null) {
            System.out.printf ("   values      : %s in the PUBLIC range [%.0f, %.0f]; Laplace(0, Delta/eps), Delta = %.0f, once per (node, property).%n", v.property(), v.min(), v.max(), v.max() - v.min());
            System.out.println("                 The guarantee covers the VALUE only; the structure of the result is not hidden by this mechanism.");
        }
        System.out.println();
        System.out.println("2. WHICH THEOREM COVERS WHICH RELEASE");
        System.out.println("   rows, OsdpRR (path level)          : OSDP Thm 4.1 per record, under A1. Tracked release across operators = ONE OsdpRR run + post-processing,");
        System.out.println("                                        so a record costs eps once; untracked, the operators compose: sum of eps (Thm 3.3), guarantee for Pmr.");
        System.out.println("   rows, OsdpRR on base edges         : OSDP Thm 4.1 at edge level; query on the sample = post-processing. No A1 on paths.");
        System.out.println("   histogram, OSDP-Laplace T_k        : proof of OSDP Thm 5.2 with L1 change <= k: density ratio <= exp(eps * L1 / k) <= e^eps (scale k/eps).");
        System.out.println("   histogram, DP-post LM T_k          : Laplace mechanism, L1 sensitivity 2k (replace-one), scale 2k/eps.");
        System.out.println("   several histograms from the data   : sequential composition: each noisy histogram costs eps; ONE RR sample costs eps for any number of histograms.");
        System.out.println("   sensitive values, Laplace          : Laplace mechanism, sensitivity = public range, per (node, property); properties of one node add up.");
        System.out.println("   no exclusion attack                : OSDP Thm 3.1: posterior-odds gain <= e^eps, under a product-form prior (A1 again).");
        System.out.println("   T* (oracle k) arms                 : look at the data: NO guarantee; never part of a privacy claim.\n");
    }

    // =====================================================================================
    //  3a: policy consistency
    // =====================================================================================
    private static void policyConsistency(final Snap s, final List<Finding> F) {
        Set<String> seen = new HashSet<>();
        int total = 0, leak = 0, pathOnly = 0, conflicts = 0;
        Map<String, Boolean> label = new HashMap<>();
        List<String> example = new ArrayList<>();
        for (Op op : s.ops) {
            for (Row r : op.rows.values()) {
                Boolean prev = label.putIfAbsent(r.id, r.sensitive);
                if (prev != null && prev != r.sensitive) conflicts++;
                if (!seen.add(r.id)) continue;
                total++;
                boolean any = false;
                for (boolean b : r.edgeSens) any |= b;
                if (!r.sensitive && any) { leak++; if (example.size() < 3) example.add(r.id); }
                if (r.sensitive && !any) pathOnly++;
            }
        }
        System.out.printf("a) POLICY CONSISTENCY over %d distinct records%n", total);
        System.out.printf("   non-sensitive path that contains a sensitive edge : %d   <- RR would release such a path with probability p and disclose the edge%n", leak);
        if (!example.isEmpty()) System.out.println("      examples: " + example);
        System.out.printf("   sensitive path without any sensitive edge          : %d   (a path-level rule; fine, but the policy is then not edge-driven)%n", pathOnly);
        System.out.printf("   same record labelled differently by two operators  : %d%n", conflicts);
        F.add(new Finding("policy-edge-consistency", leak == 0 ? Status.PASS : Status.FAIL, "isSensitivePath covers every sensitive edge",
                leak == 0 ? "no non-sensitive path contains a sensitive edge" : leak + " non-sensitive paths contain a sensitive edge: they can be released and disclose it"));
        F.add(new Finding("policy-label-conflict", conflicts == 0 ? Status.PASS : Status.FAIL, "one sensitivity label per record across operators",
                conflicts == 0 ? "consistent" : conflicts + " records carry two labels"));
        if (pathOnly > 0) F.add(new Finding("policy-path-only", Status.INFO, "path-level sensitive rows", pathOnly + " rows are sensitive without a flagged edge"));
    }

    // =====================================================================================
    //  3b: invariants of the mechanism
    // =====================================================================================
    private static void invariants(final Snap s, final double eps, final long seed, final List<Finding> F) {
        int released = 0, mismatch = 0, runs = 200;
        for (int i = 0; i < runs; i++) {
            RecordReleaseTracker a = new RecordReleaseTracker(DPMetrics.mix(seed, i, 11));
            RecordReleaseTracker b = new RecordReleaseTracker(DPMetrics.mix(seed, i, 11));
            for (Op op : s.ops) {
                for (Row r : op.rows.values()) {
                    boolean x = a.release(r.id, r.sensitive, eps).released();
                    boolean y = b.release(r.id, r.sensitive, eps).released();
                    if (x != y) mismatch++;
                    if (r.sensitive && x) released++;
                }
            }
            for (Op op : s.ops) {                                   // a second pass: the tracker must return the SAME decision
                for (Row r : op.rows.values()) if (a.release(r.id, r.sensitive, eps).released() != b.release(r.id, r.sensitive, eps).released()) mismatch++;
            }
        }
        System.out.printf("%nb) INVARIANTS over %d independent runs: sensitive rows released = %d; non-reproducible or inconsistent decisions = %d%n", runs, released, mismatch);
        F.add(new Finding("inv-no-sensitive-release", released == 0 ? Status.PASS : Status.FAIL, "a sensitive row is never released", released + " violations"));
        F.add(new Finding("inv-reproducible", mismatch == 0 ? Status.PASS : Status.FAIL, "decisions are deterministic and reused", mismatch + " mismatches"));
    }

    // =====================================================================================
    //  3c: calibration of the coin
    // =====================================================================================
    private static void coinCalibration(final double eps, final int n, final long seed, final List<Finding> F) {
        double p = RecordReleaseTracker.releaseProbability(eps);
        int k1 = 0, k2 = 0, kPair = 0, pairs = n / 2;
        for (int i = 0; i < n; i++) if (new RecordReleaseTracker(DPMetrics.mix(seed, i, 61)).release("fixed-target", false, eps).released()) k1++;
        RecordReleaseTracker t = new RecordReleaseTracker(DPMetrics.mix(seed, 0, 62));
        boolean[] rel = new boolean[n];
        for (int i = 0; i < n; i++) { rel[i] = t.release("id-" + i, false, eps).released(); if (rel[i]) k2++; }
        for (int i = 0; i + 1 < n; i += 2) if (rel[i] && rel[i + 1]) kPair++;
        boolean ok1 = within(k1, n, p), ok2 = within(k2, n, p), ok3 = within(kPair, pairs, p * p);
        System.out.printf("c) COIN CALIBRATION (n = %d): p = 1 - e^-eps = %.5f%n", n, p);
        System.out.printf("   one record, many seeds : released %.5f   95%% CI [%.5f, %.5f]  %s%n", k1 / (double) n, DPMetrics.cpLower(k1, n, 0.025), DPMetrics.cpUpper(k1, n, 0.025), ok1 ? "ok" : "OFF");
        System.out.printf("   many records, one seed : released %.5f   95%% CI [%.5f, %.5f]  %s%n", k2 / (double) n, DPMetrics.cpLower(k2, n, 0.025), DPMetrics.cpUpper(k2, n, 0.025), ok2 ? "ok" : "OFF");
        System.out.printf("   neighbouring records   : both released %.5f vs p^2 = %.5f (independence)  %s%n", kPair / (double) pairs, p * p, ok3 ? "ok" : "OFF");
        F.add(new Finding(String.format("coin-%.3f", eps), (ok1 && ok2 && ok3) ? Status.PASS : Status.FAIL, String.format("coin probability and independence at eps = %.3f", eps),
                (ok1 && ok2 && ok3) ? "release rate equals p within its 95% interval; neighbouring records look independent" : "the hash coin deviates from p or is correlated: the e^eps bound of OsdpRR is not met"));
    }

    private static boolean within(final int k, final int n, final double p) {
        return p >= DPMetrics.cpLower(k, n, 0.025) && p <= DPMetrics.cpUpper(k, n, 0.025);
    }

    // =====================================================================================
    //  3d: budget accounting
    // =====================================================================================
    private static void accounting(final Snap s, final ValueSpec v, final double eps, final int hist, final long seed, final List<Finding> F) {
        RecordReleaseTracker rt = new RecordReleaseTracker(DPMetrics.mix(seed, 0, Double.doubleToLongBits(eps)));
        AttributeReleaseTracker vt = new AttributeReleaseTracker(DPMetrics.mix(seed, 0, 21));
        for (Op op : s.ops) {
            Set<String> seen = new HashSet<>();
            for (Row r : op.rows.values()) {
                rt.release(r.id, r.sensitive, eps);
                if (v == null) continue;
                for (String nid : r.nodeIds) {
                    if (!s.owners.contains(nid)) continue;
                    Map<String, String> pr = s.props.get(nid);
                    String raw = pr == null ? null : pr.get(v.property());
                    if (raw == null || !seen.add(nid)) continue;
                    try { vt.release(nid, v.property(), Double.parseDouble(raw.trim()), v.min(), v.max(), v.integer(), eps); } catch (NumberFormatException ignore) { /* skip */ }
                }
            }
        }
        RecordReleaseTracker.LedgerStats ls = rt.ledger(true);
        System.out.printf("d) BUDGET ACCOUNTING%n   rows   : %d sensitive records; composed eps per record: tracked max %.3f, untracked max %.3f; %d records emitted by more than one operator (max %d)%n",
                ls.records(), ls.maxTracked(), ls.maxUntracked(), rt.recordsRequestedMoreThanOnce(), rt.maxRequestsPerRecord());
        F.add(new Finding(String.format("acct-rows-%.3f", eps), Math.abs(ls.maxTracked() - eps) < 1e-9 ? Status.PASS : Status.FAIL,
                String.format("each sensitive record spends eps once (eps = %.3f)", eps),
                String.format("tracked %.3f; without tracking it would reach %.3f", ls.maxTracked(), ls.maxUntracked())));
        if (v != null) {
            AttributeReleaseTracker.LedgerStats vl = vt.ledger();
            System.out.printf("   values : %d released, %d requests, %d reused; composed eps per node: tracked max %.3f, untracked max %.3f%n", vl.cells(), vl.requests(), vl.reused(), vl.maxNodeTracked(), vl.maxNodeUntracked());
            F.add(new Finding(String.format("acct-values-%.3f", eps), vl.maxNodeTracked() <= eps + 1e-9 ? Status.PASS : Status.FAIL,
                    String.format("each sensitive value spends eps once (eps = %.3f)", eps), String.format("tracked %.3f, untracked %.3f", vl.maxNodeTracked(), vl.maxNodeUntracked())));
        }
        System.out.printf("   %d noisy histogram/count release(s) from the same data: OSDP-Laplace or LM cost %.3f in total (sequential composition); one RR sample costs %.3f for any number of histograms%n", hist, hist * eps, eps);
        if (hist > 1) F.add(new Finding(String.format("acct-hist-%.3f", eps), Status.WARN, String.format("%d noisy histograms compose", hist),
                String.format("total eps = %.3f for the noise arms; the RR sample stays at %.3f", hist * eps, eps)));
    }

    // =====================================================================================
    //  3e: audits (empirical, one-direction for one-sided mechanisms)
    // =====================================================================================
    private static void audits(final ValueSpec v, final double eps, final int n, final long seed, final List<Finding> F) {
        final double tol = Math.max(0.05, 0.1 * eps);
        final double cap = Math.log(n / 2.0 / 3.7);
        System.out.printf("e) AUDITS (%d samples per world, 95%% lower bound eps-hat; cap of the test about %.1f)%n", n, cap);

        // OsdpRR exclusion test: world D has the SENSITIVE record, world D' has a non-sensitive replacement
        double[] w1 = new double[n], w0 = new double[n], wExact = new double[n];
        for (int i = 0; i < n; i++) {
            w1[i] = new RecordReleaseTracker(DPMetrics.mix(seed, i, 71)).release("target", true, eps).released() ? 1 : 0;
            w0[i] = new RecordReleaseTracker(DPMetrics.mix(seed, i, 72)).release("target", false, eps).released() ? 1 : 0;
            wExact[i] = 1;                                           // all-NS: the non-sensitive replacement is always released
        }
        DPMetrics.AuditResult rr = DPMetrics.audit(w1, w0, 0.05, true);
        DPMetrics.AuditResult ctl = DPMetrics.audit(w1, wExact, 0.05, true);
        System.out.printf("   OsdpRR exclusion test      : eps-hat %.3f   ratio eps-hat/eps = %.2f   %s%n", rr.epsLower(), rr.epsLower() / eps, rr.epsLower() <= eps + tol ? "no violation" : "VIOLATION");
        System.out.printf("   control, all-NS exact      : eps-hat %.3f (cap %.1f)   %s%n", ctl.epsLower(), cap, ctl.epsLower() > eps + 1 ? "unbounded leakage detected: the test has power" : "NOT detected: sample too small for this eps");
        F.add(new Finding(String.format("audit-rr-%.3f", eps), rr.epsLower() <= eps + tol ? Status.PASS : Status.FAIL, String.format("OsdpRR exclusion audit at eps = %.3f", eps),
                String.format("eps-hat %.3f vs eps %.3f (ratio %.2f)", rr.epsLower(), eps, rr.epsLower() / eps)));
        F.add(new Finding(String.format("control-exact-%.3f", eps), ctl.epsLower() > eps + 1 ? Status.INFO : Status.WARN, "negative control (exact release of the non-sensitive part)",
                String.format("eps-hat %.3f: %s", ctl.epsLower(), ctl.epsLower() > eps + 1 ? "detected as unbounded, so a violation would be visible" : "too few samples to separate it at this eps")));

        // OSDP-Laplace histogram bin, k = 1: x versus x + 1 (a sensitive row replaced by a non-sensitive one adds 1 to the bin)
        double sc = 1.0 / eps;
        double[] h1 = new double[n], h0 = new double[n];
        for (int i = 0; i < n; i++) {
            h1[i] = HistogramEvaluator.osdpOut(5, sc, HistogramEvaluator.expDet(DPMetrics.mix(seed, i, 91), "bin", sc));
            h0[i] = HistogramEvaluator.osdpOut(6, sc, HistogramEvaluator.expDet(DPMetrics.mix(seed, i, 92), "bin", sc));
        }
        DPMetrics.AuditResult ol = DPMetrics.audit(h1, h0, 0.05, true);
        DPMetrics.AuditResult olTwo = DPMetrics.audit(h1, h0, 0.05, false);
        System.out.printf("   OSDP-Laplace bin (k = 1)   : eps-hat %.3f   ratio %.2f   %s%n", ol.epsLower(), ol.epsLower() / eps, ol.epsLower() <= eps + tol ? "no violation" : "VIOLATION");
        System.out.printf("   same, BOTH directions      : eps-hat %.3f   (the reverse direction is unbounded by design for a one-sided mechanism: not a violation)%n", olTwo.epsLower());
        F.add(new Finding(String.format("audit-olap-%.3f", eps), ol.epsLower() <= eps + tol ? Status.PASS : Status.FAIL, String.format("OSDP-Laplace histogram audit at eps = %.3f", eps),
                String.format("one-direction eps-hat %.3f vs eps %.3f (ratio %.2f); two-direction %.3f is expected to be larger", ol.epsLower(), eps, ol.epsLower() / eps, olTwo.epsLower())));

        // Laplace on a sensitive value: neighbours at the two ends of the public range
        if (v != null) {
            double[] a = new double[n], b = new double[n], bad = new double[n], bad2 = new double[n];
            SplittableRandom rng = new SplittableRandom(DPMetrics.mix(seed, 5, 5));
            double delta = v.max() - v.min();
            for (int i = 0; i < n; i++) {
                a[i] = new AttributeReleaseTracker(DPMetrics.mix(seed, i, 81)).release("target", v.property(), v.min(), v.min(), v.max(), v.integer(), eps).value();
                b[i] = new AttributeReleaseTracker(DPMetrics.mix(seed, i, 82)).release("target", v.property(), v.max(), v.min(), v.max(), v.integer(), eps).value();
                bad[i] = clamp(v.min() + DPMetrics.laplace(rng, delta / (4 * eps)), v);   // control: noise 4 times too small
                bad2[i] = clamp(v.max() + DPMetrics.laplace(rng, delta / (4 * eps)), v);
            }
            DPMetrics.AuditResult va = DPMetrics.audit(a, b, 0.05, false);
            DPMetrics.AuditResult vc = DPMetrics.audit(bad, bad2, 0.05, false);
            System.out.printf("   value Laplace (range ends) : eps-hat %.3f   ratio %.2f   %s%n", va.epsLower(), va.epsLower() / eps, va.epsLower() <= eps + tol ? "no violation" : "VIOLATION");
            System.out.printf("   control, noise 4x too small: eps-hat %.3f   %s%n", vc.epsLower(), vc.epsLower() > eps + tol ? "violation detected: the test has power" : "NOT detected at this eps/sample size");
            F.add(new Finding(String.format("audit-value-%.3f", eps), va.epsLower() <= eps + tol ? Status.PASS : Status.FAIL, String.format("value Laplace audit at eps = %.3f", eps),
                    String.format("eps-hat %.3f vs eps %.3f (ratio %.2f)", va.epsLower(), eps, va.epsLower() / eps)));
            F.add(new Finding(String.format("control-value-%.3f", eps), vc.epsLower() > eps + tol ? Status.INFO : Status.WARN, "negative control (mis-calibrated value noise)",
                    String.format("eps-hat %.3f: %s", vc.epsLower(), vc.epsLower() > eps + tol ? "detected" : "not detected: increase auditSamples or use a larger eps")));
        }
    }

    private static double clamp(final double x, final ValueSpec v) { return Math.min(Math.max(x, v.min()), v.max()); }

    // =====================================================================================
    //  3f: the provable anchor, OsdpRR on the base edges
    // =====================================================================================
    private static void baseEdge(final Snap s, final double[] epsilons, final int trials, final long seed, final List<Finding> F) {
        Map<String, Boolean> edgeSens = new HashMap<>();
        for (Op op : s.ops) for (Row r : op.rows.values()) for (int i = 0; i < r.edgeIds.length; i++) edgeSens.merge(r.edgeIds[i], r.edgeSens[i], Boolean::logicalOr);
        Op root = s.ops.get(s.ops.size() - 1);
        int N = root.rows.size();
        if (N == 0) return;
        List<Row> ns = new ArrayList<>();
        for (Row r : root.rows.values()) if (!r.sensitive) ns.add(r);
        System.out.printf("f) PROVABLE ANCHOR: OsdpRR on the BASE EDGES (%d distinct edges, %d sensitive), then the query runs on the released sample (post-processing)%n", edgeSens.size(),
                edgeSens.values().stream().filter(b -> b).count());
        System.out.printf("   %8s %6s | %14s %14s %14s | %s%n", "eps", "p", "path-level RR", "base-edge RR", "base-edge exp.", "released rows that contain a sensitive edge");
        for (double eps : epsilons) {
            double p = RecordReleaseTracker.releaseProbability(eps);
            double meas = 0, expected = 0;
            int bad = 0;
            for (Row r : ns) expected += Math.pow(p, new HashSet<>(java.util.Arrays.asList(r.edgeIds)).size()) / N;
            for (int t = 0; t < trials; t++) {
                RecordReleaseTracker tr = new RecordReleaseTracker(DPMetrics.mix(seed, t, 31));
                Set<String> okEdge = new HashSet<>();
                for (Map.Entry<String, Boolean> e : edgeSens.entrySet()) if (tr.release("EDGE:" + e.getKey(), e.getValue(), eps).released()) okEdge.add(e.getKey());
                int derivable = 0;
                for (Row r : ns) {
                    boolean all = true;
                    for (String id : r.edgeIds) if (!okEdge.contains(id)) { all = false; break; }
                    if (all) {
                        derivable++;
                        for (boolean b : r.edgeSens) if (b) bad++;
                    }
                }
                meas += derivable / (double) N / trials;
            }
            double pathLevel = p * ns.size() / N;
            System.out.printf("   %8.3f %6.3f | %14.4f %14.4f %14.4f | %d%n", eps, p, pathLevel, meas, expected, bad);
            F.add(new Finding(String.format("base-edge-%.3f", eps), bad == 0 ? Status.PASS : Status.FAIL, String.format("base-edge RR releases no sensitive edge (eps = %.3f)", eps),
                    String.format("recall %.4f vs path-level %.4f: the price of the provable guarantee is a factor %.2f", meas, pathLevel, pathLevel == 0 ? Double.NaN : meas / pathLevel)));
        }
        System.out.println("   Reading: the edge-level guarantee holds without any independence assumption on paths; the utility cost is the drop from p to about p^L per row.");
        System.out.println("   Use this arm as the formal baseline and present operator-level path RR as an optimisation whose guarantee rests on A1.\n");
    }

    // =====================================================================================
    //  3g: how much is assumption A1 at risk?
    // =====================================================================================
    private static void dependency(final Snap s, final List<Finding> F) {
        Op root = s.ops.get(s.ops.size() - 1);
        Set<String> sEdges = new HashSet<>(), sNodes = new HashSet<>();
        int nSens = 0, nNs = 0, shareEdge = 0, shareNode = 0;
        for (Row r : root.rows.values()) {
            if (!r.sensitive) continue;
            nSens++;
            sEdges.addAll(java.util.Arrays.asList(r.edgeIds));
            sNodes.addAll(java.util.Arrays.asList(r.nodeIds));
        }
        for (Row r : root.rows.values()) {
            if (r.sensitive) continue;
            nNs++;
            boolean e = false, n = false;
            for (String id : r.edgeIds) if (sEdges.contains(id)) { e = true; break; }
            for (String id : r.nodeIds) if (sNodes.contains(id)) { n = true; break; }
            if (e) shareEdge++;
            if (n) shareNode++;
        }
        System.out.println("g) DEPENDENCY PROBE for assumption A1 (independent records), final result");
        System.out.printf("   %d sensitive rows, %d non-sensitive rows%n", nSens, nNs);
        System.out.printf("   non-sensitive rows sharing an EDGE with a sensitive row : %d (%.1f%%)%n", shareEdge, nNs == 0 ? 0.0 : 100.0 * shareEdge / nNs);
        System.out.printf("   non-sensitive rows sharing a NODE with a sensitive row  : %d (%.1f%%)%n", shareNode, nNs == 0 ? 0.0 : 100.0 * shareNode / nNs);
        F.add(new Finding("dep-edge", shareEdge == 0 ? Status.PASS : Status.WARN, "non-sensitive rows share no edge with sensitive rows",
                shareEdge == 0 ? "no shared edge" : shareEdge + " rows contain an edge that also belongs to a sensitive row: the records are not independent (A1 fails for them); rely on the base-edge arm for the guarantee"));
        F.add(new Finding("dep-node", shareNode == 0 ? Status.PASS : Status.INFO, "non-sensitive rows share no node with sensitive rows",
                shareNode == 0 ? "no shared node" : shareNode + " rows mention nodes that also occur in sensitive rows (an attacker learns the node exists and is connected elsewhere); report it as a correlation channel of A1"));
        System.out.println();
    }

    // =====================================================================================
    //  verdict
    // =====================================================================================
    private static void verdict(final List<Finding> F) {
        System.out.println("4. VERDICT (PASS = no violation detected; it is NOT a proof)");
        System.out.printf("   %-6s %-46s | %s%n", "status", "check", "detail");
        int fail = 0, warn = 0;
        for (Finding f : F) {
            if (f.status() == Status.FAIL) fail++;
            if (f.status() == Status.WARN) warn++;
            System.out.printf("   [%-4s] %-46s | %s%n", f.status(), clipTo(f.what(), 46), f.detail());
        }
        System.out.printf("%n   %d FAIL, %d WARN. Act on FAIL first (the guarantee is broken as implemented), then WARN (an assumption is at risk).%n", fail, warn);
        System.out.println("   STILL TO BE STATED / PROVED IN THE PAPER: (1) the neighbour relation of section 1; (2) one theorem per release of section 2 with its proof;");
        System.out.println("   (3) the tracked-composition argument (one OsdpRR run plus post-processing); (4) assumption A1 or the base-edge anchor; (5) the list of public parameters");
        System.out.println("   (epsilon, ranges, k); (6) that T* oracle arms and rescaling choices are outside the claim.");
        System.out.println("   READING eps-hat / eps: > 1 is a violation (under-protection); about 1 is tight; well below 1 means noise larger than needed (over-protection), e.g. a bin with k > 1.\n");
    }

    private static String clipTo(final String s, final int w) { return s.length() <= w ? s : s.substring(0, w - 3) + "..."; }

    // =====================================================================================
    //  Example
    // =====================================================================================
    public static void main(String[] args) throws Exception {
        SensitivePatternPolicy pattern = new SensitivePatternPolicy();
        pattern.addSensitiveNodeName("President");
        String query = "MATCH ACYCLIC p = (x)-[hasCreator?.knows*]{..6}->(y) WHERE x.name = \"Karim\" OR x.txt = \"Msg1\" RETURN x.name, NODE(4).name, y.birthYear;";
        ValueSpec birthYear = new ValueSpec("birthYear", 1900, 2026, true,
                nd -> nd.getProperties() != null && "President".equals(nd.getProperties().get("name")));

        Tools.loadDefaultGraph();
        // epsilons, number of noisy histograms you plan to release, audit samples per world, seed
        run(query, pattern, new double[]{0.5, 1.0, 2.0}, birthYear, 7, 20000, 42L);
    }
}