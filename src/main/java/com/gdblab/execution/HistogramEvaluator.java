package com.gdblab.execution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;

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
 * HISTOGRAM STUDY: counting queries and n-histograms over the rows PathDB returns, evaluated the way the one-sided DP
 * paper (Doudalis et al., VLDB 2017) does.
 *
 * WHY THIS IS DONE OUTSIDE THE ENGINE. PathDB has no GROUP BY / COUNT: it returns paths (rows). A histogram is computed on
 * the retrieved rows,      H(b) = #{ rows r : b in bins(r) }      (a row counts once per bin),
 * where bins(r) is one bin for a counting query or a 1-D / categorical / joint histogram, and up to L-n+1 bins for an
 * n-gram histogram (n consecutive nodes of the path). A counting query is a histogram with a single bin.
 * Differential privacy can then be placed on the histogram (noise) or on the rows (random sampling); binning released
 * rows afterwards is free post-processing.
 *
 * POLICY. A row is SENSITIVE iff pattern.isSensitivePath(path). H_NS is the histogram of the non-sensitive rows only.
 *
 * ARMS (all measured against the FULL ground truth H, sensitive rows included). eps, replace-one-record neighbours,
 * k = the number of bins one row may contribute (truncation keeps the first k bins of a row; k = 1 for counting queries).
 *   all-NS exact      : H_NS, no noise. NOT private (a missing row reveals it was sensitive). The floor of every one-sided method.
 *   OSDP-Laplace T_k  : y = H_NS,k(b) - Exp(k/eps);  out = y + (k/eps) ln 2 if y > 0, else 0.
 *                       Replacing a sensitive row by any row can only ADD at most k to the non-sensitive histogram and never remove,
 *                       so noise in ONE direction with scale k/eps suffices (OsdpLaplaceL1). Its variance (k/eps)^2 is 1/8 of
 *                       the two-sided Laplace(2k/eps) variance 2 (2k/eps)^2, and an EMPTY bin stays exactly 0.
 *   OSDP-RR           : each non-sensitive row is released with probability p = 1 - e^-eps (a RecordReleaseTracker makes it one
 *                       decision per row); the histogram of the released rows is reported raw and divided by p (inverse-probability).
 *   DP-post LM T_k    : H_k(b) + Laplace(0, 2k/eps) on EVERY bin of the domain, all rows protected (uniform standard DP;
 *                       L1 sensitivity 2k under replace-one neighbours).
 *   T* variants       : the best k by oracle (looks at the data: NOT private), as in the paper.
 *
 * METRICS
 *   RelError(b)  = |H(b) - Hhat(b)| / max(1, H(b))      (DP4SQL; equal to OSDP's relative error with delta = 1)
 *   MRE(domain)  = [ sum over active bins RelError + (d - B) * E|noise on an empty bin| ] / d      (the paper's metric; d = domain size,
 *                  B = number of non-empty bins; E|noise| = 2k/eps for LM and 0 for the one-sided arms, added analytically)
 *   MRE(active), Rel50, Rel95 : mean, median and 95th percentile of RelError over the non-empty bins
 *   L1/mass      = sum_b |H - Hhat| / sum_b H : the fraction of the histogram mass that is misplaced
 *   FLOOR        = sum_b |H - H_NS| / sum_b H : what is lost just by not releasing the sensitive rows. L1/mass above the floor is
 *                  the price of the noise (or sampling).
 *   Values are trimmed means over the trials (20% dropped on each side); RelError is also shown as boxplots against the always-0 line.
 *
 * LIMITS. One-sided guarantees protect only the SENSITIVE rows and assume independent records (OSDP); overlapping paths and
 * n-grams of the same path are correlated, which the paper leaves open. DP-post protects every row. All noise is drawn
 * deterministically from (seed, release key), so a bin gets one noise value however often it is requested.
 */
public final class HistogramEvaluator {

    private HistogramEvaluator() {}

    // =====================================================================================
    //  Histogram specifications
    // =====================================================================================
    /** Read-only view of a result row. Positions: 0 = first node, -1 = last node, -2 = second last, ... */
    public interface RowView {
        int length();
        String nodeId(int i);
        String label(int i);
        String prop(int i, String name);
        default int at(final int pos) { return pos >= 0 ? pos : length() + pos; }
    }

    /**
     * name: label. rule: the bin rule in plain words (printed in the output). bins: the bins of a row (distinct, in path order).
     * why: for ONE row, "in bin X | evidence" or "NO BIN: reason | evidence" (printed per row). domain: number of possible bins
     * (0 = derive from the plan; for an n-gram it is the number of possible NODE keys V, the domain being V^n). ngram: n for n-grams, else 0.
     */
    public record Hist(String name, String rule, Function<RowView, List<String>> bins, Function<RowView, String> why, double domain, int ngram) {}

    private static String where(final RowView r, final int i) { return "node " + i + " " + r.nodeId(i) + "(" + r.label(i) + ")"; }

    private static String posText(final int pos) {
        return pos >= 0 ? "position " + pos + " (0 = first node)" : "position " + pos + " (-1 = last node, -2 = second last, ...)";
    }

    private static boolean cmpTest(final String cmp, final double v, final double t) {
        switch (cmp) {
            case ">=": return v >= t;
            case ">": return v > t;
            case "<=": return v <= t;
            case "<": return v < t;
            default: return v == t;
        }
    }

    /** Counting query: one bin "rows" for each row that satisfies the filter (null = every row, i.e. COUNT(*)). */
    public static Hist count(final String name, final Predicate<RowView> filter) {
        Function<RowView, List<String>> bins = r -> (filter == null || filter.test(r)) ? List.of("rows") : List.<String>of();
        return new Hist(name, "ONE bin 'rows'. A row falls in it iff " + (filter == null ? "it is a result row (no filter): this is COUNT(*)." : "the supplied filter holds for the row; every other row falls in NO bin."),
                bins, r -> bins.apply(r).isEmpty() ? "NO BIN: filter is false | the row does not satisfy the filter" : "in bin 'rows' | " + (filter == null ? "no filter" : "the filter holds"),
                1, 0);
    }

    /** Counting query with a numeric condition: COUNT(rows where prop of the node at pos  cmp  threshold), cmp in >=, >, <=, <, ==. */
    public static Hist countWhere(final String name, final String prop, final int pos, final String cmp, final double threshold) {
        Function<RowView, List<String>> bins = r -> {
            int i = r.at(pos);
            if (i < 0 || i >= r.length()) return List.<String>of();
            String raw = r.prop(i, prop);
            if (raw == null) return List.<String>of();
            try { return cmpTest(cmp, Double.parseDouble(raw.trim()), threshold) ? List.of("rows") : List.<String>of(); } catch (NumberFormatException e) { return List.<String>of(); }
        };
        Function<RowView, String> why = r -> {
            int i = r.at(pos);
            if (i < 0 || i >= r.length()) return "NO BIN: position missing | " + posText(pos) + " does not exist, the path has " + r.length() + " nodes";
            String raw = r.prop(i, prop);
            if (raw == null) return "NO BIN: property missing | " + where(r, i) + " has no '" + prop + "'";
            double v;
            try { v = Double.parseDouble(raw.trim()); } catch (NumberFormatException e) { return "NO BIN: not numeric | " + where(r, i) + " " + prop + "='" + raw + "'"; }
            boolean ok = cmpTest(cmp, v, threshold);
            return (ok ? "in bin 'rows'" : "NO BIN: condition false") + " | " + where(r, i) + " " + prop + "=" + raw + (ok ? " satisfies " : " does not satisfy ") + cmp + " " + String.format("%.0f", threshold);
        };
        return new Hist(name, String.format("ONE bin 'rows'. A row falls in it iff '%s' of the node at %s exists, is numeric and %s %.0f. Every other row falls in NO bin: the other rows of the result are not counted.",
                prop, posText(pos), cmp, threshold), bins, why, 1, 0);
    }

    /** 1-D histogram of a numeric property of the node at position pos, in buckets of the given width over [min, max]. */
    public static Hist numeric(final String name, final String prop, final int pos, final double min, final double max, final double width) {
        final double d = Math.ceil((max - min + 1) / width);
        Function<RowView, List<String>> bins = r -> {
            int i = r.at(pos);
            if (i < 0 || i >= r.length()) return List.<String>of();
            String raw = r.prop(i, prop);
            if (raw == null) return List.<String>of();
            double v;
            try { v = Double.parseDouble(raw.trim()); } catch (NumberFormatException e) { return List.<String>of(); }
            double c = Math.min(Math.max(v, min), max);
            double lo = min + Math.floor((c - min) / width) * width;
            return List.of(String.format("[%.0f,%.0f)", lo, lo + width));
        };
        Function<RowView, String> why = r -> {
            int i = r.at(pos);
            if (i < 0 || i >= r.length()) return "NO BIN: position missing | " + posText(pos) + " does not exist, the path has " + r.length() + " nodes";
            String raw = r.prop(i, prop);
            if (raw == null) return "NO BIN: property missing | " + where(r, i) + " has no '" + prop + "'";
            List<String> b = bins.apply(r);
            if (b.isEmpty()) return "NO BIN: not numeric | " + where(r, i) + " " + prop + "='" + raw + "'";
            return "in bin " + b.get(0) + " | " + where(r, i) + " " + prop + "=" + raw;
        };
        return new Hist(name, String.format("bins are buckets of width %.0f over the PUBLIC range [%.0f, %.0f]: [%.0f,%.0f), [%.0f,%.0f), ... : d = %.0f possible bins. A row falls in exactly ONE bin iff '%s' of the node at %s exists and is numeric (values outside the range are clamped to the first or last bucket).",
                width, min, max, min, min + width, min + width, min + 2 * width, d, prop, posText(pos)), bins, why, d, 0);
    }

    /** Histogram over the values of a property (null = the node label) of the node at position pos. domain 0 = derive. */
    public static Hist category(final String name, final String prop, final int pos, final double domain) {
        final String what = prop == null ? "label" : "'" + prop + "'";
        Function<RowView, List<String>> bins = r -> {
            int i = r.at(pos);
            if (i < 0 || i >= r.length()) return List.<String>of();
            String k = prop == null ? r.label(i) : r.prop(i, prop);
            return k == null ? List.<String>of() : List.of(k);
        };
        Function<RowView, String> why = r -> {
            int i = r.at(pos);
            if (i < 0 || i >= r.length()) return "NO BIN: position missing | " + posText(pos) + " does not exist, the path has " + r.length() + " nodes";
            List<String> b = bins.apply(r);
            return b.isEmpty() ? "NO BIN: property missing | " + where(r, i) + " has no " + what : "in bin '" + b.get(0) + "' | " + where(r, i) + " " + what + "=" + b.get(0);
        };
        return new Hist(name, "ONE bin per distinct value of the " + what + " of the node at " + posText(pos) + ". A row falls in the bin of its value iff that node exists and has the property."
                + (domain > 0 ? " d = " + (long) domain + " possible values (given)." : " d = the number of distinct values (derived from the plan, a lower bound)."), bins, why, domain, 0);
    }

    /** n-gram histogram: n consecutive node keys (a property, or the node id when prop is null). domainNodes = V (0 = derive). */
    public static Hist ngram(final String name, final int n, final String prop, final double domainNodes) {
        Function<RowView, List<String>> bins = r -> {
            LinkedHashSet<String> out = new LinkedHashSet<>();
            for (int i = 0; i + n <= r.length(); i++) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < n; j++) {
                    String k = prop == null ? r.nodeId(i + j) : r.prop(i + j, prop);
                    sb.append(j == 0 ? "" : ">").append(k == null ? "?" : k);
                }
                out.add(sb.toString());
            }
            return new ArrayList<>(out);
        };
        Function<RowView, String> why = r -> {
            if (r.length() < n) return "NO BIN: path too short | the path has " + r.length() + " nodes, an n-gram needs n = " + n;
            List<String> b = bins.apply(r);
            return "in " + b.size() + " bin(s) " + (b.size() > 2 ? b.subList(0, 2) + "..." : b) + " | " + r.length() + " nodes give " + (r.length() - n + 1) + " n-grams, " + b.size() + " distinct";
        };
        return new Hist(name, "bins are the DISTINCT " + n + "-grams of a path: " + n + " consecutive node keys (" + (prop == null ? "node id" : "'" + prop + "'") + ") joined by '>'. A path with L nodes gives up to L-" + n
                + "+1 bins and counts once in each; a path with fewer than " + n + " nodes falls in NO bin. d = V^" + n + " possible bins, V = the number of possible node keys.", bins, why, domainNodes, n);
    }

    /** Joint (n-dimensional) histogram: the cross product of single-bin histograms. */
    public static Hist joint(final String name, final Hist... parts) {
        double dom = 1;
        for (Hist h : parts) dom = (dom > 0 && h.domain() > 0) ? dom * h.domain() : 0;
        Function<RowView, List<String>> bins = r -> {
            StringBuilder sb = new StringBuilder();
            for (Hist h : parts) {
                List<String> b = h.bins().apply(r);
                if (b.size() != 1) return List.<String>of();
                sb.append(sb.length() == 0 ? "" : " & ").append(b.get(0));
            }
            return List.of(sb.toString());
        };
        Function<RowView, String> why = r -> {
            StringBuilder ev = new StringBuilder();
            for (Hist h : parts) {
                if (h.bins().apply(r).size() != 1) return "NO BIN: part '" + h.name() + "' gives no bin | " + h.why().apply(r);
                ev.append(ev.length() == 0 ? "" : " ; ").append(h.why().apply(r));
            }
            return "in bin " + bins.apply(r).get(0) + " | " + ev;
        };
        StringBuilder names = new StringBuilder();
        for (Hist h : parts) names.append(names.length() == 0 ? "" : " x ").append(h.name());
        return new Hist(name, "bin = the tuple of the parts' bins (" + names + ") joined by ' & '. A row falls in exactly ONE bin iff EVERY part gives exactly one bin; d = the product of the parts' domains.", bins, why, dom, 0);
    }

    // =====================================================================================
    //  Generic specs: built from the query text (RETURN clause + MATCH variables)
    // =====================================================================================
    /**
     * PUBLIC (data-independent) parameters of the properties the histograms are built on.
     *   numeric property     -> {min, max, bucket width, fine bucket width (default 1)}
     *   categorical property -> the number of POSSIBLE values (e.g. how many distinct names the graph can have). It fixes the domain
     *                           d of every histogram that uses the property: the zero bins are d minus the occupied bins, so an
     *                           unregistered property gives a derived (lower-bound) domain and HIDES the zero bins.
     */
    public static final class Ranges {
        final Map<String, double[]> m = new LinkedHashMap<>();
        final Map<String, Double> cat = new LinkedHashMap<>();
        public Ranges add(final String prop, final double min, final double max, final double width) { return add(prop, min, max, width, 1.0); }
        public Ranges add(final String prop, final double min, final double max, final double width, final double fineWidth) {
            m.put(prop, new double[]{min, max, width, fineWidth});
            return this;
        }
        public Ranges categories(final String prop, final double possibleValues) {
            cat.put(prop, possibleValues);
            return this;
        }
    }

    private static Hist midCount(final String label, final String prop, final int pos, final double mid) {
        return countWhere("COUNT(" + label + " >= " + String.format("%.0f", mid) + ")", prop, pos, ">=", mid);
    }

    /** One RETURN item: label, property, position, and the numeric range if the property is numeric. */
    private static final class RetItem {
        final String label, prop;
        final int pos;
        final double[] rg;
        RetItem(final String label, final String prop, final int pos, final double[] rg) { this.label = label; this.prop = prop; this.pos = pos; this.rg = rg; }
    }

    private static List<RetItem> parseReturnItems(final String query, final Ranges ranges, final int nodeIndexBase) {
        String q = query.replaceAll("\\s+", " ").trim();
        int ri = q.toUpperCase().indexOf("RETURN");
        if (ri < 0) throw new IllegalArgumentException("no RETURN clause in the query");
        String head = q.substring(0, ri);
        String tail = q.substring(ri + 6);
        int cut = tail.length();
        for (String kw : new String[]{" LIMIT", ";", " ORDER BY"}) {
            int idx = tail.toUpperCase().indexOf(kw);
            if (idx >= 0) cut = Math.min(cut, idx);
        }
        Map<String, Integer> vars = new HashMap<>();
        java.util.regex.Matcher mf = java.util.regex.Pattern.compile("\\(\\s*(\\w+)\\s*\\)\\s*-\\s*\\[").matcher(head);
        if (mf.find()) vars.put(mf.group(1), 0);
        java.util.regex.Matcher ml = java.util.regex.Pattern.compile("->\\s*\\(\\s*(\\w+)\\s*\\)").matcher(head);
        String lastVar = null;
        while (ml.find()) lastVar = ml.group(1);
        if (lastVar != null) vars.put(lastVar, -1);
        List<RetItem> items = new ArrayList<>();
        for (String raw : tail.substring(0, cut).split(",")) {
            String item = raw.trim();
            if (item.isEmpty()) continue;
            java.util.regex.Matcher mn = java.util.regex.Pattern.compile("^(?i:NODE)\\((\\d+)\\)\\.(\\w+)$").matcher(item);
            java.util.regex.Matcher mv = java.util.regex.Pattern.compile("^(\\w+)\\.(\\w+)$").matcher(item);
            int pos;
            String prop;
            if (mn.matches()) { pos = Integer.parseInt(mn.group(1)) - nodeIndexBase; prop = mn.group(2); }
            else if (mv.matches() && vars.containsKey(mv.group(1))) { pos = vars.get(mv.group(1)); prop = mv.group(2); }
            else { System.out.println("specsFor: RETURN item '" + item + "' is not a property of a known variable or NODE(k): skipped"); continue; }
            items.add(new RetItem(item, prop, pos, ranges == null ? null : ranges.m.get(prop)));
        }
        return items;
    }

    /** specsFor(query, ranges) with NODE(1) = first node and 2- and 3-grams of the node property "name". */
    public static List<Hist> specsFor(final String query, final Ranges ranges) {
        return specsFor(query, ranges, 1, "name", false, 2, 3);
    }

    /**
     * specsFor plus the SPARSE family (see below): finer buckets, all pairwise joint histograms of the RETURN items and the full RETURN tuple.
     * Their domains multiply (d = product of the dimensions) while the occupied bins can never exceed the number of rows, so most bins are EMPTY:
     * the situation in which one-sided noise (empty bins stay 0) can beat Laplace noise on every bin. Run `ladder` on this list.
     */
    public static List<Hist> sparseSpecsFor(final String query, final Ranges ranges) {
        return specsFor(query, ranges, 1, "name", true, 2, 3);
    }

    public static List<Hist> specsFor(final String query, final Ranges ranges, final int nodeIndexBase,
                                      final String ngramProp, final int... ngramSizes) {
        return specsFor(query, ranges, nodeIndexBase, ngramProp, false, ngramSizes);
    }

    /**
     * Builds the histograms of a query from its text, so no spec has to be written by hand:
     *  - COUNT(*) always;
     *  - every RETURN item "var.prop" or "NODE(k).prop": a numeric histogram (plus COUNT(prop >= midpoint of its range)) when `ranges`
     *    knows the property as numeric, otherwise a categorical histogram of its values (domain from Ranges.categories, else derived);
     *  - two joint histograms: (first categorical, first numeric) and (first numeric, second numeric);
     *  - one n-gram histogram of the node property `ngramProp` (null = node id) per requested n;
     *  - if sparse: every numeric item again with the FINE bucket width, every PAIR of items as a joint histogram (numeric items at the fine
     *    width) and, for three or more items, the FULL TUPLE of the RETURN clause as one joint histogram.
     * Variables: the first variable of the MATCH pattern is position 0 and the last one is position -1 (the end of the path).
     * NODE(k) is position k - nodeIndexBase (base 1 means NODE(1) is the first node: check this against your PathDB version).
     * Rows whose path is too short for a position simply fall in no bin of that histogram.
     */
    public static List<Hist> specsFor(final String query, final Ranges ranges, final int nodeIndexBase,
                                      final String ngramProp, final boolean sparse, final int... ngramSizes) {
        List<RetItem> items = parseReturnItems(query, ranges, nodeIndexBase);
        List<Hist> out = new ArrayList<>();
        out.add(count("COUNT(*)", null));
        List<Hist> numeric = new ArrayList<>(), categorical = new ArrayList<>(), parts = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        List<String> missingDomain = new ArrayList<>();
        for (RetItem it : items) {
            if (it.rg != null) {
                Hist h = numeric("HIST " + it.label, it.prop, it.pos, it.rg[0], it.rg[1], it.rg[2]);
                numeric.add(h);
                out.add(h);
                out.add(midCount(it.label, it.prop, it.pos, (it.rg[0] + it.rg[1]) / 2.0));
                double fw = it.rg.length > 3 ? it.rg[3] : 1.0;
                parts.add(numeric("HIST " + it.label + " (width " + String.format("%.0f", fw) + ")", it.prop, it.pos, it.rg[0], it.rg[1], fw));
            } else {
                Double dom = (ranges == null) ? null : ranges.cat.get(it.prop);
                if (dom == null && !missingDomain.contains(it.prop)) missingDomain.add(it.prop);
                Hist h = category("HIST " + it.label, it.prop, it.pos, dom == null ? 0 : dom);
                categorical.add(h);
                out.add(h);
                parts.add(h);
            }
            labels.add(it.label + (it.rg != null ? " (width " + String.format("%.0f", it.rg.length > 3 ? it.rg[3] : 1.0) + ")" : ""));
        }
        if (!categorical.isEmpty() && !numeric.isEmpty()) out.add(joint("JOINT (" + categorical.get(0).name().substring(5) + ", " + numeric.get(0).name().substring(5) + ")", categorical.get(0), numeric.get(0)));
        if (numeric.size() >= 2) out.add(joint("JOINT (" + numeric.get(0).name().substring(5) + ", " + numeric.get(1).name().substring(5) + ")", numeric.get(0), numeric.get(1)));
        for (int n : ngramSizes) out.add(ngram(n + "-gram of " + (ngramProp == null ? "node ids" : ngramProp), n, ngramProp, 0));

        if (sparse) {
            if (!missingDomain.isEmpty()) {
                System.out.println("specsFor(sparse): no domain size registered for " + missingDomain + ". The cross-product domains then use the bins OBSERVED in the plan (a lower bound),");
                System.out.println("                  which hides the zero bins. Register the real number of possible values with Ranges.categories(property, size).");
            }
            for (int i = 0; i < items.size(); i++) if (items.get(i).rg != null) out.add(parts.get(i));
            for (int i = 0; i < parts.size(); i++) {
                for (int j = i + 1; j < parts.size(); j++) {
                    out.add(joint("JOINT2 (" + labels.get(i) + ", " + labels.get(j) + ")", parts.get(i), parts.get(j)));
                }
            }
            if (parts.size() >= 3) out.add(joint("FULL TUPLE (" + String.join(", ", labels) + ")", parts.toArray(new Hist[0])));
        }
        return out;
    }

    // =====================================================================================
    //  Snapshot
    // =====================================================================================
    private static final class Row {
        final String id;
        boolean sensitive;
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
        final Map<String, String> labels = new HashMap<>();
    }

    private static final class View implements RowView {
        final Row r;
        final Snap s;
        View(final Row r, final Snap s) { this.r = r; this.s = s; }
        public int length() { return r.nodeIds.length; }
        public String nodeId(final int i) { return r.nodeIds[i]; }
        public String label(final int i) { return s.labels.get(r.nodeIds[i]); }
        public String prop(final int i, final String name) {
            Map<String, String> p = s.props.get(r.nodeIds[i]);
            return p == null ? null : p.get(name);
        }
    }

    private static Snap snapshot(final String query, final SensitivePatternPolicy pattern) throws Exception {
        try {
            LogicalOperator root = IntermediateResultsExplainer.parseToLogicalRoot(query);
            Snap s = new Snap();
            collect(root, 0, pattern, s);
            return s;
        } finally {
            Tools.resetContext();
        }
    }

    private static int collect(final LogicalOperator node, final int depth, final SensitivePatternPolicy pattern, final Snap s) {
        List<Integer> kids = new ArrayList<>();
        if (node instanceof UnaryLogicalOperator u) {
            kids.add(collect(u.getChild(), depth + 1, pattern, s));
        } else if (node instanceof BinaryLogicalOperator b) {
            kids.add(collect(b.getLeftChild(), depth + 1, pattern, s));
            kids.add(collect(b.getRightChild(), depth + 1, pattern, s));
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
                    s.labels.putIfAbsent(nd.getId(), String.valueOf(nd.getLabel()));
                }
            }
            boolean[] es = new boolean[esens.size()];
            for (int i = 0; i < es.length; i++) es[i] = esens.get(i);
            String rid = id.toString();
            op.rows.putIfAbsent(rid, new Row(rid, pattern.isSensitivePath(p), nids.toArray(new String[0]), names.toArray(new String[0]),
                    eids.toArray(new String[0]), elabels.toArray(new String[0]), es));
        }
        s.ops.add(op);   // post-order: children first, root LAST
        return s.ops.size() - 1;
    }

    // =====================================================================================
    //  Histograms and noise
    // =====================================================================================
    private static Map<String, Integer> hist(final Collection<Row> rows, final Map<Row, List<String>> bm, final int trunc) {
        Map<String, Integer> h = new HashMap<>();
        for (Row r : rows) {
            List<String> b = bm.get(r);
            int m = trunc > 0 ? Math.min(trunc, b.size()) : b.size();
            for (int i = 0; i < m; i++) h.merge(b.get(i), 1, Integer::sum);
        }
        return h;
    }

    private static double lapDet(final long seed, final String key, final double scale) {
        double u = RecordReleaseTracker.unit(seed, key) - 0.5;
        double a = Math.min(Math.abs(u), 0.5 - 1e-12);
        return -scale * Math.signum(u) * Math.log(1.0 - 2.0 * a);
    }

    static double expDet(final long seed, final String key, final double scale) {
        double u = Math.min(RecordReleaseTracker.unit(seed, key), 1.0 - 1e-12);
        return -scale * Math.log(1.0 - u);
    }

    /** OsdpLaplaceL1 output for one bin of the non-sensitive histogram. */
    static double osdpOut(final double x, final double scale, final double e) {
        double y = x - e;
        return y > 0 ? y + scale * Math.log(2) : 0.0;
    }

    private static final String[] METRIC = {"MRE(domain)", "MRE(active)", "Rel50", "Rel95", "L1/mass", "total RelErr"};

    /** {MRE domain, MRE active, Rel50, Rel95, L1/mass, total-count RelError} against the truth histogram. */
    private static double[] hm(final Map<String, Integer> truth, final Map<String, Double> est, final double domain,
                               final double zeroMean, final double mass) {
        int nb = truth.size();
        if (nb == 0) { double[] nan = new double[6]; Arrays.fill(nan, Double.NaN); return nan; }
        double[] rel = new double[nb];
        double sum = 0, l1 = 0, estMass = 0;
        int i = 0;
        for (Map.Entry<String, Integer> e : truth.entrySet()) {
            double x = e.getValue();
            double y = est.getOrDefault(e.getKey(), 0.0);
            double r = Math.abs(x - y) / Math.max(1.0, x);
            rel[i++] = r;
            sum += r;
            l1 += Math.abs(x - y);
            estMass += y;
        }
        double d = Math.max(domain, nb);
        double zero = Math.max(0.0, d - nb) * zeroMean;
        return new double[]{(sum + zero) / d, sum / nb, DPMetrics.median(rel), DPMetrics.percentile(rel, 0.95),
                (l1 + zero) / Math.max(1.0, mass), DPMetrics.relError(mass, estMass)};
    }

    private static double tm(final double[] a) {
        double[] c = Arrays.stream(a).filter(x -> !Double.isNaN(x)).toArray();
        return c.length == 0 ? Double.NaN : DPMetrics.trimmedMean(c, 0.2);
    }

    // =====================================================================================
    //  One (histogram, epsilon) evaluation over many trials
    // =====================================================================================
    private static final String[] ARMS = {
            "all-NS exact [not private]",
            "OSDP-Laplace T1",
            "OSDP-Laplace T* [oracle]",
            "OSDP-RR raw",
            "OSDP-RR x 1/p",
            "DP-post LM T1",
            "DP-post LM T* [oracle]"};

    private static final class Eval {
        Hist h;
        double domain;
        int B, rows, sens, K, kStar1, kStar2;
        double mass, floor, trunc1Loss, sensMass, nsMass1;
        String domainSource = "";
        int nsBins1;
        final double[][] m = new double[7][];        // trimmed mean metrics per arm
        final double[][] perTrial = new double[7][]; // MRE(active) per trial per arm
        double releasedShare;                        // RR, mean share of the root rows released
    }

    private static String domainText(final Hist h, final double d) {
        if (h.ngram() > 0) {
            long v = Math.round(Math.pow(d, 1.0 / h.ngram()));
            return (h.domain() > 0 ? "V = " + (long) h.domain() + " node keys (given)" : "V = " + v + " distinct node ids observed in the plan (derived; the real graph may have more)")
                    + ", so d = V^" + h.ngram() + " = " + String.format("%.3e", d) + " possible bins";
        }
        if (h.domain() > 0) return "d = " + (long) d + " possible bins (given by the spec)";
        return "d = " + (long) d + " possible bins = the distinct bins seen in all operators' rows (derived: a LOWER bound; pass the real size to the factory)";
    }

    private static double derivedDomain(final Hist h, final Snap s, final Map<Row, List<String>> bmAll) {
        if (h.ngram() > 0) {
            double v = h.domain();
            if (v <= 0) {
                Set<String> keys = new HashSet<>();
                for (Op op : s.ops) for (Row r : op.rows.values()) keys.addAll(Arrays.asList(r.nodeIds));
                v = keys.size();                                          // node ids; a property key may have fewer values
            }
            return Math.pow(Math.max(v, 1), h.ngram());
        }
        if (h.domain() > 0) return h.domain();
        Set<String> seen = new HashSet<>();
        for (List<String> b : bmAll.values()) seen.addAll(b);
        return Math.max(1, seen.size());
    }

    private static Eval evaluate(final Snap s, final Hist h, final double eps, final int trials, final long seed, final int maxTrunc) {
        final Op root = s.ops.get(s.ops.size() - 1);
        final List<Row> all = new ArrayList<>(root.rows.values());
        final List<Row> ns = new ArrayList<>();
        for (Row r : all) if (!r.sensitive) ns.add(r);
        final Map<Row, List<String>> bm = new HashMap<>();
        for (Row r : all) bm.put(r, h.bins().apply(new View(r, s)));
        final Map<Row, List<String>> bmPlan = new HashMap<>();                    // every operator, only used to derive a domain
        for (Op op : s.ops) for (Row r : op.rows.values()) bmPlan.putIfAbsent(r, bm.containsKey(r) ? bm.get(r) : h.bins().apply(new View(r, s)));

        Eval ev = new Eval();
        ev.h = h;
        ev.rows = all.size();
        ev.sens = all.size() - ns.size();
        ev.domain = derivedDomain(h, s, bmPlan);
        ev.domainSource = domainText(h, ev.domain);
        Map<String, Integer> truth = hist(all, bm, 0);
        ev.B = truth.size();
        double mass = 0;
        for (int v : truth.values()) mass += v;
        ev.mass = mass;
        Map<String, Integer> hNs = hist(ns, bm, 0);
        double nsMass = 0;
        for (int v : hNs.values()) nsMass += v;
        ev.sensMass = mass - nsMass;
        ev.floor = mass == 0 ? Double.NaN : (mass - nsMass) / mass;          // sum |H - H_NS| / sum H (H_NS <= H bin-wise)
        final double p = RecordReleaseTracker.releaseProbability(eps);

        int kmax = 1;
        for (List<String> b : bm.values()) kmax = Math.max(kmax, b.size());
        final int K = Math.min(kmax, Math.max(1, maxTrunc));
        ev.K = K;
        List<Map<String, Integer>> allK = new ArrayList<>(), nsK = new ArrayList<>();
        for (int k = 1; k <= K; k++) { allK.add(hist(all, bm, k)); nsK.add(hist(ns, bm, k)); }
        double mass1 = 0;
        for (int v : allK.get(0).values()) mass1 += v;
        ev.trunc1Loss = mass == 0 ? 0.0 : (mass - mass1) / mass;
        for (int v : nsK.get(0).values()) ev.nsMass1 += v;
        ev.nsBins1 = nsK.get(0).size();

        double[][][] ol = new double[K][6][trials], lm = new double[K][6][trials];
        double[][] rr = new double[6][trials], rs = new double[6][trials];
        double relShare = 0;
        for (int t = 0; t < trials; t++) {
            final long sd = DPMetrics.mix(seed, t, 41);
            // OSDP-Laplace and DP-post LM, every truncation k
            for (int k = 1; k <= K; k++) {
                double sc1 = (double) k / eps;
                Map<String, Double> eo = new HashMap<>();
                for (Map.Entry<String, Integer> e : nsK.get(k - 1).entrySet()) {
                    eo.put(e.getKey(), osdpOut(e.getValue(), sc1, expDet(sd, "ol|" + k + "|" + e.getKey(), sc1)));
                }
                double[] a = hm(truth, eo, ev.domain, 0.0, mass);
                double sc2 = 2.0 * k / eps;
                Map<String, Double> el = new HashMap<>();
                for (String bin : truth.keySet()) el.put(bin, allK.get(k - 1).getOrDefault(bin, 0) + lapDet(sd, "lm|" + k + "|" + bin, sc2));
                double[] b = hm(truth, el, ev.domain, sc2, mass);
                for (int j = 0; j < 6; j++) { ol[k - 1][j][t] = a[j]; lm[k - 1][j][t] = b[j]; }
            }
            // OSDP-RR: one coin per row
            RecordReleaseTracker tracker = new RecordReleaseTracker(DPMetrics.mix(seed, t, Double.doubleToLongBits(eps)));
            List<Row> rel = new ArrayList<>();
            for (Row r : all) if (tracker.release(r.id, r.sensitive, eps).released()) rel.add(r);
            relShare += all.isEmpty() ? 0 : rel.size() / (double) all.size() / trials;
            Map<String, Integer> sample = hist(rel, bm, 0);
            Map<String, Double> raw = new HashMap<>(), scaled = new HashMap<>();
            for (Map.Entry<String, Integer> e : sample.entrySet()) { raw.put(e.getKey(), (double) e.getValue()); scaled.put(e.getKey(), e.getValue() / p); }
            double[] a = hm(truth, raw, ev.domain, 0.0, mass), b = hm(truth, scaled, ev.domain, 0.0, mass);
            for (int j = 0; j < 6; j++) { rr[j][t] = a[j]; rs[j][t] = b[j]; }
        }
        ev.releasedShare = relShare;

        int best1 = 1, best2 = 1;
        double v1 = Double.POSITIVE_INFINITY, v2 = Double.POSITIVE_INFINITY;
        for (int k = 1; k <= K; k++) {
            double a = tm(ol[k - 1][0]), b = tm(lm[k - 1][0]);
            if (a < v1) { v1 = a; best1 = k; }
            if (b < v2) { v2 = b; best2 = k; }
        }
        ev.kStar1 = best1;
        ev.kStar2 = best2;

        double[] allNs = hm(truth, toDouble(hNs), ev.domain, 0.0, mass);
        ev.m[0] = allNs;
        double[][] arms = {null, avg(ol[0]), avg(ol[best1 - 1]), avg(rr), avg(rs), avg(lm[0]), avg(lm[best2 - 1])};
        for (int k = 1; k < 7; k++) ev.m[k] = arms[k];
        double[][] pt = {null, ol[0][1], ol[best1 - 1][1], rr[1], rs[1], lm[0][1], lm[best2 - 1][1]};
        double[] c = new double[trials];
        Arrays.fill(c, allNs[1]);
        ev.perTrial[0] = c;
        for (int k = 1; k < 7; k++) ev.perTrial[k] = pt[k];
        return ev;
    }

    private static double[] avg(final double[][] metricByTrial) {
        double[] r = new double[6];
        for (int j = 0; j < 6; j++) r[j] = tm(metricByTrial[j]);
        return r;
    }

    private static Map<String, Double> toDouble(final Map<String, Integer> h) {
        Map<String, Double> d = new HashMap<>();
        for (Map.Entry<String, Integer> e : h.entrySet()) d.put(e.getKey(), (double) e.getValue());
        return d;
    }

    // =====================================================================================
    //  Operator-level counting queries
    // =====================================================================================
    /** {all-NS, OSDP count, RR x 1/p, DP-post LM} RelError against the operator's true row count, per operator, per trial. */
    private static double[][][] operatorCounts(final Snap s, final double eps, final int trials, final long seed) {
        final int nOps = s.ops.size();
        final double p = RecordReleaseTracker.releaseProbability(eps);
        double[][][] out = new double[nOps][4][trials];
        for (int t = 0; t < trials; t++) {
            final long sd = DPMetrics.mix(seed, t, 51);
            RecordReleaseTracker tracker = new RecordReleaseTracker(DPMetrics.mix(seed, t, Double.doubleToLongBits(eps)));
            for (int oi = 0; oi < nOps; oi++) {
                int n = 0, nsn = 0, rel = 0;
                for (Row r : s.ops.get(oi).rows.values()) {
                    n++;
                    boolean released = tracker.release(r.id, r.sensitive, eps).released();     // ONE decision per row, reused by every operator
                    if (!r.sensitive) nsn++;
                    if (released) rel++;
                }
                double osdp = osdpOut(nsn, 1.0 / eps, expDet(sd, "cnt|" + oi, 1.0 / eps));
                double lmv = n + lapDet(sd, "cntlm|" + oi, 2.0 / eps);
                out[oi][0][t] = DPMetrics.relError(n, nsn);
                out[oi][1][t] = DPMetrics.relError(n, osdp);
                out[oi][2][t] = DPMetrics.relError(n, rel / p);
                out[oi][3][t] = DPMetrics.relError(n, lmv);
            }
        }
        return out;
    }

    // =====================================================================================
    //  Printing helpers
    // =====================================================================================
    private static String f4(final double v) { return Double.isNaN(v) ? "n/a" : String.format("%.4f", v); }

    private static String clip(final String s, final int w) { return s.length() <= w ? s : s.substring(0, w - 3) + "..."; }

    private static void why(final double eps) {
        double p = RecordReleaseTracker.releaseProbability(eps);
        System.out.println("WHY HISTOGRAMS ARE COMPUTED OUTSIDE PATHDB, AND HOW THEY ARE PRIVATISED");
        System.out.println("  1. PathDB returns paths (rows); it has no GROUP BY / COUNT. The histogram is computed on the retrieved rows:");
        System.out.println("         H(b) = #{ rows r : b in bins(r) }     (a row counts once per bin; counting query = one bin; n-gram = up to L-n+1 bins per row)");
        System.out.println("     DP can be put on the histogram (noise) or on the rows (random sampling); binning released rows afterwards is free post-processing.");
        System.out.println("  2. A row is SENSITIVE iff pattern.isSensitivePath(path). H_NS is the histogram of the non-sensitive rows only.");
        System.out.printf ("  3. Mechanisms (eps = %.3f; k = bins a row may add, truncation keeps the first k):%n", eps);
        System.out.println("       DP-post LM T_k  : H_k(b) + Lap(0, 2k/eps) on EVERY bin of the domain          (all rows protected, L1 sensitivity 2k)");
        System.out.println("       OSDP-Laplace    : y = H_NS,k(b) - Exp(k/eps);   out = y + (k/eps) ln 2 if y > 0, else 0   (only sensitive rows protected)");
        System.out.println("           why one direction is enough: replacing a sensitive row by ANY row can only add at most k to H_NS, never remove;");
        System.out.println("           variance (k/eps)^2 vs 2(2k/eps)^2 = 8 (k/eps)^2  ->  8 times smaller, and an EMPTY bin stays exactly 0 (y <= 0 is clamped).");
        System.out.printf ("       OSDP-RR         : each non-sensitive row released with prob. p = 1 - e^-eps = %.4f; histogram of the released rows, raw or divided by p%n", p);
        System.out.println("       all-NS exact    : H_NS without noise: NOT private (a missing row reveals it was sensitive); the floor of every one-sided method.");
        System.out.println("  4. Errors against the FULL ground truth H (sensitive rows included):");
        System.out.println("       RelError(b) = |H(b) - Hhat(b)| / max(1, H(b))                  (DP4SQL; OSDP's relative error with delta = 1)");
        System.out.println("       MRE(domain) = [ sum_active RelError + (d - B) * E|noise on an empty bin| ] / d,   E|noise| = 2k/eps (LM) or 0 (one-sided arms)");
        System.out.println("       L1/mass     = sum_b |H - Hhat| / sum_b H           FLOOR = sum_b |H - H_NS| / sum_b H  (the unavoidable loss of not releasing the sensitive rows)");
        System.out.println("       error above the FLOOR is the price of the noise or of the sampling.\n");
    }

    private static void planOverview(final Snap s) {
        System.out.println("PLAN (execution order, children first). Every operator output can be counted; the histogram is taken on the ROOT output:");
        System.out.printf("  %-4s %-30s %-6s %-9s %8s %10s%n", "idx", "operator", "level", "children", "rows", "sensitive");
        for (int i = 0; i < s.ops.size(); i++) {
            Op op = s.ops.get(i);
            long sens = op.rows.values().stream().filter(r -> r.sensitive).count();
            System.out.printf("  %-4d %-30s %-6d %-9s %8d %10d%s%n", i, op.label, op.depth, op.children.isEmpty() ? "-" : op.children.toString(),
                    op.rows.size(), sens, i == s.ops.size() - 1 ? "   <- ROOT (final result)" : "");
        }
        System.out.println();
    }

    // =====================================================================================
    //  compare
    // =====================================================================================
    /** Defaults: truncation search up to k = 10. */
    public static void compare(final String query, final SensitivePatternPolicy pattern, final double[] epsilons,
                               final List<Hist> specs, final int trials, final long seed) throws Exception {
        compare(query, pattern, epsilons, specs, trials, seed, 10);
    }

    public static void compare(final String query, final SensitivePatternPolicy pattern, final double[] epsilons,
                               final List<Hist> specs, final int trials, final long seed, final int maxTrunc) throws Exception {
        compareOn(snapshot(query, pattern), query, epsilons, specs, trials, seed, maxTrunc);
    }

    private static void compareOn(final Snap s, final String query, final double[] epsilons, final List<Hist> specs,
                                  final int trials, final long seed, final int maxTrunc) {
        System.out.println("=== HISTOGRAM STUDY: counting queries and n-histograms, OSDP-style (computed outside PathDB) ===");
        System.out.println("query: " + query + "\n");
        planOverview(s);
        Op root = s.ops.get(s.ops.size() - 1);
        long sens = root.rows.values().stream().filter(r -> r.sensitive).count();
        System.out.printf("ground truth: %d result rows, %d sensitive (%.1f%%), %d non-sensitive%n%n", root.rows.size(), sens,
                root.rows.isEmpty() ? 0.0 : 100.0 * sens / root.rows.size(), root.rows.size() - sens);
        if (root.rows.isEmpty()) { System.out.println("The final result is empty: nothing to compare."); return; }
        if (sens == 0) System.out.println("NOTE: no sensitive row in the final result, so the one-sided arms lose nothing to the policy (FLOOR = 0): only the noise differs.\n");
        if (sens == root.rows.size()) System.out.println("NOTE: every row is sensitive: the one-sided arms can release nothing (FLOOR = 1). Choose a query whose result mixes both kinds.\n");

        double focus = epsilons[0];
        for (double e : epsilons) if (Math.abs(e - 1.0) < Math.abs(focus - 1.0)) focus = e;

        for (double eps : epsilons) {
            System.out.printf("%n################  epsilon = %.3f  (%d trials, p = %.4f)  ################%n%n", eps, trials, RecordReleaseTracker.releaseProbability(eps));
            why(eps);
            for (Hist h : specs) {
                Eval ev = evaluate(s, h, eps, trials, seed, maxTrunc);
                if (ev.B == 0) { System.out.printf("-- %s: no row falls in any bin (check positions / properties).%n%n", h.name()); continue; }
                printEval(s, ev, eps, trials, eps == epsilons[0]);
            }
            if (eps == focus) printOperatorCounts(s, eps, trials, seed);
        }
        System.out.println();
    }

    /** What the histogram is: the bin rule, how many bins exist, and which rows fall in a bin (and which sensitive rows matter). */
    private static void printDefinition(final Snap s, final Eval ev) {
        Op root = s.ops.get(s.ops.size() - 1);
        Hist h = ev.h;
        int in = 0, out = 0, sIn = 0, sOut = 0;
        Map<String, Integer> reasons = new LinkedHashMap<>();
        for (Row r : root.rows.values()) {
            String w = h.why().apply(new View(r, s));
            if (!w.startsWith("NO BIN")) { in++; if (r.sensitive) sIn++; }
            else { out++; if (r.sensitive) sOut++; reasons.merge(w.contains(" | ") ? w.substring(0, w.indexOf(" | ")) : w, 1, Integer::sum); }
        }
        System.out.printf("-- DEFINITION of the histogram \"%s\"%n", h.name());
        System.out.println("   bin rule  : " + h.rule());
        System.out.printf("   bins      : %s; %d of them are non-empty in this result%n", ev.domainSource, ev.B);
        System.out.printf("   rows      : %d result rows = %d fall in a bin + %d fall in NO bin%n", root.rows.size(), in, out);
        for (Map.Entry<String, Integer> e : reasons.entrySet()) System.out.printf("               no bin because %-44s : %d rows%n", e.getKey().replaceFirst("^NO BIN: ", ""), e.getValue());
        System.out.printf("   sensitive : %d sensitive rows in the result: %d fall in a bin (the only ones this histogram loses) + %d fall in no bin (they do not touch it)%n", sIn + sOut, sIn, sOut);
        System.out.println("   => the policy loss of THIS histogram is only the sensitive rows that fall in ITS bins. Changing which nodes are sensitive changes that number, and with it");
        System.out.println("      which arm wins: with 0 sensitive rows in the bins the one-sided arms lose nothing; with a few they lose that share of the bin counts.\n");
    }

    private static void printEval(final Snap s, final Eval ev, final double eps, final int trials, final boolean withDefinition) {
        if (withDefinition) printDefinition(s, ev);
        System.out.printf("-- HISTOGRAM \"%s\": %d rows (%d sensitive), %d non-empty bins out of d = %.3e possible bins, mass = %.0f (%.0f of it sensitive), truncation search k <= %d%n",
                ev.h.name(), ev.rows, ev.sens, ev.B, ev.domain, ev.mass, ev.sensMass, ev.K);
        System.out.printf("   FLOOR (loss of not releasing the sensitive rows) = %.4f%n", ev.floor);
        System.out.printf("   %-28s | %12s %12s %9s %9s %10s %12s | %9s%n", "arm", "MRE(domain)", "MRE(active)", "Rel50", "Rel95", "L1/mass", "total RelErr", "excess");
        for (int a = 0; a < 7; a++) {
            String nm = ARMS[a].replace("T*", "T*(k=" + (a == 2 ? ev.kStar1 : ev.kStar2) + ")");
            double l1 = ev.m[a][4];
            System.out.printf("   %-28s | %12s %12s %9s %9s %10s %12s | %9s%n", nm, f4(ev.m[a][0]), f4(ev.m[a][1]), f4(ev.m[a][2]), f4(ev.m[a][3]),
                    f4(l1), f4(ev.m[a][5]), (a >= 5 || Double.isNaN(l1) || Double.isNaN(ev.floor)) ? "n/a" : String.format("%+.4f", l1 - ev.floor));
        }
        System.out.println("   excess = L1/mass - FLOOR: the part of the error caused by noise or sampling rather than by the policy (one-sided arms only; DP-post also protects");
        System.out.println("   the sensitive rows, so it has no such floor and can even be below it).");
        System.out.printf("   RELERROR (DP4SQL) per trial, mean over the non-empty bins; always-0 reference = %.1f%n", DPMetrics.ALWAYS_ZERO_REL_ERROR);
        System.out.printf("   %-28s | %8s %8s %8s %8s %8s | %s%n", "arm", "min", "Q1", "median", "Q3", "max", "vs always-0");
        for (int a = 0; a < 7; a++) {
            double[] c = Arrays.stream(ev.perTrial[a]).filter(x -> !Double.isNaN(x)).toArray();
            if (c.length == 0) continue;
            DPMetrics.BoxStats b = DPMetrics.boxStats(c);
            System.out.printf("   %-28s | %8.4f %8.4f %8.4f %8.4f %8.4f | %s%n", ARMS[a], b.min(), b.q1(), b.median(), b.q3(), b.max(), DPMetrics.verdictVsZero(b.median()));
        }
        reading(ev, eps);
        System.out.println();
    }

    /** Plain-language comments on one table, generated from the measured numbers. */
    private static void reading(final Eval ev, final double eps) {
        final double p = RecordReleaseTracker.releaseProbability(eps);
        System.out.println("   READING (what is good, what is bad, and why)");
        System.out.printf("   - Policy: %d of the %d result rows are sensitive (%.1f%%), but only %.0f of the %.0f rows that fall in THIS histogram are (%.1f%% of its mass).%n",
                ev.sens, ev.rows, ev.rows == 0 ? 0.0 : 100.0 * ev.sens / ev.rows, ev.sensMass, ev.mass, 100 * ev.floor);
        System.out.printf("     No one-sided arm can get below L1/mass = %.4f (the FLOOR); the all-NS row shows exactly that.%n", ev.floor);
        double olEx = ev.m[1][4] - ev.floor, rsEx = ev.m[4][4] - ev.floor, rrEx = ev.m[3][4] - ev.floor, lmEx = ev.m[5][4] - ev.floor;
        System.out.printf("   - OSDP-Laplace T1: L1/mass %.4f = floor + %.4f of noise. One-sided noise has 1/8 of the variance of Laplace(2k/eps) and leaves empty bins at 0, so it is %s.%n",
                ev.m[1][4], olEx, olEx < 0.05 ? "GOOD: almost all of its error is the policy" : "noisy here: the bin counts are small compared with the noise scale k/eps");
        System.out.printf("   - OSDP-RR: it keeps only a share p*NS/N = %.4f of the rows (measured %.4f). Raw L1/mass %.4f (floor + %.4f) is mostly sampling loss; dividing by p fixes the bias on average%n",
                p * (ev.rows - ev.sens) / ev.rows, ev.releasedShare, ev.m[3][4], rrEx);
        System.out.printf("     but adds variance: x 1/p gives L1/mass %.4f (floor %+.4f). RR is the only arm that hands out TRUE rows, so its error is the price of that.%n", ev.m[4][4], rsEx);
        double emptyShare = Math.max(0.0, ev.domain - ev.B) / Math.max(1.0, ev.domain);
        System.out.printf("   - DP-post LM T1: noise Lap(0, 2k/eps) = Lap(0, %.2f) on EVERY bin of the domain (d = %.3e, %.1f%% of it empty); MRE(domain) = %.4f%s%n",
                2.0 / eps, ev.domain, 100 * emptyShare, ev.m[5][0],
                emptyShare > 0.9 ? ", about 2k/eps = " + String.format("%.2f", 2.0 / eps) + ", because the empty bins dominate the domain average."
                                 : " (the domain is small here, so the empty bins do not dominate; with a large domain it approaches 2k/eps = " + String.format("%.2f", 2.0 / eps) + ").");
        double[] lmBox = Arrays.stream(ev.perTrial[5]).filter(x -> !Double.isNaN(x)).toArray();
        if (lmBox.length > 0 && DPMetrics.median(lmBox) >= DPMetrics.ALWAYS_ZERO_REL_ERROR) {
            System.out.printf("     BAD: its median RelError %.3f is >= 1, i.e. no better than answering 0: the counts per bin are too small for the noise.%n", DPMetrics.median(lmBox));
        }
        if (ev.K > 1) {
            System.out.printf("   - Truncation: a row can fall in up to %d bins, but T1 keeps only its first bin, so it misses %.1f%% of the histogram mass before any noise (a bias of the T_k arms, not of RR).%n",
                    ev.K, 100 * ev.trunc1Loss);
            System.out.printf("     A larger k loses less mass but raises the noise scale to k/eps (one-sided) or 2k/eps (LM). T*(k=%d / k=%d) shows the best trade-off but looks at the data, so it is NOT private; a private choice needs a fixed k.%n",
                    ev.kStar1, ev.kStar2);
        }
        int best = 1;
        double[] cand = {Double.NaN, ev.m[1][4], Double.NaN, ev.m[3][4], ev.m[4][4], ev.m[5][4]};
        for (int a : new int[]{1, 3, 4, 5}) if (cand[a] < cand[best] || Double.isNaN(cand[best])) best = a;
        System.out.printf("   - Lowest L1/mass among the PRIVATE arms: %s (%.4f).%n", ARMS[best], cand[best]);
        final double lam = 1.0 / eps;                                   // k = 1 for the T1 arms
        final double occ = Math.max(1.0, ev.mass / Math.max(1, ev.B));  // average rows per occupied bin
        final double perBinSens = ev.sensMass / Math.max(1, ev.B);
        final double breakEven = 1.693 * lam;                           // sensitive rows per occupied bin where one-sided and two-sided Laplace tie
        final double dropped1 = ev.mass - ev.nsMass1;                   // mass lost by the policy AND by keeping only the first bin of each row (T1)
        final double olL1 = dropped1 + 0.307 * lam * ev.nsBins1;        // + the bias of the one-sided noise after the median shift
        final double lmL1 = Math.max(ev.domain, ev.B) * 2.0 * lam + ev.trunc1Loss * ev.mass;   // E|Lap(2k/eps)| on EVERY bin + the T1 truncation loss
        System.out.println("   - WHY one arm beats the other (expected L1 in rows, k = 1):");
        System.out.printf("       OSDP-Laplace ~ mass dropped (sensitive rows%s) + 0.31 k/eps per occupied bin = %.1f + 0.31 * %.2f * %d = %.1f%n",
                ev.K > 1 ? " + bins beyond the first k" : "", dropped1, lam, ev.nsBins1, olL1);
        System.out.printf("       DP-post LM   ~ E|Lap(2k/eps)| = 2k/eps = %.2f on EVERY bin of the domain (d = %.3e)%s = %.1f%n", 2 * lam, ev.domain,
                ev.K > 1 ? " + the T1 truncation loss" : "", lmL1);
        System.out.printf("       check against the table: expected L1/mass OSDP-Laplace %.4f (measured %.4f), DP-post %.4f (measured %.4f)%n",
                olL1 / Math.max(1.0, ev.mass), ev.m[1][4], lmL1 / Math.max(1.0, ev.mass), ev.m[5][4]);
        System.out.printf("       per occupied bin: %.2f sensitive rows on average vs break-even 1.69 k/eps = %.2f  ->  %s%n", perBinSens, breakEven,
                perBinSens < breakEven ? "OSDP-Laplace should win on the occupied bins" : "DP-post should win on the occupied bins");
        System.out.printf("       a bin of about %.0f rows: DP noise of scale %.2f costs about %.1f%%, dropping its %.1f sensitive rows costs %.1f%%.%n",
                occ, 2 * lam, 100 * 2 * lam / occ, perBinSens, 100 * perBinSens / occ);
        if (emptyShare > 0.5) {
            System.out.printf("       %.1f%% of the domain is empty: LM also adds noise there and the one-sided arms do not, which is where OSDP gains most.%n", 100 * emptyShare);
        }
        if (ev.m[1][4] < ev.m[5][4]) {
            System.out.printf("   - OSDP-Laplace is closer to the ground truth than DP-post (%.4f vs %.4f): the noise it avoids on empty and small bins outweighs the sensitive rows it drops.%n", ev.m[1][4], ev.m[5][4]);
        } else {
            System.out.printf("   - DP-post is closer to the ground truth than OSDP-Laplace (%.4f vs %.4f). This is the regime where standard DP is hard to beat on error: the counts are large compared with the noise,%n", ev.m[5][4], ev.m[1][4]);
            System.out.println("     so dropping even a few sensitive rows costs more than the noise. The one-sided arms then buy their weaker protection (only sensitive rows) with a systematic bias, not with accuracy.");
        }
        System.out.println("   - The SHARE of sensitive rows is not the criterion; the NUMBER of sensitive rows per bin compared with the noise scale k/eps is. OSDP's reported advantage for policies with more than");
        System.out.println("     about a quarter non-sensitive data is specific to its data (many bins, small counts, large domain); do not apply it as a rule. Expect one-sided arms to win with small eps,");
        System.out.println("     fine-grained or large-domain histograms (n-grams, joint histograms) and few sensitive rows per bin; expect DP-post to win on a few large counts.");
        System.out.println("   - Caution: lower error is paid for with less protection: OSDP arms protect only the sensitive rows (and assume independent records; overlapping paths are an open case),");
        System.out.println("     DP-post protects every row. Compare them together with the sensitive share, not alone.");
    }

    private static void printOperatorCounts(final Snap s, final double eps, final int trials, final long seed) {
        double[][][] oc = operatorCounts(s, eps, trials, seed);
        System.out.printf("OPERATOR-LEVEL COUNTING QUERIES at eps = %.3f: COUNT(*) of each operator's output, RelError |y - yhat| / max(1, y) against that operator's true count%n", eps);
        System.out.printf("  %-3s %-28s %7s %9s %7s | %10s %12s %12s %12s%n", "lvl", "operator", "rows", "sensitive", "non-s.", "all-NS", "OSDP count", "RR x 1/p", "DP-post LM");
        for (int oi = 0; oi < s.ops.size(); oi++) {
            Op op = s.ops.get(oi);
            long sens = op.rows.values().stream().filter(r -> r.sensitive).count();
            System.out.printf("  %-3d %-28s %7d %9d %7d | %10s %12s %12s %12s%n", op.depth, op.label, op.rows.size(), sens, op.rows.size() - sens,
                    f4(tm(oc[oi][0])), f4(tm(oc[oi][1])), f4(tm(oc[oi][2])), f4(tm(oc[oi][3])));
        }
        System.out.println("  all-NS = non-sensitive rows counted exactly (the floor, not private); OSDP count = NS - Exp(1/eps), clamped, + ln2/eps; RR uses ONE coin per row for every operator;");
        System.out.println("  DP-post LM = N + Lap(2/eps). The noise is derived from (seed, operator, bin), so asking the same operator again returns the same value.\n");
    }

    // =====================================================================================
    //  explain: one draw, completely shown
    // =====================================================================================
    public static void explain(final String query, final SensitivePatternPolicy pattern, final double eps, final Hist spec,
                               final int rowsShown, final long seed, final int maxTrunc) throws Exception {
        explainOn(snapshot(query, pattern), query, eps, spec, rowsShown, seed, maxTrunc);
    }

    private static void explainOn(final Snap s, final String query, final double eps, final Hist spec,
                                  final int rowsShown, final long seed, final int maxTrunc) {
        final int nOps = s.ops.size(), last = nOps - 1;
        final double p = RecordReleaseTracker.releaseProbability(eps);
        final long sd = DPMetrics.mix(seed, 0, 41), sdc = DPMetrics.mix(seed, 0, 51);
        System.out.printf("=== EXPLAIN histogram \"%s\" (eps = %.4f) ===%n", spec.name(), eps);
        System.out.println("query: " + query + "\n");
        why(eps);
        planOverview(s);

        System.out.println("1. OPERATOR BY OPERATOR: the counting query COUNT(*) at each release point (one draw)");
        RecordReleaseTracker tracker = new RecordReleaseTracker(DPMetrics.mix(seed, 0, Double.doubleToLongBits(eps)));
        for (int oi = 0; oi < nOps; oi++) {
            Op op = s.ops.get(oi);
            int n = 0, nsn = 0, rel = 0, fresh = 0;
            for (Row r : op.rows.values()) {
                n++;
                RecordReleaseTracker.Outcome o = tracker.release(r.id, r.sensitive, eps);
                if (!r.sensitive) nsn++;
                if (o.released()) rel++;
                if (o.isNew()) fresh++;
            }
            double e = expDet(sdc, "cnt|" + oi, 1.0 / eps);
            double osdp = osdpOut(nsn, 1.0 / eps, e);
            double lap = lapDet(sdc, "cntlm|" + oi, 2.0 / eps);
            System.out.println("\n" + "=".repeat(110));
            System.out.printf("OPERATOR [%d] %s (level %d)%s%n", oi, op.label, op.depth, oi == last ? "   <- ROOT, histogram below" : "");
            System.out.printf("  exact output : N = %d rows = %d SENSITIVE (removed from H_NS) + %d NON-SENSITIVE%n", n, n - nsn, nsn);
            System.out.printf("  all-NS exact : NS = %d    -> RelError |%d - %d| / max(1,%d) = %.4f   (floor, NOT private)%n", nsn, n, nsn, n, DPMetrics.relError(n, nsn));
            System.out.printf("  OSDP count   : y = NS - Exp(1/eps) = %d - %.3f = %.3f;  out = %s = %.3f   -> RelError %.4f%n", nsn, e, nsn - e,
                    nsn - e > 0 ? String.format("y + ln2/eps = %.3f + %.3f", nsn - e, Math.log(2) / eps) : "0 (clamped)", osdp, DPMetrics.relError(n, osdp));
            System.out.printf("  OSDP-RR      : %d of %d non-sensitive rows released (E = p*NS = %.3f); count / p = %d / %.4f = %.2f   -> RelError %.4f   [decisions NEW %d, REUSED %d]%n",
                    rel, nsn, p * nsn, rel, p, rel / p, DPMetrics.relError(n, rel / p), fresh, n - fresh);
            System.out.printf("  DP-post LM   : N + Lap(0, 2/eps) = %d + (%.3f) = %.3f   -> RelError %.4f   (Lap scale 2/eps = %.2f)%n", n, lap, n + lap, DPMetrics.relError(n, n + lap), 2.0 / eps);
        }

        Op root = s.ops.get(last);
        List<Row> all = new ArrayList<>(root.rows.values());
        Map<Row, List<String>> bm = new HashMap<>();
        for (Row r : all) bm.put(r, spec.bins().apply(new View(r, s)));
        System.out.println("\n" + "=".repeat(110));
        System.out.println("2. FINAL QUERY RESULT: WHY each row does or does not fall in a bin");
        Eval pre = evaluate(s, spec, eps, 1, seed, maxTrunc);
        printDefinition(s, pre);
        final int cap = Math.max(rowsShown, 5);
        String[] titles = {"(A) SENSITIVE rows that FALL IN a bin (removed from H_NS: the loss of the one-sided arms)", "(B) SENSITIVE rows that fall in NO bin (they do not affect this histogram)",
                "(C) NON-SENSITIVE rows that FALL IN a bin (counted in H_NS)", "(D) NON-SENSITIVE rows that fall in NO bin"};
        List<List<String>> lines = new ArrayList<>();
        int[] totals = new int[4];
        for (int c = 0; c < 4; c++) lines.add(new ArrayList<>());
        for (Row r : all) {
            String w = spec.why().apply(new View(r, s));
            int c = (r.sensitive ? 0 : 2) + (w.startsWith("NO BIN") ? 1 : 0);
            totals[c]++;
            if (lines.get(c).size() < cap) {
                StringBuilder path = new StringBuilder();
                for (int i = 0; i < r.nodeNames.length; i++) { path.append(r.nodeNames[i]); if (i < r.edgeLabels.length) path.append("-[").append(r.edgeLabels[i]).append("]->"); }
                lines.get(c).add(String.format("      %-96s | %s", clip(w, 96), path));
            }
        }
        for (int c = 0; c < 4; c++) {
            System.out.printf("   %s: %d rows%n", titles[c], totals[c]);
            if (!lines.get(c).isEmpty()) System.out.printf("      %-96s | %s%n", "why (bin | evidence)", "row (path)");
            for (String l : lines.get(c)) System.out.println(l);
            if (totals[c] > cap) System.out.printf("      ... %d more%n", totals[c] - cap);
        }

        List<Row> ns = new ArrayList<>();
        for (Row r : all) if (!r.sensitive) ns.add(r);
        Map<String, Integer> truth = hist(all, bm, 0), hNs = hist(ns, bm, 0);
        double mass = 0;
        for (int v : truth.values()) mass += v;
        Map<Row, List<String>> plan = new HashMap<>();
        for (Op op : s.ops) for (Row r : op.rows.values()) plan.putIfAbsent(r, bm.containsKey(r) ? bm.get(r) : spec.bins().apply(new View(r, s)));
        final double d = derivedDomain(spec, s, plan);
        final int B = truth.size();
        if (B == 0) { System.out.println("\n   No row falls in any bin: nothing to compare (check positions / properties)."); return; }
        final int k = 1;
        final double sc1 = (double) k / eps, sc2 = 2.0 * k / eps;
        List<Row> rel = new ArrayList<>();
        for (Row r : all) if (tracker.release(r.id, r.sensitive, eps).released()) rel.add(r);
        Map<String, Integer> sample = hist(rel, bm, 0);
        Map<String, Integer> nsK = hist(ns, bm, k), allK = hist(all, bm, k);

        List<String> bins = new ArrayList<>(truth.keySet());
        bins.sort((x, y) -> truth.get(x).equals(truth.get(y)) ? x.compareTo(y) : Integer.compare(truth.get(y), truth.get(x)));
        System.out.printf("%n3. THE HISTOGRAM, BIN BY BIN (k = %d; d = %.3e possible bins, B = %d non-empty, mass = %.0f; %d most frequent bins shown)%n", k, d, B, mass, Math.min(15, B));
        System.out.printf("   %-30s %5s %5s | %6s %9s %9s | %5s %7s | %9s %9s%n", "bin", "H", "H_NS", "e", "OSDP-Lap", "RelErr", "RR", "RR/p", "LM out", "RelErr");
        Map<String, Double> eo = new HashMap<>(), erw = new HashMap<>(), ers = new HashMap<>(), el = new HashMap<>();
        double sOl = 0, sRw = 0, sRs = 0, sLm = 0, sNs = 0;
        for (int i = 0; i < B; i++) {
            String bin = bins.get(i);
            int x = truth.get(bin), xn = hNs.getOrDefault(bin, 0), xk = nsK.getOrDefault(bin, 0), r = sample.getOrDefault(bin, 0);
            double e = expDet(sd, "ol|" + k + "|" + bin, sc1);
            double ol = xk > 0 ? osdpOut(xk, sc1, e) : 0.0;
            double lm = allK.getOrDefault(bin, 0) + lapDet(sd, "lm|" + k + "|" + bin, sc2);
            sNs += Math.abs(x - xn) / Math.max(1, x);
            sOl += Math.abs(x - ol) / Math.max(1, x);
            sRw += Math.abs(x - r) / Math.max(1, x);
            sRs += Math.abs(x - r / p) / Math.max(1, x);
            sLm += Math.abs(x - lm) / Math.max(1, x);
            if (i < 15) {
                System.out.printf("   %-30s %5d %5d | %6s %9.2f %9.4f | %5d %7.2f | %9.2f %9.4f%n", clip(bin, 30), x, xn, xk > 0 ? String.format("%.2f", e) : "-", ol,
                        Math.abs(x - ol) / Math.max(1, x), r, r / p, lm, Math.abs(x - lm) / Math.max(1, x));
            }
        }
        System.out.println("   e = the one-sided Exp(k/eps) draw (shown only for bins with H_NS > 0; empty bins get no noise and stay 0); LM adds Lap(0, 2k/eps) to every bin.");
        System.out.println("\n4. METRICS WORKED OUT FOR THIS DRAW");
        System.out.printf("   sum of RelError over the %d non-empty bins: all-NS %.3f | OSDP-Laplace %.3f | RR raw %.3f | RR x1/p %.3f | LM %.3f%n", B, sNs, sOl, sRw, sRs, sLm);
        System.out.printf("   MRE(active) = sum / B   : %.4f | %.4f | %.4f | %.4f | %.4f%n", sNs / B, sOl / B, sRw / B, sRs / B, sLm / B);
        System.out.printf("   MRE(domain) = [sum + (d - B) * E|noise on an empty bin|] / d,  d = %.3e:  all-NS %.3e | OSDP-Laplace %.3e (empty bins stay 0) | RR %.3e / %.3e | LM %.4f (E|noise| = 2k/eps = %.2f)%n",
                d, sNs / d, sOl / d, sRw / d, sRs / d, (sLm + Math.max(0, d - B) * sc2) / d, sc2);
        double nsMass = 0;
        for (int v : hNs.values()) nsMass += v;
        System.out.printf("   FLOOR = (mass - mass_NS) / mass = (%.0f - %.0f) / %.0f = %.4f%n", mass, nsMass, mass, (mass - nsMass) / mass);
        System.out.println();
        Eval ev = evaluate(s, spec, eps, 50, seed, maxTrunc);
        System.out.println("   Summary over 50 trials for the same histogram (what a reader should conclude):");
        printEval(s, ev, eps, 50, false);
    }

    // =====================================================================================
    //  AUTOMATIC SPECS from the RETURN clause (so the specs do not have to be rewritten for every query)
    // =====================================================================================
    /**
     * How RETURN items are mapped and bucketed.
     *  nodeBase : NODE(i) counts from this base (1 = NODE(1) is the first node; set 0 if PathDB counts from 0).
     *  ranges   : property -> {min, max, bucket width}. For a differentially private release these ranges MUST be public;
     *             a numeric property without a range gets one derived from the data, with a warning (fine for evaluation, not a guarantee).
     */
    public static final class SpecOptions {
        public int nodeBase = 1;
        public final Map<String, double[]> ranges = new HashMap<>();
        public final Map<String, Double> domains = new HashMap<>();   // categorical property -> number of possible values (public)
        public int[] ngrams = {2, 3};
        public boolean joint = true;
        public SpecOptions() {
            ranges.put("birthYear", new double[]{1900, 2026, 10});
            ranges.put("length", new double[]{0, 1000, 100});
        }
        public SpecOptions nodeBase(final int b) { this.nodeBase = b; return this; }
        /** Number of possible values of a categorical property (for example the number of persons for "name"). */
        public SpecOptions domain(final String prop, final double possibleValues) { domains.put(prop, possibleValues); return this; }
        public SpecOptions range(final String prop, final double min, final double max, final double width) { ranges.put(prop, new double[]{min, max, width}); return this; }
        public SpecOptions ngrams(final int... n) { this.ngrams = n; return this; }
        public SpecOptions joint(final boolean j) { this.joint = j; return this; }
    }

    private record Item(String text, int pos, String prop) {}

    /** RETURN items of the form var.prop or NODE(i).prop, mapped to a node position. The first and last variables of the MATCH pattern are positions 0 and -1. */
    private static List<Item> parseReturn(final String query, final int nodeBase, final List<String> notes) {
        List<Item> items = new ArrayList<>();
        String up = query.toUpperCase();
        int ri = up.indexOf("RETURN");
        if (ri < 0) { notes.add("no RETURN clause found: no field specs generated"); return items; }
        String match = query.substring(0, ri);
        int wi = match.toUpperCase().indexOf("WHERE");
        if (wi >= 0) match = match.substring(0, wi);
        java.util.regex.Matcher vm = java.util.regex.Pattern.compile("\\(\\s*(\\w+)\\s*\\)").matcher(match);
        String first = null, last = null;
        while (vm.find()) { if (first == null) first = vm.group(1); last = vm.group(1); }
        String ret = query.substring(ri + 6);
        for (String stop : new String[]{";", " LIMIT", " ORDER", " limit", " order"}) {
            int c = ret.indexOf(stop);
            if (c >= 0) ret = ret.substring(0, c);
        }
        for (String tok : ret.split(",")) {
            String t = tok.trim();
            java.util.regex.Matcher nm = java.util.regex.Pattern.compile("(?i)^NODE\\(\\s*(\\d+)\\s*\\)\\.(\\w+)$").matcher(t);
            java.util.regex.Matcher vr = java.util.regex.Pattern.compile("^(\\w+)\\.(\\w+)$").matcher(t);
            if (nm.matches()) {
                items.add(new Item(t, Integer.parseInt(nm.group(1)) - nodeBase, nm.group(2)));
            } else if (vr.matches()) {
                if (vr.group(1).equals(first)) items.add(new Item(t, 0, vr.group(2)));
                else if (vr.group(1).equals(last)) items.add(new Item(t, -1, vr.group(2)));
                else notes.add("'" + t + "': variable '" + vr.group(1) + "' is neither the first (" + first + ") nor the last (" + last + ") node: skipped");
            } else {
                notes.add("'" + t + "' is not var.property or NODE(i).property: skipped");
            }
        }
        return items;
    }

    private static String num(final double v) { return String.format("%.0f", v); }

    private static List<Hist> autoSpecs(final String query, final Snap s, final SpecOptions o) {
        List<String> notes = new ArrayList<>();
        List<Item> items = parseReturn(query, o.nodeBase, notes);
        Op root = s.ops.get(s.ops.size() - 1);
        List<Hist> specs = new ArrayList<>(), single = new ArrayList<>();
        specs.add(count("COUNT(*)", null));
        String ngramProp = null;
        System.out.println("SPECS GENERATED FROM THE RETURN CLAUSE (NODE(i) counted from " + o.nodeBase + "; position 0 = first node, -1 = last node)");
        for (Item it : items) {
            int nonNull = 0, numeric = 0;
            double mn = Double.POSITIVE_INFINITY, mx = Double.NEGATIVE_INFINITY;
            for (Row r : root.rows.values()) {
                View v = new View(r, s);
                int i = v.at(it.pos());
                if (i < 0 || i >= v.length()) continue;
                String raw = v.prop(i, it.prop());
                if (raw == null) continue;
                nonNull++;
                try { double d = Double.parseDouble(raw.trim()); numeric++; mn = Math.min(mn, d); mx = Math.max(mx, d); } catch (NumberFormatException ignore) { /* categorical */ }
            }
            if (nonNull == 0) {
                notes.add(it.text() + ": no result row has '" + it.prop() + "' at position " + it.pos() + " -> skipped (check the position or NODE base)");
                continue;
            }
            if (numeric >= 0.9 * nonNull) {
                double[] rg = o.ranges.get(it.prop());
                if (rg == null) {
                    double width = Math.max(1, Math.ceil((mx - mn) / 10.0));
                    rg = new double[]{Math.floor(mn), Math.ceil(mx), width};
                    notes.add(it.text() + ": numeric, no public range given -> derived from the data [" + num(rg[0]) + ", " + num(rg[1]) + "], width " + num(rg[2])
                            + " (use SpecOptions.range(...) for a strict DP guarantee)");
                }
                Hist h = numeric(it.text() + " (width " + num(rg[2]) + ")", it.prop(), it.pos(), rg[0], rg[1], rg[2]);
                single.add(h);
                specs.add(h);
                final double mid = (rg[0] + rg[1]) / 2;
                final int pos = it.pos();
                final String prop = it.prop();
                specs.add(count("COUNT(" + it.text() + " >= " + num(mid) + ")", r -> {
                    int i = r.at(pos);
                    if (i < 0 || i >= r.length()) return false;
                    String raw = r.prop(i, prop);
                    if (raw == null) return false;
                    try { return Double.parseDouble(raw.trim()) >= mid; } catch (NumberFormatException e) { return false; }
                }));
                System.out.printf("  %-18s -> numeric   (%d of %d rows), range [%s, %s], width %s%n", it.text(), numeric, nonNull, num(rg[0]), num(rg[1]), num(rg[2]));
            } else {
                double dom = o.domains.getOrDefault(it.prop(), 0.0);
                Hist h = category(it.text(), it.prop(), it.pos(), dom);
                single.add(h);
                specs.add(h);
                if (ngramProp == null) ngramProp = it.prop();
                if (dom <= 0) notes.add(it.text() + ": categorical, domain derived from the observed values (declare it with SpecOptions.domain(\"" + it.prop() + "\", n) for a strict DP guarantee)");
                System.out.printf("  %-18s -> category  (%d rows)%n", it.text(), nonNull);
            }
        }
        if (o.joint && single.size() >= 2) {
            for (int i = 0; i < single.size(); i++) {
                for (int j = i + 1; j < single.size(); j++) {
                    specs.add(joint("(" + single.get(i).name() + " , " + single.get(j).name() + ")", single.get(i), single.get(j)));
                }
            }
        }
        if (o.joint && single.size() >= 2 && single.stream().anyMatch(h -> h.domain() <= 0)) {
            notes.add("joint histograms with a categorical item use a domain derived from the observed bins; this understates DP-post's MRE(domain). Declare the domains for strict results.");
        }
        double ngDomain = ngramProp == null ? 0.0 : o.domains.getOrDefault(ngramProp, 0.0);
        for (int n : o.ngrams) specs.add(ngram(n + "-gram of " + (ngramProp == null ? "node ids" : ngramProp), n, ngramProp, ngDomain));
        for (String note : notes) System.out.println("  NOTE: " + note);
        System.out.println("  generated histograms:");
        for (int i = 0; i < specs.size(); i++) System.out.printf("    [%d] %s%n", i, specs.get(i).name());
        System.out.println();
        return specs;
    }

    /** Builds the specs from the query's RETURN clause and compares all of them. The query is executed once. */
    public static void compareAuto(final String query, final SensitivePatternPolicy pattern, final double[] epsilons,
                                   final SpecOptions opts, final int trials, final long seed) throws Exception {
        Snap s = snapshot(query, pattern);
        compareOn(s, query, epsilons, autoSpecs(query, s, opts), trials, seed, 10);
    }

    /** Builds the specs from the RETURN clause and explains ONE of them (specIndex as printed in the list), bin by bin. */
    public static void explainAuto(final String query, final SensitivePatternPolicy pattern, final double eps,
                                   final SpecOptions opts, final int specIndex, final int rowsShown, final long seed) throws Exception {
        Snap s = snapshot(query, pattern);
        List<Hist> specs = autoSpecs(query, s, opts);
        explainOn(s, query, eps, specs.get(Math.min(Math.max(specIndex, 0), specs.size() - 1)), rowsShown, seed, 10);
    }

    // ---- explicit specs for the three example queries (the same thing the automatic mode builds) ----
    /** MATCH ACYCLIC p = (x)-[hasCreator?.knows*]{..6}->(y) ... RETURN x.name, NODE(4).name, y.birthYear;   (NODE(i) 1-based) */
    public static List<Hist> specsQuery1() {
        Hist xName = category("x.name", "name", 0, 0);
        Hist n4Name = category("NODE(4).name", "name", 3, 0);
        Hist yYear = numeric("y.birthYear by decade", "birthYear", -1, 1900, 2026, 10);
        return List.of(
                count("COUNT(*)", null),
                count("COUNT(y.birthYear >= 1963)", r -> { String v = r.prop(r.at(-1), "birthYear"); return v != null && Double.parseDouble(v.trim()) >= 1963; }),
                xName, n4Name, yYear,
                joint("(x.name , y.birthYear)", xName, yYear),
                ngram("2-gram of name", 2, "name", 0),
                ngram("3-gram of name", 3, "name", 0));
    }

    /** MATCH [SIMPLE] p = (x)-[knows+.created]->(msg) RETURN x.birthYear, x.name, NODE(2).name, msg.length;   (queries 2 and 3) */
    public static List<Hist> specsQuery2And3() {
        Hist xYear = numeric("x.birthYear by decade", "birthYear", 0, 1900, 2026, 10);
        Hist xName = category("x.name", "name", 0, 0);
        Hist n2Name = category("NODE(2).name", "name", 1, 0);
        Hist mLen = numeric("msg.length by 100", "length", -1, 0, 1000, 100);
        return List.of(
                count("COUNT(*)", null),
                count("COUNT(msg.length >= 500)", r -> { String v = r.prop(r.at(-1), "length"); return v != null && Double.parseDouble(v.trim()) >= 500; }),
                xYear, xName, n2Name, mLen,
                joint("(x.birthYear , msg.length)", xYear, mLen),
                joint("(x.name , msg.length)", xName, mLen),
                ngram("2-gram of name", 2, "name", 0),
                ngram("3-gram of name", 3, "name", 0));
    }

    // =====================================================================================
    //  Sparsity ladder: where does one-sided noise win?
    // =====================================================================================
    /**
     * Evaluates every histogram in `specs` at one epsilon and lists them by DOMAIN SIZE d, with the number of occupied bins B, the share of
     * empty bins, the sensitive rows that fall in the bins, and the error of OSDP-Laplace T1 against DP-post LM T1 on the whole domain (the
     * paper's MRE) and on the occupied bins only.
     *
     * Why the ladder: DP-post adds Laplace noise to EVERY one of the d bins, so its error grows with d (zero-mean noise, but its absolute error
     * 2k/eps per empty bin adds up; if you clamp negative values to 0 it even becomes positively biased, +b/2 per empty bin). OSDP-Laplace leaves
     * an empty bin at exactly 0 and its noise on occupied bins has 1/8 of the variance. Making bins finer or adding dimensions multiplies d while
     * B stays below the number of rows, so the advantage appears as the ladder climbs. The occupied-bin columns show the catch: when each
     * occupied bin holds about one row, the information is gone for both arms.
     */
    public static void ladder(final String query, final SensitivePatternPolicy pattern, final double eps,
                              final List<Hist> specs, final int trials, final long seed) throws Exception {
        System.out.printf("=== SPARSITY LADDER (eps = %.3f, %d trials): histograms ordered by domain size d ===%n", eps, trials);
        System.out.println("query: " + query + "\n");
        Snap s = snapshot(query, pattern);
        List<Eval> evs = new ArrayList<>();
        for (Hist h : specs) {
            Eval ev = evaluate(s, h, eps, trials, seed, 10);
            if (ev.B > 0) evs.add(ev);
            else System.out.println("skipped (no row falls in any bin): " + h.name());
        }
        evs.sort(Comparator.comparingDouble(e -> e.domain));
        System.out.printf("%-44s %10s %7s %7s %8s %7s | %10s %10s %8s | %10s %10s | %s%n", "histogram", "d", "B", "empty%", "rows/bin", "S_in",
                "OSDP-Lap", "DP-post", "LM/OSDP", "OSDP-Lap", "DP-post", "winner (whole domain)");
        System.out.printf("%-44s %10s %7s %7s %8s %7s | %-30s | %-21s |%n", "", "", "", "", "", "", "MRE(domain)", "MRE(active) occupied bins");
        for (Eval ev : evs) {
            double empty = 100.0 * Math.max(0, ev.domain - ev.B) / Math.max(1.0, ev.domain);
            double ol = ev.m[1][0], lm = ev.m[5][0];
            System.out.printf("%-44s %10s %7d %6.1f%% %8.1f %7.0f | %10.4f %10.4f %8s | %10.4f %10.4f | %s%n", clip(ev.h.name(), 44), sci(ev.domain), ev.B, empty,
                    ev.mass / Math.max(1, ev.B), ev.sensMass, ol, lm, ol > 0 ? String.format("%.1fx", lm / ol) : "-", ev.m[1][1], ev.m[5][1],
                    ol < lm ? "OSDP-Laplace" : "DP-post");
        }
        System.out.println();
        System.out.println("How to read:");
        System.out.println("  d      number of possible bins (the product of the dimensions); B = occupied bins; empty% = (d - B) / d; rows/bin = mass / B; S_in = sensitive rows that fall in the bins.");
        System.out.printf ("  DP-post adds noise of scale 2k/eps = %.2f to all d bins, so MRE(domain) approaches 2k/eps as the domain empties; OSDP-Laplace keeps empty bins at 0.%n", 2.0 / eps);
        System.out.println("  Where OSDP-Laplace wins ONLY in the MRE(domain) column and both arms are poor in the MRE(active) column, the gain is that it does not invent noise on empty bins,");
        System.out.println("  not that it recovers the occupied bins: with about one row per bin the counts themselves cannot be released usefully by either arm.");
        System.out.println("  Plain Laplace is not the only DP option on sparse domains: thresholding / stability-based histograms (an (eps, delta) technique) also avoid noise on empty bins,");
        System.out.println("  so a reviewer may ask for such a baseline before accepting this advantage as specific to one-sided DP.\n");
    }

    private static String sci(final double v) { return v < 1e5 ? String.format("%.0f", v) : String.format("%.2e", v); }

    // =====================================================================================
    //  Sweep of the sensitive share (OSDP's non-sensitive ratio)
    // =====================================================================================
    /**
     * Re-labels sensitivity at EDGE level (an edge is sensitive iff a seeded coin < fraction; a path is sensitive when any edge is),
     * and prints the error of the main arms for one histogram, to show where one-sided release stops paying off.
     */
    public static void sweepSensitiveEdges(final String query, final SensitivePatternPolicy pattern, final double[] fractions,
                                           final double eps, final Hist spec, final int trials, final long seed) throws Exception {
        System.out.printf("=== SWEEP of the sensitive-edge share for \"%s\" (eps = %.3f, %d trials) ===%n", spec.name(), eps, trials);
        System.out.println("query: " + query + "\n");
        Snap s = snapshot(query, pattern);
        Op root = s.ops.get(s.ops.size() - 1);
        System.out.printf("%6s %10s %8s | %-44s | %s%n", "sens%", "rows sens%", "FLOOR", "L1/mass: all-NS / OSDP-Lap / RR x1/p / DP-post LM", "MRE(active): OSDP-Lap / RR x1/p / DP-post LM");
        for (double f : fractions) {
            for (Op op : s.ops) {
                for (Row r : op.rows.values()) {
                    boolean sensitive = false;
                    for (String id : r.edgeIds) {
                        if (RecordReleaseTracker.unit(DPMetrics.mix(seed, 31, 0), id) < f) { sensitive = true; break; }
                    }
                    r.sensitive = sensitive;
                }
            }
            long sr = root.rows.values().stream().filter(r -> r.sensitive).count();
            Eval ev = evaluate(s, spec, eps, trials, seed, 10);
            if (ev.B == 0) continue;
            System.out.printf("%5.0f%% %9.1f%% %8.4f | %9.4f %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f%n", 100 * f,
                    root.rows.isEmpty() ? 0.0 : 100.0 * sr / root.rows.size(), ev.floor,
                    ev.m[0][4], ev.m[1][4], ev.m[4][4], ev.m[5][4], ev.m[1][1], ev.m[4][1], ev.m[5][1]);
        }
        System.out.println();
        System.out.println("Reading: the all-NS column equals the FLOOR (the policy loss). OSDP-Laplace stays close to it while its noise is small, so it wins while the sensitive share");
        System.out.println("is small (fewer than about 1.7 k/eps sensitive rows per bin); DP-post does not depend on the share, so the two curves cross. At 100% sensitive nothing one-sided can be released. The crossover is the number to report.\n");
    }

    // =====================================================================================
    //  Example
    // =====================================================================================
    public static void main(String[] args) throws Exception {
        SensitivePatternPolicy pattern = new SensitivePatternPolicy();
        pattern.addSensitiveNodeName("President");

        // PUBLIC parameters: numeric ranges (bucket width, fine width) and the number of POSSIBLE values of each categorical property.
        // Set "name" to the real number of distinct names in your graph: it defines the domain, i.e. how many zero bins exist.
        Ranges ranges = new Ranges().add("birthYear", 1900, 2026, 10, 1).add("length", 0, 1000, 50, 10).categories("name", 10000);

        String q4 = "MATCH WALK p = (x)-[hasCreator?.knows*]->(y) WHERE p.LENGTH < 4 AND NODE(2).name != \"Karim\" RETURN x.name, NODE(4).name, x.birthYear, y.birthYear;";

        List<Hist> specs = sparseSpecsFor(q4, ranges);     // coarse histograms + fine buckets + all pairs + the full RETURN tuple
        for (Hist h : specs) System.out.println("   - " + h.name());

        Tools.loadDefaultGraph();
        ladder(q4, pattern, 1.0, specs, 50, 42L);          // which histograms make the advantage visible, and at what price
        Tools.loadDefaultGraph();
        ladder(q4, pattern, 0.1, specs, 50, 42L);

        Tools.loadDefaultGraph();
        explain(q4, pattern, 1.0, specs.get(specs.size() - 1), 8, 42L, 10);   // the full tuple, bin by bin
    }
}