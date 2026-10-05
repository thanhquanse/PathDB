package com.gdblab.privacy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.gdblab.privacy.SelectiveDPRedactor.ExponentialMechanism;

/**
 * Utility and privacy evaluation for the selective-DP mechanisms in
 * {@link SelectiveDPRedactor}. Two complementary things are measured:
 *
 * <ul>
 *   <li><b>Utility</b>: how close the privatized output stays to the truth
 *       (match rate / edit distance for categorical point releases, MAE / relative
 *       error for aggregate counts).</li>
 *   <li><b>Privacy</b>: an empirical Monte-Carlo audit that the implementation
 *       actually respects the eps-DP bound it claims - for two "neighboring" true
 *       values A and B (a dataset differing in one sensitive attribute), the ratio
 *       of how often the mechanism produces the same output for A vs B should never
 *       exceed exp(eps), by the definition of DP. Measuring this directly is what
 *       lets you argue the mechanism is correct, not just calibrated on paper.</li>
 * </ul>
 *
 * None of this changes SelectiveDPRedactor's behavior; it only exercises it
 * repeatedly (off to the side, with its own throwaway mechanism instances) to
 * produce numbers you can put in a table or a plot.
 */
public final class DPMetrics {

    private DPMetrics() {}

    // ------------------------------------------------------------------
    // Aggregate (COUNT) utility
    // ------------------------------------------------------------------

    public static final class AggregateUtilityReport {
        public final Map<String, Long> rawCounts;
        public final Map<String, Double> noisyCounts;
        public final double mae;              // mean absolute error across groups
        public final double meanRelativeError; // mean |noisy-raw|/raw across groups (raw>0 only)

        AggregateUtilityReport(Map<String, Long> raw, Map<String, Double> noisy,
                                double mae, double meanRelativeError) {
            this.rawCounts = raw;
            this.noisyCounts = noisy;
            this.mae = mae;
            this.meanRelativeError = meanRelativeError;
        }
    }

    public static AggregateUtilityReport evaluateAggregate(final Map<String, Long> raw,
                                                             final Map<String, Double> noisy) {
        double sumAbsErr = 0;
        double sumRelErr = 0;
        int relCount = 0;
        for (Map.Entry<String, Long> e : raw.entrySet()) {
            double noisyVal = noisy.getOrDefault(e.getKey(), 0.0);
            double absErr = Math.abs(noisyVal - e.getValue());
            sumAbsErr += absErr;
            if (e.getValue() > 0) {
                sumRelErr += absErr / e.getValue();
                relCount++;
            }
        }
        double mae = raw.isEmpty() ? 0 : sumAbsErr / raw.size();
        double mre = relCount == 0 ? 0 : sumRelErr / relCount;
        return new AggregateUtilityReport(raw, noisy, mae, mre);
    }

    /** Runs the noisy-count mechanism `trials` times (fresh randomness each time,
     *  same raw counts) and reports the AVERAGE mae/relative error across trials -
     *  a single trial's error is itself a random variable, this smooths it out. */
    public static AggregateUtilityReport evaluateAggregateOverTrials(
            final List<com.gdblab.graph.schema.Path> paths,
            final java.util.function.Function<com.gdblab.graph.schema.Path, String> groupKeyFn,
            final double epsilon,
            final int trials) {
        Map<String, Long> raw = SelectiveDPRedactor.DPCountAggregator.rawCounts(paths, groupKeyFn);
        double sumMae = 0, sumMre = 0;
        Map<String, Double> lastNoisy = new LinkedHashMap<>();
        for (int t = 0; t < trials; t++) {
            SelectiveDPRedactor.DPCountAggregator agg = new SelectiveDPRedactor.DPCountAggregator();
            SelectiveDPRedactor.PrivacyAccountant scratchAccountant = new SelectiveDPRedactor.PrivacyAccountant();
            Map<String, Double> noisy = agg.noisyCounts(paths, groupKeyFn, epsilon, scratchAccountant);
            AggregateUtilityReport r = evaluateAggregate(raw, noisy);
            sumMae += r.mae;
            sumMre += r.meanRelativeError;
            lastNoisy = noisy;
        }
        return new AggregateUtilityReport(raw, lastNoisy, sumMae / trials, sumMre / trials);
    }

    // ------------------------------------------------------------------
    // Point-release (categorical Exponential Mechanism) utility
    // ------------------------------------------------------------------

    public static final class PointUtilityReport {
        public final double matchRate;          // empirical P(output == trueValue)
        public final double meanEditDistance;    // empirical average edit distance from truth
        public PointUtilityReport(double matchRate, double meanEditDistance) {
            this.matchRate = matchRate;
            this.meanEditDistance = meanEditDistance;
        }
    }

    public static PointUtilityReport evaluatePointRelease(final String trueValue,
                                                            final List<String> domain,
                                                            final double epsilon,
                                                            final int trials) {
        ExponentialMechanism mech = new ExponentialMechanism();
        int matches = 0;
        long totalDist = 0;
        for (int i = 0; i < trials; i++) {
            String out = mech.privatize(trueValue, domain, epsilon);
            if (out.equals(trueValue)) matches++;
            totalDist += levenshtein(trueValue, out);
        }
        return new PointUtilityReport((double) matches / trials, (double) totalDist / trials);
    }

    private static int levenshtein(String a, String b) {
        int[][] dp = new int[a.length() + 1][b.length() + 1];
        for (int i = 0; i <= a.length(); i++) dp[i][0] = i;
        for (int j = 0; j <= b.length(); j++) dp[0][j] = j;
        for (int i = 1; i <= a.length(); i++) {
            for (int j = 1; j <= b.length(); j++) {
                int cost = (a.charAt(i - 1) == b.charAt(j - 1)) ? 0 : 1;
                dp[i][j] = Math.min(Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + cost);
            }
        }
        return dp[a.length()][b.length()];
    }

    // ------------------------------------------------------------------
    // Repeated-trial utility evaluation, following Buil-Aranda, Lobo & Olmedo,
    // "Differential Privacy and SPARQL" (Semantic Web 15, 2024), Section 6.3.
    // Their metric: run the noisy query `trials` times (they use 100) and report
    //   median( (ActualCount - NoisyResult_i) / ActualCount * 100 )  for i=1..trials
    // alongside the true count and the sensitivity/mechanism used, then compare
    // across query "shapes" (their star/path/snowflake) to see how utility degrades
    // with join count and shrinks with result size. This is domain-independent -
    // the formula doesn't reference SPARQL specifics - so it applies unchanged here.
    // ------------------------------------------------------------------

    public static final class RepeatedTrialReport {
        public final long trueCount;
        public final double medianSignedErrorPercent;   // their exact formula, signed
        public final double medianAbsoluteErrorPercent;  // magnitude only - what Fig. 5 actually plots
        public final double meanPrivateResult;
        public final double epsilon;
        public final int trials;
        public final double sensitivity;   // how much one record can change the true count: 1, for a COUNT
        public final double noiseScale;    // Laplace scale = sensitivity / epsilon - the "typical noise size"

        RepeatedTrialReport(long trueCount, double medianSignedErrorPercent,
                             double medianAbsoluteErrorPercent, double meanPrivateResult,
                             double epsilon, int trials, double sensitivity) {
            this.trueCount = trueCount;
            this.medianSignedErrorPercent = medianSignedErrorPercent;
            this.medianAbsoluteErrorPercent = medianAbsoluteErrorPercent;
            this.meanPrivateResult = meanPrivateResult;
            this.epsilon = epsilon;
            this.trials = trials;
            this.sensitivity = sensitivity;
            this.noiseScale = sensitivity / Math.max(epsilon, 1e-9);
        }

        /** Plain-language read on whether this result is usable, purely from the
         *  measured median error - no DP background needed to interpret it. These
         *  cutoffs are a reasonable rule of thumb, not a formal standard: adjust them
         *  if your application has a specific accuracy requirement. */
        public String utilityRating() {
            return DPMetrics.utilityRating(medianAbsoluteErrorPercent);
        }
    }

    /** <10%: differs so little from the truth it's close to publishing raw data - fine
     *  for most uses, but double-check eps is actually strong enough privacy for your
     *  threat model, since very low error often means very weak protection.
     *  10-25%: usable for trends/rankings, not for citing a specific number.
     *  25-50%: only the rough order of magnitude survives.
     *  50-100%: noise is comparable to or bigger than the signal - largely unusable.
     *  >100%: the noisy value is expected to be further from the truth than zero
     *  would be - the release conveys essentially nothing about the true count. */
    public static String utilityRating(final double medianAbsoluteErrorPercent) {
        if (medianAbsoluteErrorPercent < 10) return "EXCELLENT (<10% error)";
        if (medianAbsoluteErrorPercent < 25) return "GOOD (10-25% error)";
        if (medianAbsoluteErrorPercent < 50) return "MODERATE (25-50% error)";
        if (medianAbsoluteErrorPercent < 100) return "POOR (50-100% error)";
        return "UNUSABLE (>100% error - noise exceeds the signal)";
    }

    /**
     * Evaluates a single, ungrouped COUNT(*) query (Buil-Aranda et al.'s plain
     * COUNT_x(B) form - no GROUP BY) over `paths.size()` trials.
     */
    public static RepeatedTrialReport evaluateCountRepeatedTrials(final List<com.gdblab.graph.schema.Path> paths,
                                                                    final double epsilon,
                                                                    final int trials) {
        long trueCount = paths.size();
        double[] signedErrors = new double[trials];
        double sumPrivate = 0;
        SelectiveDPRedactor.DPCountAggregator agg = new SelectiveDPRedactor.DPCountAggregator();

        for (int t = 0; t < trials; t++) {
            SelectiveDPRedactor.PrivacyAccountant scratch = new SelectiveDPRedactor.PrivacyAccountant();
            Map<String, Double> noisy = agg.noisyCounts(paths, p -> "ALL", epsilon, scratch);
            double noisyVal = noisy.getOrDefault("ALL", 0.0);
            sumPrivate += noisyVal;
            signedErrors[t] = trueCount == 0 ? 0.0 : (trueCount - noisyVal) * 100.0 / trueCount;
        }

        double medianSigned = median(signedErrors);
        double[] absErrors = new double[trials];
        for (int t = 0; t < trials; t++) absErrors[t] = Math.abs(signedErrors[t]);
        double medianAbs = median(absErrors);

        return new RepeatedTrialReport(trueCount, medianSigned, medianAbs, sumPrivate / trials, epsilon, trials, /*sensitivity=*/1.0);
    }

    private static double median(final double[] values) {
        double[] sorted = values.clone();
        java.util.Arrays.sort(sorted);
        int n = sorted.length;
        return (n % 2 == 1) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    // ------------------------------------------------------------------
    // Privacy audit: empirically check the eps-DP bound holds between two
    // "neighboring" true values (a dataset differing in one sensitive attribute).
    // ------------------------------------------------------------------

    public static final class PrivacyAuditReport {
        public final double claimedEpsilon;
        public final double observedMaxLogRatio; // max over outcomes of |ln(P_A(o)/P_B(o))|
        public final boolean withinBound;         // observedMaxLogRatio <= claimedEpsilon (+ tolerance)
        PrivacyAuditReport(double claimedEpsilon, double observedMaxLogRatio, boolean withinBound) {
            this.claimedEpsilon = claimedEpsilon;
            this.observedMaxLogRatio = observedMaxLogRatio;
            this.withinBound = withinBound;
        }
    }

    /**
     * Monte-Carlo audit of the Exponential Mechanism: draws `trials` samples for
     * trueValue = valueA and `trials` samples for trueValue = valueB (two neighboring
     * "databases" differing in this one attribute), builds an empirical output
     * distribution for each, and checks that for every observed outcome o,
     * P(mechanism(A) = o) / P(mechanism(B) = o) stays within exp(epsilon), which is
     * exactly the eps-DP guarantee's definition. This is a sanity check on the
     * IMPLEMENTATION, not a substitute for the analytical proof (finite trials only
     * bound this with statistical, not certain, confidence - increase `trials` for a
     * tighter check, and treat a violation as a real bug signal, not noise, once
     * trials is in the thousands+).
     */
    public static PrivacyAuditReport auditExponentialMechanism(final String valueA,
                                                                 final String valueB,
                                                                 final List<String> domain,
                                                                 final double epsilon,
                                                                 final int trials) {
        ExponentialMechanism mech = new ExponentialMechanism();
        Map<String, Integer> countsA = new HashMap<>();
        Map<String, Integer> countsB = new HashMap<>();
        for (String d : domain) { countsA.put(d, 0); countsB.put(d, 0); }

        for (int i = 0; i < trials; i++) {
            countsA.merge(mech.privatize(valueA, domain, epsilon), 1, Integer::sum);
            countsB.merge(mech.privatize(valueB, domain, epsilon), 1, Integer::sum);
        }

        double maxLogRatio = 0;
        for (String outcome : domain) {
            // Laplace-smooth by 0.5 to avoid divide-by-zero / log(0) from rare outcomes
            // at small `trials`; increase trials rather than relying on this to
            // reduce estimation error.
            double pA = (countsA.get(outcome) + 0.5) / (trials + 0.5 * domain.size());
            double pB = (countsB.get(outcome) + 0.5) / (trials + 0.5 * domain.size());
            double logRatio = Math.abs(Math.log(pA / pB));
            maxLogRatio = Math.max(maxLogRatio, logRatio);
        }

        // small statistical slack since this is empirical, not exact
        boolean withinBound = maxLogRatio <= epsilon + 0.5;
        return new PrivacyAuditReport(epsilon, maxLogRatio, withinBound);
    }
}