package com.gdblab.privacy;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Metrics used to evaluate a differentially private query pipeline. No dependency on the query engine.
 *
 * UTILITY (error against ground truth)
 *   relErr, trimmedMean, median, mean, std
 *   The trimmed mean follows the protocol used in the R2T line of work (as stated by its follow-up DP-S4S):
 *   100 runs, drop the 20 largest and 20 smallest, average the rest.
 *
 * PRIVACY (empirical audit, after Jagielski-Ullman-Oprea 2020 and Tramer et al. 2022)
 *   audit(...) plays the distinguishing game "was record r in the data?" on two worlds D and D-r:
 *   the attacker picks a threshold rule on one half of the released values, and on the other half we compute
 *   Clopper-Pearson bounds on its true/false positive rates. For an eps-DP mechanism, any event E satisfies
 *   P[E | D] <= e^eps * P[E | D-r], so  eps >= ln( P[E|D] / P[E|D-r] )  and the empirical value is a LOWER bound
 *   on the true eps of the mechanism. If it exceeds the claimed eps, the implementation or the sensitivity
 *   assumption is wrong. The audit can never certify privacy, only detect violations or exact leaks.
 */
public final class DPMetrics {

    private DPMetrics() {}

    // ------------------------------------------------------------------ utility statistics
    public static double mean(final double[] a) {
        double s = 0;
        int n = 0;
        for (double v : a) if (!Double.isNaN(v)) { s += v; n++; }
        return n == 0 ? Double.NaN : s / n;
    }

    public static double std(final double[] a) {
        double m = mean(a);
        double s = 0;
        int n = 0;
        for (double v : a) if (!Double.isNaN(v)) { s += (v - m) * (v - m); n++; }
        return n < 2 ? 0.0 : Math.sqrt(s / (n - 1));
    }

    public static double median(final double[] a) {
        double[] c = a.clone();
        Arrays.sort(c);
        int n = c.length;
        if (n == 0) return Double.NaN;
        return n % 2 == 1 ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    }

    /** p-th percentile (0..1) with linear interpolation; p = 0.95 gives the "Rel95" tail error used by one-sided DP. */
    public static double percentile(final double[] a, final double p) {
        double[] c = a.clone();
        Arrays.sort(c);
        int n = c.length;
        if (n == 0) return Double.NaN;
        double pos = p * (n - 1);
        int lo = (int) Math.floor(pos), hi = (int) Math.ceil(pos);
        return c[lo] + (pos - lo) * (c[hi] - c[lo]);
    }

    /** Mean after dropping floor(frac*n) smallest and floor(frac*n) largest values (frac = 0.2 -> 20 of 100). */
    public static double trimmedMean(final double[] a, final double frac) {
        double[] c = a.clone();
        Arrays.sort(c);
        int n = c.length;
        int k = (int) Math.floor(frac * n);
        if (n - 2 * k <= 0) return median(a);
        double s = 0;
        for (int i = k; i < n - k; i++) s += c[i];
        return s / (n - 2 * k);
    }

    /**
     * RelError of DP4SQL (Cascio et al., 2026, Section 9.2): |y - yHat| / max(1, y), y = true answer, yHat = noisy answer.
     * It equals relErr(y, yHat, 1.0) for y >= 0 (counts, sums of non-negative values).
     */
    public static double relError(final double y, final double yHat) {
        return Math.abs(y - yHat) / Math.max(1.0, y);
    }

    /**
     * RelError of a system that always outputs 0 (the dashed reference line in DP4SQL's boxplots): 1.0 for every y >= 1.
     * An arm whose RelError is >= 1.0 is no better than answering 0.
     */
    public static final double ALWAYS_ZERO_REL_ERROR = 1.0;

    /** Reading of a median RelError against the always-0 reference line. */
    public static String verdictVsZero(final double medianRelError) {
        if (medianRelError < ALWAYS_ZERO_REL_ERROR - 1e-9) return "better than always-0";
        if (medianRelError <= ALWAYS_ZERO_REL_ERROR + 1e-9) return "same as always-0 (nothing useful released)";
        return "WORSE than always-0";
    }

    /** Five-number summary + mean, the numbers behind one boxplot. */
    public record BoxStats(int n, double min, double q1, double median, double q3, double max, double mean) {}

    public static BoxStats boxStats(final double[] a) {
        return new BoxStats(a.length, percentile(a, 0.0), percentile(a, 0.25), percentile(a, 0.5),
                percentile(a, 0.75), percentile(a, 1.0), mean(a));
    }

    /** |est - truth| / max(|truth|, sanity). The sanity bound keeps tiny true values from exploding the ratio. */
    public static double relErr(final double truth, final double est, final double sanity) {
        return Math.abs(est - truth) / Math.max(Math.abs(truth), sanity);
    }

    // ------------------------------------------------------------------ Clopper-Pearson
    private static double[] LF = new double[]{0.0};

    private static synchronized double logFact(final int n) {
        if (n >= LF.length) {
            int old = LF.length;
            LF = Arrays.copyOf(LF, Math.max(n + 1, old * 2));
            for (int i = old; i < LF.length; i++) LF[i] = LF[i - 1] + Math.log(i);
        }
        return LF[n];
    }

    /** P[X <= k] for X ~ Binomial(n, p). */
    static double binomCdf(final int k, final int n, final double p) {
        if (k < 0) return 0.0;
        if (k >= n) return 1.0;
        if (p <= 0.0) return 1.0;
        if (p >= 1.0) return 0.0;
        double lp = Math.log(p);
        double lq = Math.log1p(-p);
        double s = 0;
        for (int i = 0; i <= k; i++) {
            s += Math.exp(logFact(n) - logFact(i) - logFact(n - i) + i * lp + (n - i) * lq);
        }
        return Math.min(1.0, s);
    }

    /** One-sided upper confidence bound for a binomial proportion with k successes in n trials. */
    public static double cpUpper(final int k, final int n, final double alpha) {
        if (k >= n) return 1.0;
        double lo = k / (double) n, hi = 1.0;
        for (int it = 0; it < 80; it++) {
            double mid = 0.5 * (lo + hi);
            if (binomCdf(k, n, mid) > alpha) lo = mid; else hi = mid;
        }
        return hi;
    }

    /** One-sided lower confidence bound for a binomial proportion with k successes in n trials. */
    public static double cpLower(final int k, final int n, final double alpha) {
        if (k <= 0) return 0.0;
        double lo = 0.0, hi = k / (double) n;
        for (int it = 0; it < 80; it++) {
            double mid = 0.5 * (lo + hi);
            if (1.0 - binomCdf(k - 1, n, mid) > alpha) hi = mid; else lo = mid;
        }
        return lo;
    }

    // ------------------------------------------------------------------ empirical privacy audit
    /**
     * epsLower : empirical lower bound on eps (0 if no violation of eps=0 can be shown)
     * p1/p0    : rate of the attacker's event in world D / world D-r on the test half
     * rule     : the attacker's rule, chosen on the other half
     * nTest    : samples per world used for the bounds (the largest measurable eps is about ln(nTest/3.7))
     */
    public record AuditResult(double epsLower, double p1, double p0, int nTest, String rule) {}

    /** Distinguishing game on released scalars from world D (w1) and world D-r (w0). alpha = 0.05 -> 95%. */
    public static AuditResult audit(final double[] w1, final double[] w0, final double alpha) {
        double[] v1 = half(w1, 0), t1 = half(w1, 1);
        double[] v0 = half(w0, 0), t0 = half(w0, 1);

        double[] cand = thresholds(v1, v0, 300);
        final double z = 2.24;     // ~ two-sided 97.5% -> used only to CHOOSE the rule on the validation half
        double bestScore = -1;
        double bestTau = 0;
        boolean bestGe = true;
        boolean bestDir = true;   // true: bound uses P[E|D]/P[E|D-r]; false: the reverse

        for (double tau : cand) {
            for (int o = 0; o < 2; o++) {
                boolean ge = o == 0;
                int c1 = count(v1, tau, ge);
                int c0 = count(v0, tau, ge);
                for (int d = 0; d < 2; d++) {
                    boolean dir = d == 0;
                    int num = dir ? c1 : c0, nNum = dir ? v1.length : v0.length;
                    int den = dir ? c0 : c1, nDen = dir ? v0.length : v1.length;
                    double lo = wilsonLower(num, nNum, z);
                    double hi = wilsonUpper(den, nDen, z);
                    double score = lo > hi ? Math.log(lo / hi) : 0.0;
                    if (score > bestScore) {
                        bestScore = score;
                        bestTau = tau; bestGe = ge; bestDir = dir;
                    }
                }
            }
        }

        int k1 = count(t1, bestTau, bestGe);
        int k0 = count(t0, bestTau, bestGe);
        int kNum = bestDir ? k1 : k0, nNum = bestDir ? t1.length : t0.length;
        int kDen = bestDir ? k0 : k1, nDen = bestDir ? t0.length : t1.length;
        double pL = cpLower(kNum, nNum, alpha / 2);
        double pU = cpUpper(kDen, nDen, alpha / 2);
        double eps = pL > pU ? Math.log(pL / pU) : 0.0;
        String rule = String.format("guess by (x %s %.2f), %s", bestGe ? ">=" : "<=", bestTau,
                bestDir ? "world D" : "world D-r");
        return new AuditResult(eps, k1 / (double) t1.length, k0 / (double) t0.length, Math.min(t1.length, t0.length), rule);
    }

    /** Wilson score bounds: cheap, used only to pick the attacker's rule (the reported bounds are Clopper-Pearson). */
    private static double wilsonLower(final int k, final int n, final double z) {
        if (n == 0) return 0;
        double p = k / (double) n, z2 = z * z;
        return Math.max(0.0, (p + z2 / (2 * n) - z * Math.sqrt(p * (1 - p) / n + z2 / (4.0 * n * n))) / (1 + z2 / n));
    }

    private static double wilsonUpper(final int k, final int n, final double z) {
        if (n == 0) return 1;
        double p = k / (double) n, z2 = z * z;
        return Math.min(1.0, (p + z2 / (2 * n) + z * Math.sqrt(p * (1 - p) / n + z2 / (4.0 * n * n))) / (1 + z2 / n));
    }

    private static double[] half(final double[] a, final int parity) {
        int n = (a.length + (parity == 0 ? 1 : 0)) / 2;
        double[] r = new double[n];
        int j = 0;
        for (int i = parity; i < a.length && j < n; i += 2) r[j++] = a[i];
        return Arrays.copyOf(r, j);
    }

    private static double[] thresholds(final double[] a, final double[] b, final int max) {
        double[] all = new double[a.length + b.length];
        System.arraycopy(a, 0, all, 0, a.length);
        System.arraycopy(b, 0, all, a.length, b.length);
        Arrays.sort(all);
        int u = 0;
        double[] uniq = new double[all.length];
        for (int i = 0; i < all.length; i++) if (i == 0 || all[i] != all[i - 1]) uniq[u++] = all[i];
        if (u <= max) return Arrays.copyOf(uniq, u);
        double[] r = new double[max];
        for (int i = 0; i < max; i++) r[i] = uniq[(int) ((long) i * (u - 1) / (max - 1))];
        return r;
    }

    private static int count(final double[] a, final double tau, final boolean ge) {
        int c = 0;
        for (double v : a) if (ge ? v >= tau : v <= tau) c++;
        return c;
    }

    // ------------------------------------------------------------------ Laplace sampling for audits
    public static double laplace(final SplittableRandom rng, final double scale) {
        double u;
        do { u = rng.nextDouble() - 0.5; } while (u == -0.5);
        return -scale * Math.signum(u) * Math.log(1.0 - 2.0 * Math.abs(u));
    }

    /** splitmix64-style mixing, so consecutive trial numbers give unrelated seeds. */
    public static long mix(final long a, final long b, final long c) {
        long z = a * 0x9E3779B97F4A7C15L + b * 0xBF58476D1CE4E5B9L + c * 0x94D049BB133111EBL;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    // ------------------------------------------------------------------ self-test:  java DpMetrics.java
    public static void main(String[] args) {
        SplittableRandom r = new SplittableRandom(7);
        int n = 6000;
        double[] w1 = new double[n], w0 = new double[n];

        for (int i = 0; i < n; i++) { w1[i] = 1 + laplace(r, 1.0); w0[i] = laplace(r, 1.0); }
        AuditResult ok = audit(w1, w0, 0.05);
        System.out.printf("correct   Laplace(1/eps), eps=1.0 : eps_lower=%.2f  (must be <= 1.0)  [%s]%n", ok.epsLower(), ok.rule());

        for (int i = 0; i < n; i++) { w1[i] = 1 + laplace(r, 0.3); w0[i] = laplace(r, 0.3); }
        AuditResult bad = audit(w1, w0, 0.05);
        System.out.printf("too little noise (scale 0.3, true eps 3.33): eps_lower=%.2f  (> 1.0 -> violation caught)%n", bad.epsLower());

        for (int i = 0; i < n; i++) { w1[i] = 1; w0[i] = 0; }
        AuditResult ex = audit(w1, w0, 0.05);
        System.out.printf("exact release (no noise)          : eps_lower=%.2f  (limited only by the number of trials, cap ~ %.1f)%n",
                ex.epsLower(), Math.log(ex.nTest() / 3.7));

        System.out.printf("Clopper-Pearson check: cpUpper(0,1000,0.025)=%.5f (expect ~0.00369), cpLower(500,1000,0.025)=%.4f (expect ~0.4685)%n",
                cpUpper(0, 1000, 0.025), cpLower(500, 1000, 0.025));
        double[] tm = new double[100];
        for (int i = 0; i < 100; i++) tm[i] = i;
        System.out.printf("trimmedMean(0..99, 0.2) = %.1f (expect 49.5)%n", trimmedMean(tm, 0.2));
        System.out.printf("percentile(0..99, 0.95) = %.2f (expect 94.05)%n", percentile(tm, 0.95));
        System.out.printf("relError(100, 90) = %.2f (expect 0.10); relError(0, 3) = %.2f (expect 3.00); relError(100, 0) = %.2f (expect %.1f)%n",
                relError(100, 90), relError(0, 3), relError(100, 0), ALWAYS_ZERO_REL_ERROR);
        BoxStats bs = boxStats(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        System.out.printf("boxStats(1..9): min=%.0f q1=%.0f median=%.0f q3=%.0f max=%.0f mean=%.0f (expect 1 3 5 7 9 5)%n",
                bs.min(), bs.q1(), bs.median(), bs.q3(), bs.max(), bs.mean());
    }
}