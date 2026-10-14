package com.gdblab.privacy;

import java.util.ArrayList;
import java.util.List;

/**
 * Elastic sensitivity for COUNT queries over chains of edge hops, following
 *   Buil-Aranda, Lobo, Olmedo, "Differential privacy and SPARQL", Semantic Web 15 (2024),
 *   which adapts Johnson, Near, Song, "Towards practical differential privacy for SQL queries" (VLDB 2018),
 *   and smooth sensitivity from Nissim, Raskhodnikova, Smith (STOC 2007).
 *
 * DP UNIT (dp-schema): an "individual" is a star centred on a node (a Person with its outgoing edges and
 * properties). Two graphs are neighbours if one individual's sub-graph is replaced.
 *
 * A path query of n hops (?v0)-[l1]->(?v1)-[l2]->...-[ln]->(?vn) splits into n elementary BGPs
 * B_i = (?v_{i-1}, l_i, ?v_i): each hop has a different centre (its subject), so each is its own elementary BGP.
 *
 * Definitions used (paper, Section 5.2), with B' = B_{i+1} JOIN ... JOIN B_n and x the join variable of B_i and B':
 *   mpv(k)(x, B_n)   = mpv(x, B_n) + k * kappa(S_n)                                  (base case)
 *   mpv(k)(x, B)     = mpv(k)(x1, B') * mpv(k)(x, B_1)                               (inductive case)
 *   S(k)(B_n)        = kappa(S_n)                                                    (base case)
 *   S(k)(B), if the covering star S_i of B_i covers no other B_j in B':
 *                    = max( mpv(k)(x,B_i) * S(k)(B') , mpv(k)(x,B') * S(k)(B_i) )
 *   S(k)(B), if S_i also covers another B_j in B' (self-join):
 *                    = mpv(k)(x,B_i)*S(k)(B') + mpv(k)(x,B')*S(k)(B_i) + S(k)(B_i)*S(k)(B')
 *   ES(k)(COUNT(B)) = S(k)(B)
 *   U(G)            = max_{0<=k<=size(G)} exp(-beta*k) * ES(k)(G),   beta = eps / (2 ln(2/delta))
 *   release         = true count + Lap(2*U/eps)                      -> (eps, delta)-DP   (Theorem 3)
 *
 * For a hop, mpv of its LEFT variable (the subject) is the max out-degree of its label, and mpv of its RIGHT
 * variable (the object) is the max in-degree of its label (the "single triple pattern" bound the paper allows).
 *
 * EXTENSIONS NOT IN THE PAPER (they are sound but are mine): a recursive path pattern such as knows{1,4} is
 * handled as a UNION of fixed-length chains, and ES of a sum of counts is the sum of the ES values
 * (local sensitivity is sub-additive).
 */
public final class ElasticSensitivity {

    private ElasticSensitivity() {}

    /** One hop. star = individual type covering the hop; kappa = star multiplicity used for that hop. */
    public record Hop(String label, String star, double maxOut, double maxIn, double kappa) {}

    /** Result of the smoothing step. */
    public record Smooth(double es0, double u, long argK, double beta, double noiseScale) {}

    /** S(k) of COUNT over one chain of hops. */
    public static double stability(final List<Hop> hops, final long k) {
        final int n = hops.size();
        if (n == 0) return 0.0;
        double[] t = new double[n];   // t[j] = S(k)(B_j JOIN ... JOIN B_n)
        double[] m = new double[n];   // m[j] = mpv(k)(left var of B_j, B_j JOIN ... JOIN B_n)

        Hop last = hops.get(n - 1);
        t[n - 1] = last.kappa();
        m[n - 1] = last.maxOut() + k * last.kappa();

        for (int j = n - 2; j >= 0; j--) {
            Hop h = hops.get(j);
            double a = h.maxIn() + k * h.kappa();            // mpv(k)(x, B_j), x = right var of B_j
            double mRest = m[j + 1];                         // mpv(k)(x, B_{j+1..n})
            double sOne = h.kappa();                         // S(k)(B_j)
            double sRest = t[j + 1];                         // S(k)(B_{j+1..n})

            boolean sameStarLater = false;
            for (int q = j + 1; q < n; q++) {
                if (hops.get(q).star().equals(h.star())) { sameStarLater = true; break; }
            }
            if (sameStarLater) {
                t[j] = a * sRest + mRest * sOne + sOne * sRest;
            } else {
                t[j] = Math.max(a * sRest, mRest * sOne);
            }
            m[j] = mRest * (h.maxOut() + k * h.kappa());
        }
        return t[0];
    }

    /** ES(k) of the COUNT of a UNION of chains (sum of the per-chain values). */
    public static double elastic(final List<List<Hop>> patterns, final long k) {
        double s = 0;
        for (List<Hop> p : patterns) s += stability(p, k);
        return s;
    }

    /** beta = eps / (2 ln(2/delta)); lnTwoOverDelta is passed in so tiny delta values cannot underflow. */
    public static Smooth smooth(final List<List<Hop>> patterns, final long maxK,
                                final double eps, final double lnTwoOverDelta) {
        final double beta = eps / (2.0 * lnTwoOverDelta);
        double best = 0;
        long arg = 0;
        for (long k = 0; k <= maxK; k++) {
            double v = Math.exp(-beta * k) * elastic(patterns, k);
            if (v > best) { best = v; arg = k; }
        }
        return new Smooth(elastic(patterns, 0), best, arg, beta, 2.0 * best / eps);
    }

    /** ln(2/delta) for the paper's choice delta = n^(-ln n):  ln 2 + (ln n)^2. */
    public static double lnTwoOverDeltaPaper(final long n) {
        double ln = Math.log(Math.max(2, n));
        return Math.log(2.0) + ln * ln;
    }

    /** ln(2/delta) for an explicit delta in (0,1). */
    public static double lnTwoOverDelta(final double delta) {
        return Math.log(2.0 / delta);
    }

    // ----------------------------------------------------------------------------------
    //  Self-test:  java ElasticSensitivity.java
    // ----------------------------------------------------------------------------------
    public static void main(String[] args) {
        // With kappa = 1 and one star with uniform degree d, the self-join chain of L hops has
        // ES(k) = (d + k + 1)^L - (d + k)^L  (Johnson et al.'s self-join recursion unrolled).
        int d = 5;
        for (int len = 1; len <= 4; len++) {
            List<Hop> chain = new ArrayList<>();
            for (int i = 0; i < len; i++) chain.add(new Hop("knows", "Person", d, d, 1));
            for (long k : new long[]{0, 3}) {
                double expected = Math.pow(d + k + 1, len) - Math.pow(d + k, len);
                double got = stability(chain, k);
                System.out.printf("L=%d k=%d  got=%.1f expected=%.1f %s%n", len, k, got, expected,
                        Math.abs(got - expected) < 1e-9 ? "OK" : "MISMATCH");
            }
        }
        // Two different stars -> the max() case: a(1)*S(B') vs mpv(B')*kappa
        List<Hop> mixed = List.of(new Hop("a", "S1", 2, 3, 4), new Hop("b", "S2", 5, 6, 7));
        System.out.printf("two-star chain, k=0: %.1f (expected max(3*7, 5*4)=21)%n", stability(mixed, 0));

        // Smoothing on the L=4 chain, n=1000 individuals, eps=1, paper's delta
        List<List<Hop>> pats = new ArrayList<>();
        List<Hop> c4 = new ArrayList<>();
        for (int i = 0; i < 4; i++) c4.add(new Hop("knows", "Person", d, d, 1));
        pats.add(c4);
        Smooth s = smooth(pats, 1000, 1.0, lnTwoOverDeltaPaper(1000));
        System.out.printf("smooth: ES(0)=%.1f  U=%.1f at k*=%d  beta=%.4f  noise scale 2U/eps=%.1f%n",
                s.es0(), s.u(), s.argK(), s.beta(), s.noiseScale());
    }
}