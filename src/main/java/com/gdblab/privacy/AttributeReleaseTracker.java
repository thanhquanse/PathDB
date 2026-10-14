package com.gdblab.privacy;

import java.util.HashMap;
import java.util.Map;

/**
 * Tracking mechanism for releasing a SENSITIVE ATTRIBUTE VALUE (for example the birthYear of the President) with
 * Laplace noise, exactly once per data instance.
 *
 * MECHANISM (per node n and property A, with public bounds [min, max], Delta = max - min):
 *     released(n, A) = clamp( round( clamp(v) + Lap(0, Delta / eps) ) )
 * Changing the value v inside the public range moves the output density by at most e^eps, so one release is
 * eps-DP for that attribute value. Rounding and re-clamping are post-processing.
 *
 * WHY TRACKING: the same node appears in the output of several operators and in many result paths. If every
 * retrieval drew fresh noise, (a) the operators would show different values for the same person, and (b) averaging
 * the draws would recover the true value and the cost would compose to k * eps. Here the noise is a deterministic
 * function of (seed, node, property): the first retrieval creates the release, every later retrieval (any operator,
 * any path, the final result) gets the SAME value (REUSED) and costs nothing more. Reusing an already released
 * value is post-processing.
 *
 * Scope: the guarantee is for the ATTRIBUTE VALUE. The structure of the result (which nodes and edges exist) is
 * not changed by this mechanism, and predicates that filter or join on the sensitive attribute must be evaluated
 * on the released value, not on the true one, otherwise the result set itself leaks the true value.
 *
 * One tracker instance = one epsilon.
 */
public final class AttributeReleaseTracker {

    /** value: the released (noisy) value; isNew: first retrieval (NEW) or a reused release (REUSED); noise: the added noise. */
    public record Outcome(double value, boolean isNew, double noise) {}

    /** Composed epsilon per NODE (sum over its attributes): tracked = one release each, untracked = a fresh draw per retrieval. */
    public record LedgerStats(int cells, int nodes, long requests, long reused,
                              double maxNodeTracked, double meanNodeTracked,
                              double maxNodeUntracked, double meanNodeUntracked) {}

    private static final class Entry {
        final String node;
        final double eps;
        final double trueValue;
        final double value;
        final double noise;
        int requests;
        Entry(final String node, final double eps, final double trueValue, final double value, final double noise) {
            this.node = node; this.eps = eps; this.trueValue = trueValue; this.value = value; this.noise = noise;
        }
    }

    private final Map<String, Entry> entries = new HashMap<>();
    private final long seed;
    private long totalRequests = 0;
    private long reusedRequests = 0;

    public AttributeReleaseTracker(final long seed) { this.seed = seed; }

    private static String key(final String node, final String property) { return node + "\u0000" + property; }

    public synchronized Outcome release(final String nodeId, final String property, final double trueValue,
                                        final double min, final double max, final boolean integer, final double eps) {
        totalRequests++;
        String k = key(nodeId, property);
        Entry e = entries.get(k);
        if (e != null) {
            if (Math.abs(e.eps - eps) > 1e-12) {
                throw new IllegalStateException("(" + nodeId + ", " + property + ") was released with eps=" + e.eps
                        + " and is requested again with eps=" + eps + " (use one tracker per epsilon)");
            }
            if (Math.abs(e.trueValue - trueValue) > 1e-9) {
                throw new IllegalStateException("(" + nodeId + ", " + property + ") has two different true values: "
                        + e.trueValue + " and " + trueValue);
            }
            e.requests++;
            reusedRequests++;
            return new Outcome(e.value, false, e.noise);
        }
        double scale = (max - min) / eps;
        double u = RecordReleaseTracker.unit(seed, k) - 0.5;                    // uniform in [-0.5, 0.5)
        double a = Math.min(Math.abs(u), 0.5 - 1e-12);
        double noise = -scale * Math.signum(u) * Math.log(1.0 - 2.0 * a);       // inverse-CDF Laplace(0, scale)
        double v = Math.min(Math.max(trueValue, min), max) + noise;
        if (integer) v = Math.round(v);
        v = Math.min(Math.max(v, min), max);
        Entry created = new Entry(nodeId, eps, trueValue, v, noise);
        created.requests = 1;
        entries.put(k, created);
        return new Outcome(v, true, noise);
    }

    public synchronized boolean isTracked(final String nodeId, final String property) { return entries.containsKey(key(nodeId, property)); }
    public synchronized int releasedCells() { return entries.size(); }
    public synchronized long totalRequests() { return totalRequests; }
    public synchronized long reusedRequests() { return reusedRequests; }

    public synchronized LedgerStats ledger() {
        Map<String, double[]> perNode = new HashMap<>();   // node -> {tracked eps, untracked eps}
        for (Entry e : entries.values()) {
            double[] acc = perNode.computeIfAbsent(e.node, x -> new double[2]);
            acc[0] += e.eps;
            acc[1] += e.requests * e.eps;
        }
        double maxT = 0, sumT = 0, maxU = 0, sumU = 0;
        for (double[] v : perNode.values()) {
            maxT = Math.max(maxT, v[0]); sumT += v[0];
            maxU = Math.max(maxU, v[1]); sumU += v[1];
        }
        int n = perNode.size();
        return new LedgerStats(entries.size(), n, totalRequests, reusedRequests,
                maxT, n == 0 ? 0 : sumT / n, maxU, n == 0 ? 0 : sumU / n);
    }
}