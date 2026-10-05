package com.gdblab.privacy;

import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Guarantees a noise draw happens ONCE per distinct protected target and is reused
 * everywhere that same target is released again - the same principle
 * SelectiveDPRedactor already applies to point-values (cached by objectId+propertyKey),
 * generalized to arbitrary AGGREGATE targets (a sensitive sub-sum, a sensitive
 * sub-count), keyed by a caller-supplied canonical identity string.
 *
 * Why this matters for aggregates specifically: an operator's "sensitive sub-sum" is
 * not a fixed fact the way a node's name is - it can grow as more operators are
 * traversed (recursion finds more sensitive edges, a union adds a branch). So the
 * key must capture WHICH sensitive records actually contribute, not just "this
 * operator" or "this query": if two operators protect the exact same underlying set
 * of sensitive contributors, they get the exact same noisy answer (reused, no new
 * draw); if the set differs even slightly, that's legitimately a different target
 * and gets its own fresh draw. Building the key (e.g. a sorted, joined list of
 * contributing record ids) is the caller's responsibility - see
 * LDBCPatternBenchmark for the canonical-key pattern used there.
 */
public final class NoiseCache {

    private final Map<String, Double> cache = new ConcurrentHashMap<>();
    private final Random rnd;

    public NoiseCache() { this(System.nanoTime()); }
    public NoiseCache(final long seed) { this.rnd = new Random(seed); }

    /** Returns the cached Laplace(0, scale) draw for `key` if one was already made;
     *  otherwise draws a fresh one, caches it, and returns it. The SAME key always
     *  yields the SAME noise value for the lifetime of this cache. */
    public double laplaceNoiseFor(final String key, final double scale) {
        return cache.computeIfAbsent(key, k -> sampleLaplace(scale));
    }

    /** True if `key` already has a cached draw - i.e. this specific target has
     *  already been protected once, somewhere earlier, and will be reused rather
     *  than redrawn. Useful for printing "NEW" vs "REUSED" next to a release. */
    public boolean isCached(final String key) {
        return cache.containsKey(key);
    }

    public int size() { return cache.size(); }

    private double sampleLaplace(final double scale) {
        double u = rnd.nextDouble() - 0.5;
        return -scale * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
    }
}