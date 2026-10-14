package com.gdblab.privacy;

import java.util.HashMap;
import java.util.Map;

/**
 * Tracking mechanism for releasing TRUE records under one-sided differential privacy
 * (OsdpRR, Doudalis et al., VLDB 2017, Algorithm 1).
 *
 * OsdpRR: a sensitive record is never released; a non-sensitive record is released with probability 1 - e^-eps.
 *
 * WHY TRACKING: in an operator-level pipeline the same record (the same path) can be present in the output of
 * several operators. If every operator drew a fresh coin, (a) the record could be released by one operator and
 * suppressed by another, and (b) the privacy cost of the slot it occupies would compose over operators
 * (k operators -> k * eps). This class remembers the first decision for each record id and reuses it, so a record
 * gets exactly ONE randomized decision per run, however many operators emit it, and the cost stays eps.
 *
 * The coin is a deterministic function of (seed, record id), so it is also reproducible; the map additionally
 * counts requests, which feeds the budget ledger (tracked vs untracked epsilon).
 *
 * One tracker instance = one epsilon. Asking for the same record with a different epsilon or a different
 * sensitivity label is a bug and throws.
 */
public final class RecordReleaseTracker {

    /** released: may the record be released; isNew: first time this record was seen (NEW) or a reused decision (REUSED). */
    public record Outcome(boolean released, boolean isNew) {}

    /** Per-record composed epsilon: tracked = one decision per record, untracked = a fresh coin per request. */
    public record LedgerStats(int records, double maxTracked, double meanTracked, double maxUntracked, double meanUntracked) {}

    private static final class Entry {
        final boolean sensitive;
        final boolean released;
        final double eps;
        int requests;
        Entry(final boolean sensitive, final boolean released, final double eps) {
            this.sensitive = sensitive;
            this.released = released;
            this.eps = eps;
        }
    }

    private final Map<String, Entry> entries = new HashMap<>();
    private final long seed;
    private long totalRequests = 0;
    private long reusedRequests = 0;

    public RecordReleaseTracker(final long seed) { this.seed = seed; }

    /** Release probability of a non-sensitive record under OsdpRR: 1 - e^-eps (about 63% at eps = 1). */
    public static double releaseProbability(final double eps) { return 1.0 - Math.exp(-eps); }

    /** The one decision for this record (made on first request, reused afterwards). */
    public synchronized Outcome release(final String recordId, final boolean sensitive, final double eps) {
        totalRequests++;
        Entry e = entries.get(recordId);
        if (e != null) {
            if (e.sensitive != sensitive) {
                throw new IllegalStateException("record " + recordId + " was first seen as "
                        + (e.sensitive ? "sensitive" : "non-sensitive") + " and is now labelled differently");
            }
            if (Math.abs(e.eps - eps) > 1e-12) {
                throw new IllegalStateException("record " + recordId + " was first released with eps=" + e.eps
                        + " and is now requested with eps=" + eps + " (use one tracker per epsilon)");
            }
            e.requests++;
            reusedRequests++;
            return new Outcome(e.released, false);
        }
        boolean released = !sensitive && unit(seed, recordId) < releaseProbability(eps);
        Entry created = new Entry(sensitive, released, eps);
        created.requests = 1;
        entries.put(recordId, created);
        return new Outcome(released, true);
    }

    public synchronized boolean isTracked(final String recordId) { return entries.containsKey(recordId); }
    public synchronized int distinctRecords() { return entries.size(); }
    public synchronized long totalRequests() { return totalRequests; }
    public synchronized long reusedRequests() { return reusedRequests; }

    /** Number of records that were emitted by more than one operator (the ones tracking protects). */
    public synchronized int recordsRequestedMoreThanOnce() {
        int c = 0;
        for (Entry e : entries.values()) if (e.requests > 1) c++;
        return c;
    }

    public synchronized int maxRequestsPerRecord() {
        int m = 0;
        for (Entry e : entries.values()) m = Math.max(m, e.requests);
        return m;
    }

    /**
     * Composed epsilon per record. A sensitive record is never released, but its guarantee is about the
     * neighbouring database where it is replaced by a record that WOULD be released; each independent coin
     * that record gets multiplies the cost. Tracked: eps. Untracked: requests * eps.
     */
    public synchronized LedgerStats ledger(final boolean sensitiveOnly) {
        int n = 0;
        double maxT = 0, sumT = 0, maxU = 0, sumU = 0;
        for (Entry e : entries.values()) {
            if (sensitiveOnly && !e.sensitive) continue;
            double tracked = e.eps;
            double untracked = e.requests * e.eps;
            n++;
            maxT = Math.max(maxT, tracked);
            sumT += tracked;
            maxU = Math.max(maxU, untracked);
            sumU += untracked;
        }
        return new LedgerStats(n, maxT, n == 0 ? 0 : sumT / n, maxU, n == 0 ? 0 : sumU / n);
    }

    // ------------------------------------------------------------------ deterministic coin
    /** 64-bit FNV-1a hash of a string. */
    public static long hash64(final String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x100000001b3L;
        }
        return h;
    }

    /** Uniform value in [0,1) determined by (seed, id). */
    public static double unit(final long seed, final String id) {
        long z = DPMetrics.mix(seed, hash64(id), 0x5DEECE66DL);
        return (z >>> 11) * 0x1.0p-53;
    }
}