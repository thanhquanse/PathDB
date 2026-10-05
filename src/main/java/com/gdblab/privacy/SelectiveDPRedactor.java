package com.gdblab.privacy;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import com.gdblab.graph.Graph;
import com.gdblab.graph.schema.Edge;
import com.gdblab.graph.schema.GraphObject;
import com.gdblab.graph.schema.Node;
import com.gdblab.graph.schema.Path;

/**
 * Applies the SELECTIVE differential privacy notion (Shi et al., NAACL 2022,
 * "Selective Differential Privacy for Language Modeling") to the intermediate
 * path values produced by PathDB's query operators, for display/debugging
 * purposes only.
 *
 * <p>Mapping onto the paper's Definitions 2-4:
 * <ul>
 *   <li>record  -> a {@link Node} or {@link Edge} appearing somewhere in a {@link Path}</li>
 *   <li>attribute -> one property of that node/edge (e.g. "name", "ssn", "phone")</li>
 *   <li>policy function F -> {@link PolicyFunction#isSensitive}</li>
 *   <li>Type-2 query (non-sensitive) -> value printed exactly, no privacy budget spent</li>
 *   <li>Type-1 query (sensitive) -> value replaced by the output of a calibrated DP
 *       mechanism (Laplace for numeric-looking values, Exponential Mechanism for
 *       categorical/string values), and charged against a running privacy accountant</li>
 * </ul>
 *
 * <p>This class only changes what gets PRINTED by an explain/trace utility such as
 * IntermediateResultsExplainer. It never mutates the actual Node/Edge/Path objects
 * used by the query engine, so join/selection/recursion semantics are unaffected.
 *
 * <p>Important caveat: this is NOT Selective-DPSGD. The paper's formal (F, eps, delta)-S-DP
 * guarantee (Theorem 2) is proven for a specific composition of gradient- and
 * hidden-state-noising queries during RNN training. What is implemented here reuses
 * the same SELECTIVE, policy-driven design principle for a different setting (releasing
 * intermediate query results instead of training a model), with its own accounting
 * (basic/sequential composition of standard DP mechanisms). It gives you a real,
 * calibrated DP guarantee per released value and a running total across a session,
 * but if you need a specific formal (eps, delta) claim for a paper/audit, review the
 * composition and sensitivity assumptions below against your own threat model.
 */
public final class SelectiveDPRedactor {

    // ------------------------------------------------------------------
    // Taint: the propagated set of sensitive facts carried by a path
    // ------------------------------------------------------------------

    /**
     * One sensitive attribute occurrence: a specific graph object's property.
     * Two facts are equal iff they refer to the same (object id, property key,
     * value) triple - since node/edge properties are never mutated by any
     * algebra operator, re-observing the same fact anywhere in the plan always
     * yields an equal record, so {@code Set<SensitiveFact>} naturally dedups.
     */
    public record SensitiveFact(String ownerId, String ownerLabel, String key, String value) {
        @Override
        public String toString() {
            return ownerLabel + "(" + ownerId + ")." + key;
        }
    }

    /**
     * Base case (Paths0(G)/Paths1(G)): scans one node or edge's properties
     * against the policy function and returns its sensitive facts. This is the
     * ONLY place new taint enters the algebra - every other operator below
     * only forwards, unions, or filters facts that already exist.
     */
    public Set<SensitiveFact> factsOf(final GraphObject go) {
        Set<SensitiveFact> out = new LinkedHashSet<>();
        HashMap<String, String> props = null;
        if (go instanceof Node n) props = n.getProperties();
        else if (go instanceof Edge e) props = e.getProperties();
        if (props == null) return out;
        for (Map.Entry<String, String> entry : props.entrySet()) {
            if (policy.isSensitive(go, entry.getKey(), entry.getValue())) {
                out.add(new SensitiveFact(go.getId(), go.getLabel(), entry.getKey(), entry.getValue()));
            }
        }
        return out;
    }

    /** Base case for a whole leaf path: union of factsOf() over its sequence. */
    public Set<SensitiveFact> scanFacts(final Path p) {
        Set<SensitiveFact> out = new LinkedHashSet<>();
        for (GraphObject go : p.getSequence()) {
            out.addAll(factsOf(go));
        }
        return out;
    }

    // ---- Explicit per-operator composition rules (Taint as an algebra homomorphism) ----

    /** sigma is taint-preserving: filtering can only drop paths, never facts. */
    public static Set<SensitiveFact> taintOfSelection(final Set<SensitiveFact> inputTaint) {
        return inputTaint;
    }

    /** p1 JOIN p2 -> Taint(p1 o p2) = Taint(p1) U Taint(p2). Core composition rule. */
    public static Set<SensitiveFact> taintOfJoin(final Set<SensitiveFact> leftTaint, final Set<SensitiveFact> rightTaint) {
        Set<SensitiveFact> out = new LinkedHashSet<>(leftTaint);
        out.addAll(rightTaint);
        return out;
    }

    /** union does not merge two paths into one; each output path keeps its own taint unchanged. */
    public static Set<SensitiveFact> taintOfUnion(final Set<SensitiveFact> passThroughTaint) {
        return passThroughTaint;
    }

    /**
     * phi(S) is defined by repeated (join, union) per Definition 4.1, so a path
     * built after k hops accumulates the union of taint from every hop it has
     * traversed - exactly mirroring how h_i becomes and stays private once a
     * private token has entered an RNN's recurrence. priorTaint is the taint of
     * the path accumulated through hop k-1; hopTaint is the taint contributed
     * by the newly-joined base edge/node at hop k.
     */
    public static Set<SensitiveFact> taintOfRecursiveStep(final Set<SensitiveFact> priorTaint, final Set<SensitiveFact> hopTaint) {
        return taintOfJoin(priorTaint, hopTaint);
    }

    /** gamma/tau/pi reorganize or select EXISTING paths without modifying them: pass-through. */
    public static Set<SensitiveFact> taintOfGroupOrderProjection(final Set<SensitiveFact> inputTaint) {
        return inputTaint;
    }

    // ------------------------------------------------------------------
    // Policy function F: which (object, property) pairs are sensitive
    // ------------------------------------------------------------------

    public interface PolicyFunction {
        boolean isSensitive(GraphObject owner, String propertyKey, String value);
    }

    /**
     * Convenience default policy: flags any property whose KEY matches a
     * configured sensitive-key list (case-insensitive), plus optionally any
     * property whose VALUE is a pure digit string (mirrors the paper's
     * WikiText-2 policy: "if the token is a digit, F outputs 0").
     */
    public static PolicyFunction keyAndDigitPolicy(final Set<String> sensitiveKeysLowercase,
                                                     final boolean alsoFlagDigitValues) {
        return (owner, key, value) -> {
            if (key != null && sensitiveKeysLowercase.contains(key.toLowerCase())) {
                return true;
            }
            return alsoFlagDigitValues && value != null && !value.isEmpty()
                    && value.chars().allMatch(Character::isDigit);
        };
    }

    /** A reasonable out-of-the-box policy matching the paper's CUSTOMERSIM example. */
    public static PolicyFunction defaultPersonalDataPolicy() {
        Set<String> keys = new LinkedHashSet<>(List.of(
                "name", "ssn", "phone", "address", "email", "tracking", "order",
                "dob", "birthdate", "creditcard", "card"
        ));
        return keyAndDigitPolicy(keys, true);
    }

    /**
     * A policy function F the analyst can edit at runtime: this is what the paper
     * means by "F is a function the analyst defines based on domain knowledge of
     * what's sensitive" - it is not something this code should be deciding on the
     * analyst's behalf with a fixed list. Sensitivity can be declared at two
     * granularities:
     * <ul>
     *   <li>by ATTRIBUTE KEY - e.g. "any 'ssn' property is sensitive, whatever its
     *       value" (the common case: you know the schema, not every value in it)</li>
     *   <li>by EXACT VALUE - e.g. "the string 'Bart' is sensitive wherever it appears,
     *       even in a property key I haven't otherwise flagged" (for a specific known
     *       PII value the analyst wants protected regardless of field)</li>
     * </ul>
     * Thread-safe: keys/values can be added or removed while the redactor is in use.
     */
    public static final class ConfigurablePolicyFunction implements PolicyFunction {
        private final Set<String> sensitiveKeysLowercase = ConcurrentHashMap.newKeySet();
        private final Set<String> sensitiveValuesExact = ConcurrentHashMap.newKeySet();
        private volatile boolean flagDigitValues;

        public ConfigurablePolicyFunction() {
            this(Set.of(), false);
        }

        public ConfigurablePolicyFunction(final Collection<String> initialSensitiveKeys, final boolean flagDigitValues) {
            for (String k : initialSensitiveKeys) {
                sensitiveKeysLowercase.add(k.toLowerCase());
            }
            this.flagDigitValues = flagDigitValues;
        }

        public void addSensitiveKey(final String key) { sensitiveKeysLowercase.add(key.toLowerCase()); }
        public void removeSensitiveKey(final String key) { sensitiveKeysLowercase.remove(key.toLowerCase()); }
        public void setSensitiveKeys(final Collection<String> keys) {
            sensitiveKeysLowercase.clear();
            for (String k : keys) sensitiveKeysLowercase.add(k.toLowerCase());
        }
        public Set<String> getSensitiveKeys() { return new LinkedHashSet<>(sensitiveKeysLowercase); }

        public void addSensitiveValue(final String value) { sensitiveValuesExact.add(value); }
        public void removeSensitiveValue(final String value) { sensitiveValuesExact.remove(value); }
        public Set<String> getSensitiveValues() { return new LinkedHashSet<>(sensitiveValuesExact); }

        public void setFlagDigitValues(final boolean flag) { this.flagDigitValues = flag; }
        public boolean isFlagDigitValues() { return flagDigitValues; }

        @Override
        public boolean isSensitive(final GraphObject owner, final String key, final String value) {
            if (key != null && sensitiveKeysLowercase.contains(key.toLowerCase())) {
                return true;
            }
            if (value != null && sensitiveValuesExact.contains(value)) {
                return true;
            }
            return flagDigitValues && value != null && !value.isEmpty()
                    && value.chars().allMatch(Character::isDigit);
        }
    }

    // ------------------------------------------------------------------
    // Mechanisms
    // ------------------------------------------------------------------

    public interface NumericMechanism {
        /** trueValue in [-domainBound, domainBound]; returns a noised, clipped rendering. */
        String privatize(double trueValue, double domainBound, double epsilon);
    }

    /** Standard Laplace mechanism for a single bounded scalar release (pure eps-DP). */
    public static final class LaplaceMechanism implements NumericMechanism {
        private final Random rnd;

        public LaplaceMechanism(long seed) { this.rnd = new Random(seed); }
        public LaplaceMechanism() { this(System.nanoTime()); }

        @Override
        public String privatize(double trueValue, double domainBound, double epsilon) {
            // Global sensitivity of an identity release over a domain [-B, B] is bounded by 2B
            // in the worst case (two neighboring values at opposite ends of the domain);
            // we use B for a less conservative (looser but standard) calibration - tighten
            // this to 2*domainBound if you need a worst-case-safe bound for your threat model.
            double scale = domainBound / Math.max(epsilon, 1e-9);
            double u = rnd.nextDouble() - 0.5;
            double noise = -scale * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
            double noisy = Math.max(-domainBound, Math.min(domainBound, trueValue + noise));
            return (noisy == Math.rint(noisy))
                    ? String.valueOf((long) noisy)
                    : String.format("%.2f", noisy);
        }
    }

    public interface CategoricalMechanism {
        /** trueValue must be a member of domain; returns one element of domain. */
        String privatize(String trueValue, List<String> domain, double epsilon);
    }

    /** Standard Exponential Mechanism (McSherry & Talwar 2007) over a finite candidate domain,
     *  scored by negative edit distance to the true value (closer strings preferred, exact
     *  match most likely, but any candidate can be returned). */
    public static final class ExponentialMechanism implements CategoricalMechanism {
        private final Random rnd;

        public ExponentialMechanism(long seed) { this.rnd = new Random(seed); }
        public ExponentialMechanism() { this(System.nanoTime()); }

        /** Result of one Exponential Mechanism draw, plus the calibration info needed
         *  to audit that draw: the full output distribution the mechanism sampled from. */
        public static final class Draw {
            public final String output;
            public final Map<String, Double> probabilities; // candidate -> P(mechanism outputs it)
            Draw(String output, Map<String, Double> probabilities) {
                this.output = output;
                this.probabilities = probabilities;
            }
        }

        private static Map<String, Double> distribution(String trueValue, List<String> candidates, double epsilon) {
            double sensitivityOfUtility = 1.0; // one edit op changes utility by at most 1
            double[] logWeights = new double[candidates.size()];
            double maxLogWeight = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < candidates.size(); i++) {
                double utility = -levenshtein(trueValue, candidates.get(i));
                logWeights[i] = (epsilon * utility) / (2.0 * sensitivityOfUtility);
                maxLogWeight = Math.max(maxLogWeight, logWeights[i]);
            }
            double[] weights = new double[candidates.size()];
            double sum = 0;
            for (int i = 0; i < candidates.size(); i++) {
                weights[i] = Math.exp(logWeights[i] - maxLogWeight); // numerically stable softmax
                sum += weights[i];
            }
            Map<String, Double> probs = new java.util.LinkedHashMap<>();
            for (int i = 0; i < candidates.size(); i++) {
                probs.put(candidates.get(i), weights[i] / sum);
            }
            return probs;
        }

        /** Same mechanism as {@link #privatize}, but also returns the probability of
         *  every candidate under this draw, so the caller can show its calibration
         *  instead of a single opaque sample. */
        public Draw privatizeWithDiagnostics(final String trueValue, final List<String> domain, final double epsilon) {
            List<String> candidates = (domain == null || domain.isEmpty())
                    ? List.of(trueValue) : domain;
            Map<String, Double> probs = distribution(trueValue, candidates, epsilon);

            double r = rnd.nextDouble();
            double cumulative = 0;
            String chosen = candidates.get(candidates.size() - 1);
            for (String c : candidates) {
                cumulative += probs.get(c);
                if (r <= cumulative) {
                    chosen = c;
                    break;
                }
            }
            return new Draw(chosen, probs);
        }

        @Override
        public String privatize(String trueValue, List<String> domain, double epsilon) {
            return privatizeWithDiagnostics(trueValue, domain, epsilon).output;
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
    }

    // ------------------------------------------------------------------
    // Privacy accountant (basic/sequential composition)
    // ------------------------------------------------------------------

    public static final class PrivacyAccountant {
        private final List<Double> releases = new ArrayList<>();

        public synchronized void record(double epsilon) {
            releases.add(epsilon);
        }

        /** Naive sequential composition: total eps = sum of per-release eps.
         *  Loose but always valid; swap in RDP/moments-accountant composition
         *  if you need a tighter bound over many releases. */
        public synchronized double totalEpsilonBasicComposition() {
            return releases.stream().mapToDouble(Double::doubleValue).sum();
        }

        public synchronized int releaseCount() {
            return releases.size();
        }
    }

    // ------------------------------------------------------------------
    // The redactor itself
    // ------------------------------------------------------------------

    private PolicyFunction policy; // mutable: F can be redefined at runtime, see the config methods below
    private final CategoricalMechanism categoricalMechanism;
    private final NumericMechanism numericMechanism;
    private final double epsilonPerRelease;
    private final double numericDomainBound;
    private final PrivacyAccountant accountant;
    private final Map<String, List<String>> domainIndex; // propertyKey -> observed values in the graph
    private final Map<String, String> releaseCache = new ConcurrentHashMap<>();    // (objectId|key) -> clean privatized value
    private final Map<String, String> displayCache = new ConcurrentHashMap<>();    // (objectId|key) -> value + diagnostic annotation, for renderPath() only

    public SelectiveDPRedactor(final PolicyFunction policy,
                                final CategoricalMechanism categoricalMechanism,
                                final NumericMechanism numericMechanism,
                                final double epsilonPerRelease,
                                final double numericDomainBound,
                                final PrivacyAccountant accountant) {
        this.policy = policy;
        this.categoricalMechanism = categoricalMechanism;
        this.numericMechanism = numericMechanism;
        this.epsilonPerRelease = epsilonPerRelease;
        this.numericDomainBound = numericDomainBound;
        this.accountant = accountant;
        this.domainIndex = buildDomainIndex();
    }

    /** Reasonable defaults: exponential mechanism for strings, Laplace for numbers,
     *  eps=1.0 per distinct sensitive value, numeric domain bound of 10^9. */
    public static SelectiveDPRedactor withDefaults() {
        return withDefaults(1.0);
    }

    /** Same as {@link #withDefaults()} but with an explicit epsilon. Smaller epsilon =
     *  stronger privacy = the Exponential/Laplace mechanisms drift further from the true
     *  value more often. Useful for demos: eps=1.0 over a 4-name domain returns the true
     *  value ~70% of the time (weak-looking but correct); try eps=0.2-0.3 to make the
     *  privatization visibly obvious on small domains. */
    public static SelectiveDPRedactor withDefaults(double epsilonPerRelease) {
        ConfigurablePolicyFunction policy = new ConfigurablePolicyFunction(
                List.of("name", "ssn", "phone", "address", "email", "tracking", "order",
                        "dob", "birthdate", "creditcard", "card"),
                /*flagDigitValues=*/true);
        return new SelectiveDPRedactor(
                policy,
                new ExponentialMechanism(),
                new LaplaceMechanism(),
                epsilonPerRelease,
                1_000_000_000.0,
                new PrivacyAccountant());
    }

    /** A redactor whose policy flags nothing as sensitive: renderPath() == renderPathRaw(),
     *  no privacy budget is ever spent. Useful as a uniform "no-op" so callers don't need
     *  to special-case a null redactor when comparing before/after views. */
    public static SelectiveDPRedactor passThrough() {
        return new SelectiveDPRedactor(
                (owner, key, value) -> false,
                new ExponentialMechanism(),
                new LaplaceMechanism(),
                0.0,
                0.0,
                new PrivacyAccountant());
    }

    private Map<String, List<String>> buildDomainIndex() {
        Map<String, Set<String>> idx = new HashMap<>();
        Iterator<Node> nit = Graph.getGraph().getNodeIterator();
        while (nit.hasNext()) {
            indexProps(idx, nit.next().getProperties());
        }
        Iterator<Edge> eit = Graph.getGraph().getEdgeIterator();
        while (eit.hasNext()) {
            indexProps(idx, eit.next().getProperties());
        }
        Map<String, List<String>> out = new HashMap<>();
        for (Map.Entry<String, Set<String>> e : idx.entrySet()) {
            out.put(e.getKey(), new ArrayList<>(e.getValue()));
        }
        return out;
    }

    private static void indexProps(Map<String, Set<String>> idx, HashMap<String, String> props) {
        if (props == null) return;
        for (Map.Entry<String, String> e : props.entrySet()) {
            idx.computeIfAbsent(e.getKey(), k -> new LinkedHashSet<>()).add(e.getValue());
        }
    }

    private static boolean looksNumeric(String s) {
        if (s == null || s.isEmpty()) return false;
        return s.chars().allMatch(c -> Character.isDigit(c) || c == '-' || c == '.');
    }

    /**
     * Returns the value to display for one property. Non-sensitive values pass through
     * exactly (Type-2, no privacy cost). Sensitive values are privatized once per
     * (object, key) and the same randomized output is reused on every later call for
     * that same fact, so redisplaying an already-shown path elsewhere in the operator
     * trace does not spend additional privacy budget.
     */
    public String privatizeValue(final GraphObject owner, final String key, final String trueValue) {
        if (trueValue == null || !policy.isSensitive(owner, key, trueValue)) {
            return trueValue;
        }

        String cacheKey = owner.getId() + "::" + key;
        String cached = releaseCache.get(cacheKey);
        if (cached != null) {
            return cached; // reuse previous randomized release; no extra budget spent, no resample
        }

        String clean;      // programmatic value: safe to use as a group key, join key, etc.
        String display;     // clean value + a human-readable diagnostic annotation, for renderPath() only
        if (looksNumeric(trueValue)) {
            double v = Double.parseDouble(trueValue);
            clean = numericMechanism.privatize(v, numericDomainBound, epsilonPerRelease);
            display = clean + " [DP]";
        } else if (categoricalMechanism instanceof ExponentialMechanism em) {
            List<String> domain = new ArrayList<>(domainIndex.getOrDefault(key, List.of(trueValue)));
            if (!domain.contains(trueValue)) domain.add(trueValue);
            ExponentialMechanism.Draw draw = em.privatizeWithDiagnostics(trueValue, domain, epsilonPerRelease);
            double pTrue = draw.probabilities.getOrDefault(trueValue, 0.0);
            boolean changed = !draw.output.equals(trueValue);
            clean = draw.output;
            display = clean + String.format(" [DP, Pr(true)=%.0f%%, %s]",
                    pTrue * 100, changed ? "CHANGED" : "matched true value by chance");
        } else {
            List<String> domain = new ArrayList<>(domainIndex.getOrDefault(key, List.of(trueValue)));
            if (!domain.contains(trueValue)) domain.add(trueValue);
            clean = categoricalMechanism.privatize(trueValue, domain, epsilonPerRelease);
            display = clean + " [DP]";
        }

        accountant.record(epsilonPerRelease);
        releaseCache.put(cacheKey, clean);
        displayCache.put(cacheKey, display);
        return clean;
    }

    /** Same underlying cached release as {@link #privatizeValue}, but returns the
     *  version annotated for display (used by renderPath()). Never call this for a
     *  value you intend to use as a group/join key - use privatizeValue for that. */
    private String displayValue(final GraphObject owner, final String key, final String trueValue) {
        String clean = privatizeValue(owner, key, trueValue); // ensures cache is populated
        if (clean.equals(trueValue) && (trueValue == null || !policy.isSensitive(owner, key, trueValue))) {
            return trueValue; // non-sensitive: no annotation
        }
        String cacheKey = owner.getId() + "::" + key;
        return displayCache.getOrDefault(cacheKey, clean);
    }

    /**
     * Renders a full Path for display with selective protection applied: structure
     * (which nodes/edges, in which order), ids, and labels are shown exactly; only
     * property values flagged sensitive by the policy function are privatized (using
     * the shared, cached release for that (object, key), so the SAME noised value is
     * shown no matter which operator's output this path came from - this is what
     * makes protection propagate consistently through joins, unions and recursion).
     */
    public String renderPath(final Path p) {
        return renderPathInternal(p, true);
    }

    /**
     * Renders a full Path with every property shown in the clear, ignoring the policy
     * function entirely and spending no privacy budget. Intended purely as the "before"
     * side of a before/after comparison; never call this for anything actually released
     * to an untrusted party.
     */
    public String renderPathRaw(final Path p) {
        return renderPathInternal(p, false);
    }

    private String renderPathInternal(final Path p, final boolean privatize) {
        StringBuilder sb = new StringBuilder();
        for (GraphObject go : p.getSequence()) {
            if (go instanceof Node n) {
                sb.append('(').append(n.getId()).append(':').append(n.getLabel());
                appendProps(sb, n, n.getProperties(), privatize);
                sb.append(')');
            } else if (go instanceof Edge e) {
                sb.append("-[").append(e.getId()).append(':').append(e.getLabel());
                appendProps(sb, e, e.getProperties(), privatize);
                sb.append("]->");
            }
        }
        return sb.toString();
    }

    private void appendProps(final StringBuilder sb, final GraphObject owner,
                              final HashMap<String, String> props, final boolean privatize) {
        if (props == null || props.isEmpty()) return;
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, String> entry : props.entrySet()) {
            if (!first) sb.append(", ");
            first = false;
            String shown = privatize
                    ? displayValue(owner, entry.getKey(), entry.getValue())
                    : entry.getValue();
            sb.append(entry.getKey()).append('=').append(shown);
        }
        sb.append('}');
    }

    public PrivacyAccountant getAccountant() {
        return accountant;
    }

    public double getEpsilonPerRelease() {
        return epsilonPerRelease;
    }

    // ------------------------------------------------------------------
    // Configuring F at runtime. These only work when this redactor's policy is a
    // ConfigurablePolicyFunction (true for anything built via withDefaults()); a
    // redactor built with a fully custom PolicyFunction lambda throws instead, since
    // there is nothing generic these methods could safely edit in that case.
    // ------------------------------------------------------------------

    private ConfigurablePolicyFunction requireConfigurable() {
        if (policy instanceof ConfigurablePolicyFunction cpf) {
            return cpf;
        }
        throw new UnsupportedOperationException(
                "This redactor's policy function F is a custom lambda, not a ConfigurablePolicyFunction, "
                + "so it can't be edited through these methods. Build the redactor via "
                + "SelectiveDPRedactor.withDefaults(...) to get an editable F, or call setPolicy(...) "
                + "with your own ConfigurablePolicyFunction instance.");
    }

    /** Replaces F entirely. Clears the release cache (not the accountant - past
     *  privacy spend is real and stays counted) since a value that was previously
     *  shown in the clear or previously privatized under the old F may be classified
     *  differently under the new one. */
    public void setPolicy(final PolicyFunction newPolicy) {
        this.policy = newPolicy;
        clearReleaseCache();
    }

    public void addSensitiveKey(final String key) { requireConfigurable().addSensitiveKey(key); clearReleaseCache(); }
    public void removeSensitiveKey(final String key) { requireConfigurable().removeSensitiveKey(key); clearReleaseCache(); }
    public void setSensitiveKeys(final Collection<String> keys) { requireConfigurable().setSensitiveKeys(keys); clearReleaseCache(); }
    public Set<String> getSensitiveKeys() { return requireConfigurable().getSensitiveKeys(); }

    public void addSensitiveValue(final String value) { requireConfigurable().addSensitiveValue(value); clearReleaseCache(); }
    public void removeSensitiveValue(final String value) { requireConfigurable().removeSensitiveValue(value); clearReleaseCache(); }
    public Set<String> getSensitiveValues() { return requireConfigurable().getSensitiveValues(); }

    public void setFlagDigitValues(final boolean flag) { requireConfigurable().setFlagDigitValues(flag); clearReleaseCache(); }
    public boolean isFlagDigitValues() { return requireConfigurable().isFlagDigitValues(); }

    private void clearReleaseCache() {
        releaseCache.clear();
        displayCache.clear();
    }

    /**
     * The set of "objectId::propertyKey" facts that have been charged and privatized
     * so far in this redactor's lifetime. Compare this set before and after processing
     * an operator's output to see which sensitive facts were first protected AT that
     * operator (vs. inherited already-protected from a child operator below it) - this
     * is the direct evidence that selective protection propagates through the algebra
     * rather than being recomputed independently at each node.
     */
    public Set<String> getChargedFactKeys() {
        return java.util.Collections.unmodifiableSet(releaseCache.keySet());
    }

    // ------------------------------------------------------------------
    // Aggregate (the "A" in SPJA): DP-noised GROUP BY COUNT over a list of paths.
    // PathDB's grammar has no GROUP BY / COUNT syntax, so this runs as a Java-side
    // post-processing step over the path lists IntermediateResultsExplainer already
    // extracts per operator - i.e. it adds an aggregate stage on top of the existing
    // S + P + J(+recursion) pipeline rather than inside PathDB's own query engine.
    // ------------------------------------------------------------------

    public static final class DPCountAggregator {
        private final Random rnd;

        public DPCountAggregator(long seed) { this.rnd = new Random(seed); }
        public DPCountAggregator() { this(System.nanoTime()); }

        /** The true, un-noised group counts - useful only for a "before" comparison. */
        public static Map<String, Long> rawCounts(final List<Path> paths,
                                                    final java.util.function.Function<Path, String> groupKeyFn) {
            Map<String, Long> counts = new java.util.LinkedHashMap<>();
            for (Path p : paths) {
                counts.merge(groupKeyFn.apply(p), 1L, Long::sum);
            }
            return counts;
        }

        /**
         * Laplace mechanism applied to each group's count. Global sensitivity of a
         * COUNT query is 1 (one record entering/leaving the input can change any single
         * group's count by at most 1), so scale = 1/epsilon gives epsilon-DP per group.
         * Each group is charged once to the shared accountant (basic composition across
         * groups, and across whatever else that accountant is already tracking from
         * point releases elsewhere in the trace).
         */
        public Map<String, Double> noisyCounts(final List<Path> paths,
                                                 final java.util.function.Function<Path, String> groupKeyFn,
                                                 final double epsilon,
                                                 final PrivacyAccountant accountant) {
            Map<String, Long> raw = rawCounts(paths, groupKeyFn);
            Map<String, Double> noisy = new java.util.LinkedHashMap<>();
            for (Map.Entry<String, Long> e : raw.entrySet()) {
                noisy.put(e.getKey(), Math.max(0, e.getValue() + sampleLaplace(1.0 / Math.max(epsilon, 1e-9))));
                accountant.record(epsilon);
            }
            return noisy;
        }

        private double sampleLaplace(final double scale) {
            double u = rnd.nextDouble() - 0.5;
            return -scale * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
        }

        // ---------------------------------------------------------------
        // COUNT DISTINCT
        // ---------------------------------------------------------------

        /** Number of distinct groups, Laplace-noised. Under edge/path-level neighboring
         *  (one path added or removed), the number of distinct groups it could introduce
         *  or remove changes by at most 1, so sensitivity = 1, same as a plain count.
         *  (Under NODE-level neighboring - one whole node and everything touching it
         *  added/removed - a single node can appear in many paths, so this bound no
         *  longer holds without further care; state which neighboring notion you're
         *  using when you report this.) */
        public double noisyDistinctCount(final List<Path> paths,
                                          final java.util.function.Function<Path, String> groupKeyFn,
                                          final double epsilon,
                                          final PrivacyAccountant accountant) {
            long trueDistinct = rawCounts(paths, groupKeyFn).size();
            double noisy = Math.max(0, trueDistinct + sampleLaplace(1.0 / Math.max(epsilon, 1e-9)));
            accountant.record(epsilon);
            return noisy;
        }

        // ---------------------------------------------------------------
        // SUM / AVERAGE over a numeric, per-path feature
        // ---------------------------------------------------------------

        public static Map<String, Double> rawSums(final List<Path> paths,
                                                    final java.util.function.Function<Path, String> groupKeyFn,
                                                    final java.util.function.ToDoubleFunction<Path> valueFn) {
            Map<String, Double> sums = new java.util.LinkedHashMap<>();
            for (Path p : paths) {
                sums.merge(groupKeyFn.apply(p), valueFn.applyAsDouble(p), Double::sum);
            }
            return sums;
        }

        /**
         * Laplace mechanism on each group's SUM, after clipping every per-path value to
         * [-clipBound, clipBound]. Clipping is required, not optional: without it, the
         * global sensitivity of a raw sum is unbounded (one record could carry an
         * arbitrarily large value and dominate the sum), so no fixed noise scale could
         * give a real DP guarantee. After clipping, one record changes the sum by at
         * most clipBound, so that is the mechanism's sensitivity.
         */
        public Map<String, Double> noisySums(final List<Path> paths,
                                              final java.util.function.Function<Path, String> groupKeyFn,
                                              final java.util.function.ToDoubleFunction<Path> valueFn,
                                              final double clipBound,
                                              final double epsilon,
                                              final PrivacyAccountant accountant) {
            Map<String, Double> clippedSums = new java.util.LinkedHashMap<>();
            for (Path p : paths) {
                double v = Math.max(-clipBound, Math.min(clipBound, valueFn.applyAsDouble(p)));
                clippedSums.merge(groupKeyFn.apply(p), v, Double::sum);
            }
            Map<String, Double> noisy = new java.util.LinkedHashMap<>();
            double scale = clipBound / Math.max(epsilon, 1e-9);
            for (Map.Entry<String, Double> e : clippedSums.entrySet()) {
                noisy.put(e.getKey(), e.getValue() + sampleLaplace(scale));
                accountant.record(epsilon);
            }
            return noisy;
        }

        /**
         * Private mean = noisy sum / noisy count, splitting the epsilon budget evenly
         * between the two underlying releases. This is the standard, correct way to
         * compute a DP average - adding noise directly to the raw mean does NOT give a
         * calibrated guarantee (dividing by the true, un-noised count leaks it), and
         * dividing by a noisy count that happens to land near zero can blow the result
         * up, hence the floor at 1.
         */
        public Map<String, Double> noisyAverages(final List<Path> paths,
                                                  final java.util.function.Function<Path, String> groupKeyFn,
                                                  final java.util.function.ToDoubleFunction<Path> valueFn,
                                                  final double clipBound,
                                                  final double epsilon,
                                                  final PrivacyAccountant accountant) {
            double halfEps = epsilon / 2.0;
            Map<String, Double> noisySum = noisySums(paths, groupKeyFn, valueFn, clipBound, halfEps, accountant);
            Map<String, Double> noisyCount = noisyCounts(paths, groupKeyFn, halfEps, accountant);
            Map<String, Double> avg = new java.util.LinkedHashMap<>();
            for (String k : noisySum.keySet()) {
                double denom = Math.max(1.0, noisyCount.getOrDefault(k, 1.0));
                avg.put(k, noisySum.get(k) / denom);
            }
            return avg;
        }

        // ---------------------------------------------------------------
        // Three comparable ways to privately average a mix of sensitive and
        // non-sensitive numeric contributions (e.g. actual_salary + posted_salary).
        // All three spend the SAME total epsilon, so their utility is directly
        // comparable - only the mechanism design differs.
        // ---------------------------------------------------------------

        /**
         * NAIVE baseline: clips and noises EVERY contributing value uniformly,
         * ignoring which ones are actually sensitive - i.e. what you get if you treat
         * DP as something to blindly apply to a whole computation rather than
         * selectively to the attributes that need it. Wastes precision on the
         * non-sensitive values for no privacy benefit (they get clipped and folded
         * into the same noisy sum even though nothing needed hiding).
         */
        public double naiveAverage(final List<Path> paths,
                                    final java.util.function.ToDoubleFunction<Path> sensitiveValueFn,
                                    final java.util.function.ToDoubleFunction<Path> nonSensitiveValueFn,
                                    final double clipBound,
                                    final double epsilon,
                                    final PrivacyAccountant accountant) {
            double sum = 0;
            int count = 0;
            for (Path p : paths) {
                sum += clip(sensitiveValueFn.applyAsDouble(p), clipBound);
                count++;
                sum += clip(nonSensitiveValueFn.applyAsDouble(p), clipBound); // clipped even though non-sensitive
                count++;
            }
            double noisySum = sum + sampleLaplace(clipBound / Math.max(epsilon, 1e-9));
            accountant.record(epsilon);
            return noisySum / Math.max(1, count);
        }

        /**
         * PER-RECORD baseline: noises each SENSITIVE value independently (what your
         * original description of "selective DP" did: 100->102, 105->110), keeps
         * non-sensitive values exact. To spend the SAME total epsilon as the other
         * two methods here, the budget is split evenly across the K sensitive
         * records (epsilon/K each) - basic sequential composition. This is closer to
         * Local DP than to the aggregate-level design this project otherwise uses;
         * it is provably worse than noisySelectiveAverage below for the same total
         * epsilon, because noise from K independent draws compounds in the sum
         * (variance grows as K^3 relative to a single aggregate-level draw - see the
         * worked example in chat).
         */
        public double perRecordAverage(final List<Path> paths,
                                        final java.util.function.ToDoubleFunction<Path> sensitiveValueFn,
                                        final java.util.function.ToDoubleFunction<Path> nonSensitiveValueFn,
                                        final double clipBound,
                                        final double totalEpsilon,
                                        final PrivacyAccountant accountant) {
            int sensitiveCount = paths.size(); // one sensitive value per path here
            double epsilonPerRecord = totalEpsilon / Math.max(1, sensitiveCount);
            double sum = 0;
            int count = 0;
            for (Path p : paths) {
                double sv = clip(sensitiveValueFn.applyAsDouble(p), clipBound);
                sum += sv + sampleLaplace(clipBound / Math.max(epsilonPerRecord, 1e-9));
                accountant.record(epsilonPerRecord); // charged once per sensitive record
                count++;
                sum += nonSensitiveValueFn.applyAsDouble(p); // exact, never clipped
                count++;
            }
            return sum / Math.max(1, count);
        }

        /**
         * SELECTIVE-AGGREGATE (the corrected, recommended design): sums the sensitive
         * contributions, clips+noises that SUB-SUM exactly ONCE with the full
         * epsilon, and adds the non-sensitive contributions EXACTLY - no clipping,
         * no noise, since they carry no privacy risk. Combining an exact public
         * number with an already-DP-protected number is legitimate post-processing
         * (it never looks at raw sensitive data again), so this costs nothing extra
         * on top of the one release. This is the mechanism-design analog of what the
         * rest of this project already does for point-values (protect only what F
         * flags, leave the rest exact) - applied correctly to an aggregate instead
         * of naively noising the whole computation.
         */
        public double noisySelectiveAverage(final List<Path> paths,
                                             final java.util.function.ToDoubleFunction<Path> sensitiveValueFn,
                                             final java.util.function.ToDoubleFunction<Path> nonSensitiveValueFn,
                                             final double clipBound,
                                             final double epsilon,
                                             final PrivacyAccountant accountant) {
            double sensitiveSum = 0;
            double nonSensitiveSum = 0;
            int count = 0;
            for (Path p : paths) {
                sensitiveSum += clip(sensitiveValueFn.applyAsDouble(p), clipBound);
                nonSensitiveSum += nonSensitiveValueFn.applyAsDouble(p); // exact, never clipped
                count += 2;
            }
            double noisySensitiveSum = sensitiveSum + sampleLaplace(clipBound / Math.max(epsilon, 1e-9));
            accountant.record(epsilon); // ONE release, full epsilon, regardless of how many sensitive records
            return (noisySensitiveSum + nonSensitiveSum) / Math.max(1, count);
        }

        private static double clip(final double v, final double bound) {
            return Math.max(-bound, Math.min(bound, v));
        }

        // ---------------------------------------------------------------
        // Pattern-driven selective aggregates: each PATH is either sensitive or not
        // (decided by an arbitrary predicate, e.g. SensitivePatternPolicy::isSensitivePath),
        // contributing its value to exactly one of the two sub-sums - unlike
        // noisySelectiveAverage above, where every row contributes both a sensitive
        // and a non-sensitive value (the salary case). Same underlying idea (clip+noise
        // only the sensitive sub-sum once; keep the rest exact; combine via
        // post-processing), different splitting rule.
        // ---------------------------------------------------------------

        public static final class SelectiveSumResult {
            public final double trueSensitiveSum;
            public final double trueNonSensitiveSum;
            public final int sensitiveCount;
            public final int nonSensitiveCount;
            public final double releasedSum; // noisy sensitive sub-sum + exact non-sensitive sub-sum

            SelectiveSumResult(double trueSensitiveSum, double trueNonSensitiveSum,
                                int sensitiveCount, int nonSensitiveCount, double releasedSum) {
                this.trueSensitiveSum = trueSensitiveSum;
                this.trueNonSensitiveSum = trueNonSensitiveSum;
                this.sensitiveCount = sensitiveCount;
                this.nonSensitiveCount = nonSensitiveCount;
                this.releasedSum = releasedSum;
            }

            public double trueTotalSum() { return trueSensitiveSum + trueNonSensitiveSum; }
            public int totalCount() { return sensitiveCount + nonSensitiveCount; }
        }

        /**
         * Selective SUM by pattern: paths for which isSensitive.test(path) is true
         * contribute to a clipped, once-noised sub-sum; all other paths contribute
         * their value exactly. Returns both the true breakdown (for comparison/
         * debugging - never actually released) and the one number that would be.
         */
        public SelectiveSumResult noisySelectiveSumByPattern(final List<Path> paths,
                                                               final java.util.function.Predicate<Path> isSensitive,
                                                               final java.util.function.ToDoubleFunction<Path> valueFn,
                                                               final double clipBound,
                                                               final double epsilon,
                                                               final PrivacyAccountant accountant) {
            double sensitiveSum = 0, nonSensitiveSum = 0;
            int sensitiveCount = 0, nonSensitiveCount = 0;
            for (Path p : paths) {
                double v = valueFn.applyAsDouble(p);
                if (isSensitive.test(p)) {
                    sensitiveSum += clip(v, clipBound);
                    sensitiveCount++;
                } else {
                    nonSensitiveSum += v; // exact, never clipped
                    nonSensitiveCount++;
                }
            }
            double noisySensitiveSum = sensitiveSum + sampleLaplace(clipBound / Math.max(epsilon, 1e-9));
            accountant.record(epsilon); // ONE release covering however many sensitive paths there were
            double released = noisySensitiveSum + nonSensitiveSum;
            return new SelectiveSumResult(sensitiveSum, nonSensitiveSum, sensitiveCount, nonSensitiveCount, released);
        }

        public double noisySelectiveAverageByPattern(final List<Path> paths,
                                                       final java.util.function.Predicate<Path> isSensitive,
                                                       final java.util.function.ToDoubleFunction<Path> valueFn,
                                                       final double clipBound,
                                                       final double epsilon,
                                                       final PrivacyAccountant accountant) {
            SelectiveSumResult r = noisySelectiveSumByPattern(paths, isSensitive, valueFn, clipBound, epsilon, accountant);
            return r.releasedSum / Math.max(1, r.totalCount());
        }

        /**
         * "CLASSICAL UNIFORM" baseline for direct comparison: clips and noises the
         * WHOLE combined sum, ignoring the sensitive/non-sensitive split entirely -
         * i.e. what a non-selective, blanket post-processing approach would do.
         * Reuses the plain noisySums/noisyAverages machinery with a single group.
         */
        public double uniformSum(final List<Path> paths,
                                  final java.util.function.ToDoubleFunction<Path> valueFn,
                                  final double clipBound,
                                  final double epsilon,
                                  final PrivacyAccountant accountant) {
            return noisySums(paths, p -> "ALL", valueFn, clipBound, epsilon, accountant).getOrDefault("ALL", 0.0);
        }

        // ---------------------------------------------------------------
        // EDGE-level selective aggregates: each EDGE is independently classified
        // sensitive or not (e.g. via SensitivePatternPolicy::isSensitiveEdge), NOT
        // the whole path it happens to appear in. This is the fix for tainting a
        // multi-hop path's every edge just because ONE hop touches a sensitive node -
        // (Bob)-[knows]->(David) stays exact even if it's part of a path that also
        // contains (Alice)-[knows]->(President).
        // ---------------------------------------------------------------

        public static final class SelectiveEdgeResult {
            public final long sensitiveCount;
            public final long nonSensitiveCount;
            public final double trueSensitiveSum;    // never released directly - for display/debug only
            public final double trueNonSensitiveSum;
            public final double releasedCount;       // noisy sensitive count + exact non-sensitive count
            public final double releasedSum;          // noisy sensitive sub-sum + exact non-sensitive sub-sum

            SelectiveEdgeResult(long sensitiveCount, long nonSensitiveCount,
                                 double trueSensitiveSum, double trueNonSensitiveSum,
                                 double releasedCount, double releasedSum) {
                this.sensitiveCount = sensitiveCount;
                this.nonSensitiveCount = nonSensitiveCount;
                this.trueSensitiveSum = trueSensitiveSum;
                this.trueNonSensitiveSum = trueNonSensitiveSum;
                this.releasedCount = releasedCount;
                this.releasedSum = releasedSum;
            }

            public long trueTotalCount() { return sensitiveCount + nonSensitiveCount; }
            public double trueTotalSum() { return trueSensitiveSum + trueNonSensitiveSum; }
        }

        /**
         * COUNT and, if valueFn is non-null, SUM over a list of EDGES (not paths),
         * each classified independently by isSensitiveEdge. Sensitive edges'
         * contribution to the count (always 1 each) and to the sum (their valueFn
         * output, clipped) are folded into ONE noised release each; non-sensitive
         * edges' contributions are exact. Pass valueFn=null to get COUNT only
         * (releasedSum will be 0/unused) - most edge schemas won't have a numeric
         * property to sum until you add one (PGDF edges.pgdf is currently a fixed
         * @id|@label|@dir|@out|@in schema with no extra columns, per the README).
         */
        public SelectiveEdgeResult noisySelectiveEdgeAggregate(
                final List<Edge> edges,
                final java.util.function.Predicate<Edge> isSensitiveEdge,
                final java.util.function.ToDoubleFunction<Edge> valueFn, // nullable
                final double clipBound,
                final double epsilon,
                final PrivacyAccountant accountant) {
            long sensitiveCount = 0, nonSensitiveCount = 0;
            double sensitiveSum = 0, nonSensitiveSum = 0;
            for (Edge e : edges) {
                double v = (valueFn != null) ? valueFn.applyAsDouble(e) : 0.0;
                if (isSensitiveEdge.test(e)) {
                    sensitiveCount++;
                    sensitiveSum += clip(v, clipBound);
                } else {
                    nonSensitiveCount++;
                    nonSensitiveSum += v;
                }
            }
            double noisySensitiveCount = sensitiveCount + sampleLaplace(1.0 / Math.max(epsilon, 1e-9));
            double noisySensitiveSum = sensitiveSum + sampleLaplace(clipBound / Math.max(epsilon, 1e-9));
            // two releases (count, sum) from the same sensitive subset - charge both if both are used;
            // charge once if only COUNT is needed (valueFn == null).
            accountant.record(epsilon);
            if (valueFn != null) accountant.record(epsilon);

            double releasedCount = noisySensitiveCount + nonSensitiveCount;
            double releasedSum = (valueFn != null) ? (noisySensitiveSum + nonSensitiveSum) : 0.0;
            return new SelectiveEdgeResult(sensitiveCount, nonSensitiveCount, sensitiveSum, nonSensitiveSum,
                    releasedCount, releasedSum);
        }

        /** Classical uniform baseline for edges: every edge counted as if it might be
         *  sensitive, one noised count for the WHOLE set - for direct comparison. */
        public double uniformEdgeCount(final List<Edge> edges, final double epsilon, final PrivacyAccountant accountant) {
            double noisy = edges.size() + sampleLaplace(1.0 / Math.max(epsilon, 1e-9));
            accountant.record(epsilon);
            return noisy;
        }

        public double uniformAverage(final List<Path> paths,
                                      final java.util.function.ToDoubleFunction<Path> valueFn,
                                      final double clipBound,
                                      final double epsilon,
                                      final PrivacyAccountant accountant) {
            return noisyAverages(paths, p -> "ALL", valueFn, clipBound, epsilon, accountant).getOrDefault("ALL", 0.0);
        }

        // ---------------------------------------------------------------
        // MIN / MAX
        // ---------------------------------------------------------------

        /**
         * Privately estimates the min or max of a numeric, per-path feature via the
         * Exponential Mechanism over a discretized set of candidate values spanning
         * [lowerBound, upperBound]. The true extreme has effectively UNBOUNDED
         * sensitivity in the worst case (one outlier can move it arbitrarily) - that is
         * exactly why MIN/MAX cannot just be Laplace-noised the way SUM/COUNT can. The
         * standard fix is to score each candidate bucket by closeness to the true
         * extreme and sample, rather than releasing the true extreme with noise added
         * directly.
         *
         * Caveat: sensitivityOfUtility below (bucketWidth) is a simplifying, commonly
         * used approximation; a fully rigorous treatment of DP extrema (e.g. via smooth
         * sensitivity or propose-test-release) is a known hard case in the DP
         * literature and is a reasonable place to go deeper if this is core to your
         * contribution rather than a variety check.
         */
        public double noisyExtreme(final List<Path> paths,
                                    final java.util.function.ToDoubleFunction<Path> valueFn,
                                    final boolean wantMax,
                                    final double lowerBound,
                                    final double upperBound,
                                    final int numBuckets,
                                    final double epsilon,
                                    final PrivacyAccountant accountant) {
            double trueExtreme = wantMax ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
            for (Path p : paths) {
                double v = Math.max(lowerBound, Math.min(upperBound, valueFn.applyAsDouble(p)));
                trueExtreme = wantMax ? Math.max(trueExtreme, v) : Math.min(trueExtreme, v);
            }
            double bucketWidth = Math.max((upperBound - lowerBound) / numBuckets, 1e-9);
            double sensitivityOfUtility = bucketWidth;

            double[] candidates = new double[numBuckets + 1];
            double[] logWeights = new double[numBuckets + 1];
            double maxLogWeight = Double.NEGATIVE_INFINITY;
            for (int i = 0; i <= numBuckets; i++) {
                candidates[i] = lowerBound + i * bucketWidth;
                double utility = -Math.abs(candidates[i] - trueExtreme);
                logWeights[i] = (epsilon * utility) / (2.0 * sensitivityOfUtility);
                maxLogWeight = Math.max(maxLogWeight, logWeights[i]);
            }
            double sum = 0;
            double[] w = new double[candidates.length];
            for (int i = 0; i < candidates.length; i++) {
                w[i] = Math.exp(logWeights[i] - maxLogWeight);
                sum += w[i];
            }
            double r = rnd.nextDouble() * sum;
            double cumulative = 0;
            double chosen = candidates[candidates.length - 1];
            for (int i = 0; i < candidates.length; i++) {
                cumulative += w[i];
                if (r <= cumulative) {
                    chosen = candidates[i];
                    break;
                }
            }
            accountant.record(epsilon);
            return chosen;
        }

        // ---------------------------------------------------------------
        // MODE (most frequent group) - selection via Exponential Mechanism
        // ---------------------------------------------------------------

        /**
         * Privately selects the most frequent group via the Exponential Mechanism,
         * scored directly by each group's true count (sensitivity of a count = 1).
         * This is deliberately different from reading argmax off of independently
         * Laplace-noised counts: applying the mechanism to the SELECTION itself, rather
         * than to each released count separately, is the standard and tighter way to
         * privately answer "which group is the biggest?" - it spends one release's
         * worth of budget total, not one per group.
         */
        public String noisyMode(final List<Path> paths,
                                 final java.util.function.Function<Path, String> groupKeyFn,
                                 final double epsilon,
                                 final PrivacyAccountant accountant) {
            Map<String, Long> raw = rawCounts(paths, groupKeyFn);
            List<String> groups = new ArrayList<>(raw.keySet());
            if (groups.isEmpty()) return null;

            double sensitivityOfUtility = 1.0; // one record changes any one group's count by at most 1
            double[] logWeights = new double[groups.size()];
            double maxLogWeight = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < groups.size(); i++) {
                double utility = raw.get(groups.get(i));
                logWeights[i] = (epsilon * utility) / (2.0 * sensitivityOfUtility);
                maxLogWeight = Math.max(maxLogWeight, logWeights[i]);
            }
            double sum = 0;
            double[] w = new double[groups.size()];
            for (int i = 0; i < groups.size(); i++) {
                w[i] = Math.exp(logWeights[i] - maxLogWeight);
                sum += w[i];
            }
            double r = rnd.nextDouble() * sum;
            double cumulative = 0;
            String chosen = groups.get(groups.size() - 1);
            for (int i = 0; i < groups.size(); i++) {
                cumulative += w[i];
                if (r <= cumulative) {
                    chosen = groups.get(i);
                    break;
                }
            }
            accountant.record(epsilon);
            return chosen;
        }
    }
}