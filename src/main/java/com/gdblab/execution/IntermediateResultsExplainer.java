package com.gdblab.execution;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

import com.gdblab.algebra.condition.Condition;
import com.gdblab.algebra.condition.First;
import com.gdblab.algebra.condition.Last;
import com.gdblab.algebra.parser.RPQErrorListener;
import com.gdblab.algebra.parser.RPQExpression;
import com.gdblab.algebra.parser.RPQGrammarListener;
import com.gdblab.algebra.parser.error.SyntaxErrorException;
import com.gdblab.algebra.parser.impl.RPQtoAlgebraVisitor;
import com.gdblab.algebra.queryplan.logical.BinaryLogicalOperator;
import com.gdblab.algebra.queryplan.logical.LogicalOperator;
import com.gdblab.algebra.queryplan.logical.NullaryLogicalOperator;
import com.gdblab.algebra.queryplan.logical.UnaryLogicalOperator;
import com.gdblab.algebra.queryplan.logical.impl.LogicalOpSelection;
import com.gdblab.algebra.queryplan.logical.visitor.LogicalToBFPhysicalVisitor;
import com.gdblab.algebra.queryplan.logical.visitor.PredicatePushdownLogicalPlanVisitor;
import com.gdblab.algebra.queryplan.physical.PhysicalOperator;
import com.gdlab.parser.RPQGrammarLexer;
import com.gdlab.parser.RPQGrammarParser;
import com.gdblab.graph.schema.Path;
import com.gdblab.privacy.SelectiveDPRedactor;

/**
 * Walks the LOGICAL plan tree of a PathDB query and, at every operator node
 * (selection / join / union / recursive / etc.), rebuilds an INDEPENDENT physical
 * sub-plan rooted at that node and fully drains it. This gives you the exact
 * intermediate set of paths that operator would produce during real execution,
 * without disturbing the actual query (which is a separate, freshly-built physical
 * plan each time this is called).
 *
 * Rebuilding per node is intentional: PhysicalOperator instances are destructive
 * iterators (Iterator<Path>), so you cannot "peek" at a live node inside the tree
 * that is actually running without consuming it. Since LogicalOperator nodes are
 * immutable/stateless w.r.t. acceptVisitor(), rebuilding a fresh PhysicalOperator
 * from any LogicalOperator subtree is cheap, safe, and repeatable.
 */
public final class IntermediateResultsExplainer {

    private IntermediateResultsExplainer() {}

    /**
     * Parses {@code query}, builds the (optimized) logical plan exactly the way
     * Execute.EvalRPQWithAlgebra() does, then prints the intermediate set of paths
     * produced by every operator in the tree, from the leaves up to the root.
     *
     * @param query          a complete PathDB query, e.g. "MATCH TRAIL p = (x)-[knows*]->(y) RETURN p;"
     * @param maxPathsPerNode cap on how many paths to print per node (the full set
     *                        is still computed; this only limits console output)
     */
    public static void explain(final String query, final int maxPathsPerNode) {
        explain(query, maxPathsPerNode, null);
    }

    /**
     * Same as {@link #explain(String, int)}, but if {@code redactor} is non-null,
     * every printed path is rendered through it, so any property value the redactor's
     * policy function flags as sensitive is shown privatized (Laplace/Exponential
     * mechanism output) instead of in the clear, while path structure, ids and labels
     * remain exact. Pass null to print raw values (original behavior).
     */
    public static void explain(final String query, final int maxPathsPerNode, final SelectiveDPRedactor redactor) {
        maxPathsPerNodeStatic = maxPathsPerNode;
        // A pass-through redactor flags nothing as sensitive, so renderPath() == renderPathRaw()
        // and no budget is spent - this lets walk() use one uniform code path regardless of
        // whether the caller actually wants privacy applied.
        redactorStatic = (redactor != null) ? redactor : SelectiveDPRedactor.passThrough();
        try {
            LogicalOperator root = parseToLogicalRoot(query);

            System.out.println("\n=== Intermediate values per operator (leaves -> root) ===");
            walk(root, 0);
            System.out.println("=== End of trace ===");
            SelectiveDPRedactor.PrivacyAccountant acc = redactorStatic.getAccountant();
            System.out.printf("Privacy spent this trace: %d sensitive value(s) released, "
                    + "total eps (basic composition) = %.3f%n%n",
                    acc.releaseCount(), acc.totalEpsilonBasicComposition());

            Tools.resetContext();
        } catch (SyntaxErrorException | RecognitionException syntaxError) {
            Tools.resetContext();
            System.out.println(syntaxError.toString());
        }
    }

    /**
     * Same as {@link #explain(String, int, SelectiveDPRedactor)} (prints the full
     * per-operator RAW/DP propagation trace), but also returns the query's complete
     * final result set (not truncated to maxPathsPerNode). Use this when a downstream
     * stage - e.g. an aggregate - needs to consume the SAME query result the trace
     * just displayed propagating through, rather than re-deriving it independently.
     */
    public static List<Path> explainAndReturnFinal(final String query, final int maxPathsPerNode,
                                                     final SelectiveDPRedactor redactor)
            throws SyntaxErrorException, RecognitionException {
        explain(query, maxPathsPerNode, redactor);
        return evaluateFinal(query);
    }

    /**
     * Parses and evaluates {@code query} like {@link #explain}, but returns just the
     * final result path list (the SPJ part of a query: selection + projection pattern
     * + join/recursion), with no printing and no privacy applied. Intended for callers
     * (e.g. an aggregate/"A" stage built on top, such as {@code SPJADemo}) that want to
     * run their own grouping/aggregation over the exact same evaluation PathDB would
     * produce, since PathDB's own grammar has no GROUP BY/aggregate syntax to do this
     * as part of the query itself.
     */
    public static List<Path> evaluateFinal(final String query) throws SyntaxErrorException, RecognitionException {
        LogicalOperator root = parseToLogicalRoot(query);
        List<Path> results = materialize(root);
        Tools.resetContext();
        return results;
    }

    /** Rebuilds an independent physical plan rooted at `node` and fully drains it.
     *  Package-visible so other evaluation tooling (e.g. OperatorLevelBenchmark) in
     *  com.gdblab.execution can materialize any operator's output without duplicating
     *  this rebuild-per-node pattern. */
    static List<Path> materialize(final LogicalOperator node) {
        LogicalToBFPhysicalVisitor v = new LogicalToBFPhysicalVisitor();
        node.acceptVisitor(v);
        PhysicalOperator po = v.getPhysicalPlan().getRootOperator();
        List<Path> results = new ArrayList<>();
        while (po.hasNext()) {
            results.add(po.next());
        }
        return results;
    }

    /** Package-visible (not private) so other evaluation tooling in
     *  com.gdblab.execution can parse a query to its logical plan without
     *  duplicating the ANTLR/visitor wiring. */
    static LogicalOperator parseToLogicalRoot(final String query)
            throws SyntaxErrorException, RecognitionException {
        RPQGrammarLexer lexer = new RPQGrammarLexer(CharStreams.fromString(query));
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        RPQGrammarParser parser = new RPQGrammarParser(tokens);

        parser.removeErrorListeners();
        parser.addErrorListener(new RPQErrorListener());

        ParseTreeWalker walker = new ParseTreeWalker();
        RPQGrammarListener listener = new RPQGrammarListener();
        walker.walk(listener, parser.query());

        RPQExpression rpqExp = Context.getInstance().getRegularExpression();
        RPQtoAlgebraVisitor visitor = new RPQtoAlgebraVisitor();
        rpqExp.acceptVisit(visitor);

        LogicalOperator root = visitor.getRoot();
        Condition condition = Context.getInstance().getCondition();
        return checkAndAddFilter(root, condition);
    }

    private static void walk(final LogicalOperator node, final int depth) {
        // 1. Recurse into children FIRST, so output reads bottom-up like the
        //    paper's evaluation trees (leaves -> root), mirroring Paths0(G)/
        //    Paths1(G) at the bottom and sigma/join/union/phi above them.
        //    This ordering also matters for privacy propagation: whichever operator
        //    first exposes a sensitive fact (almost always a leaf scan) is the one
        //    that gets charged for it; every operator above it just reuses that
        //    same cached, already-privatized value.
        if (node instanceof UnaryLogicalOperator u) {
            walk(u.getChild(), depth + 1);
        } else if (node instanceof BinaryLogicalOperator b) {
            walk(b.getLeftChild(), depth + 1);
            walk(b.getRightChild(), depth + 1);
        } else if (node instanceof NullaryLogicalOperator) {
            // leaf: Paths0(G) / Paths1(G) equivalents, nothing further to recurse into
        }

        // 2. Rebuild an independent physical operator for THIS node only, and drain it.
        List<Path> allResults = materialize(node);
        int total = allResults.size();
        List<Path> results = allResults.subList(0, Math.min(total, maxPathsPerNodeStatic));

        String indent = "  ".repeat(depth);
        System.out.println(indent + "- " + describe(node) + "  [" + total + " path(s)]");

        // Snapshot which facts were already charged (by a child operator below this one)
        // before we render this node's output, so we can tell "new here" from "inherited".
        Set<String> chargedBefore = new HashSet<>(redactorStatic.getChargedFactKeys());

        for (Path p : results) {
            String raw = redactorStatic.renderPathRaw(p);
            String priv = redactorStatic.renderPath(p);
            if (raw.equals(priv)) {
                // nothing in this path is sensitive under the active policy
                System.out.println(indent + "    " + raw);
            } else {
                System.out.println(indent + "    RAW: " + raw);
                System.out.println(indent + "    DP : " + priv);
            }
        }
        if (total > results.size()) {
            System.out.println(indent + "    ... (" + (total - results.size()) + " more, truncated for display)");
        }

        Set<String> newlyCharged = new HashSet<>(redactorStatic.getChargedFactKeys());
        newlyCharged.removeAll(chargedBefore);
        if (!newlyCharged.isEmpty()) {
            System.out.println(indent + "    [privacy] " + newlyCharged.size()
                    + " sensitive value(s) first protected at this operator: " + newlyCharged
                    + " -> reused unchanged at every ancestor operator above");
        }
    }

    // simple static holders so walk() doesn't need to thread these through recursion
    private static int maxPathsPerNodeStatic = 20;
    private static SelectiveDPRedactor redactorStatic = null;

    private static String describe(final LogicalOperator node) {
        // LogicalOp* classes already have informative toString() implementations
        // (e.g. "SELECT<cond>(child)", "NODE-JOIN(l, r)", "UNION(l, r)", "RECURSIVE(l, r)")
        // but those recurse into children's toString too, which gets noisy; show only
        // the operator's own class name plus a short label where useful.
        String cls = node.getClass().getSimpleName();
        if (node instanceof LogicalOpSelection sel) {
            return cls + " <" + sel.getCondition() + ">";
        }
        return cls;
    }

    // duplicated from Execute.java (private there) so this class has no external dependency
    // on Execute's internals; keep in sync if that logic changes.
    private static LogicalOperator checkAndAddFilter(LogicalOperator lo, Condition condition) {
        if (condition == null) {
            return lo;
        }
        if (condition instanceof First || condition instanceof Last) {
            lo = new LogicalOpSelection(lo, condition);
            PredicatePushdownLogicalPlanVisitor v = new PredicatePushdownLogicalPlanVisitor();
            lo.acceptVisitor(v);
            lo = v.getRoot();
            return lo;
        }
        return new LogicalOpSelection(lo, condition);
    }

    /** Overload with the default cap used elsewhere in PathDB output. */
    public static void explain(final String query) {
        explain(query, 20);
    }
}