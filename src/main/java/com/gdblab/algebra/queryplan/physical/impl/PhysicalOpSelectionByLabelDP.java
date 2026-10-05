package com.gdblab.algebra.queryplan.physical.impl;

import java.util.Iterator;

import com.gdblab.algebra.condition.Condition;
import com.gdblab.algebra.condition.Label;
import com.gdblab.algebra.queryplan.logical.impl.LogicalOpSelectionByLabel;
import com.gdblab.algebra.queryplan.physical.NullaryPhysicalOperator;
import com.gdblab.algebra.queryplan.physical.PhysicalPlanVisitor;
import com.gdblab.graph.Graph;
import com.gdblab.graph.schema.Edge;
import com.gdblab.graph.schema.Path;

import java.util.Random;

/**
 * Selective-DP sibling of PhysicalOpSelectionByLabel. Same scan, same output type (Path), so it
 * drops into any plan wherever PhysicalOpSelectionByLabel would go -- upstream of a join, a union,
 * recursion, whatever the query needs. The only behavioral difference: an edge for which
 * `sensitivityCondition` evaluates true on its length-1 Path is never emitted; only a Laplace-noised
 * count of how many were suppressed is available afterward, via getNoisySensitiveCount().
 *
 * Use this when the QUERY'S OWN OUTPUT is the raw path/edge listing (nothing downstream can protect
 * it for you). If a query aggregates or otherwise consumes the result further before returning
 * anything to the user, prefer Utils.printAndCountPathsDP / Utils.sumWithDP instead: they let every
 * non-sensitive edge flow through the ordinary, unmodified operators (any join, any recursion depth,
 * any union) and only decide sensitivity once, at the very end, by re-evaluating the same condition
 * against the complete assembled Path (which always retains its full edge history). Both are valid
 * "operator-level" designs; which one applies depends on where your query actually releases data.
 */
public class PhysicalOpSelectionByLabelDP implements NullaryPhysicalOperator {

    protected final LogicalOpSelectionByLabel lop;
    private final Iterator<Edge> edges;
    private final Condition sensitivityCondition;
    private final double epsilon;
    private final Random random = new Random();

    private Path slot;
    private long releasedCount = 0;
    private long exactSensitiveCount = 0;
    private boolean exhausted = false;

    public PhysicalOpSelectionByLabelDP(LogicalOpSelectionByLabel lop, Condition sensitivityCondition, double epsilon) {
        this.lop = lop;
        this.edges = Graph.getGraph().getEdgeIteratorByLabel(((Label) lop.getCondition()).getLabel());
        this.sensitivityCondition = sensitivityCondition;
        this.epsilon = epsilon;
    }

    @Override
    public void acceptVisitor(final PhysicalPlanVisitor visitor) {
        visitor.visit(this);
    }

    @Override
    public boolean hasNext() {
        if (slot != null) return true;
        while (edges.hasNext()) {
            Edge e = edges.next();
            Path candidate = new Path("", e);
            if (sensitivityCondition.eval(candidate)) {
                exactSensitiveCount++;
                continue;                       // suppressed -- never handed to whatever sits above this operator
            }
            slot = candidate;
            releasedCount++;
            return true;
        }
        exhausted = true;
        return false;
    }

    @Override
    public Path next() {
        if (hasNext()) {
            Path p = slot;
            slot = null;
            return p;
        }
        return null;
    }

    public long getReleasedCount() { return releasedCount; }

    public long getExactSensitiveCount() {
        if (!exhausted) throw new IllegalStateException("Drain the operator fully before reading counts.");
        return exactSensitiveCount;
    }

    /** sensitivity = 1: removing/adding one sensitive edge changes this count by exactly 1. */
    public double getNoisySensitiveCount() {
        if (!exhausted) throw new IllegalStateException("Drain the operator fully before reading counts.");
        double u = random.nextDouble() - 0.5;
        double noise = -(1.0 / epsilon) * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
        return exactSensitiveCount + noise;
    }
}