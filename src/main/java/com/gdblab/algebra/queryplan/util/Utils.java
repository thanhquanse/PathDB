package com.gdblab.algebra.queryplan.util;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import com.gdblab.algebra.queryplan.physical.PhysicalOperator;
import com.gdblab.algebra.returncontent.ReturnContent;
import com.gdblab.execution.Context;
import com.gdblab.graph.schema.Edge;
import com.gdblab.graph.schema.Node;
import com.gdblab.graph.schema.Path;

import de.vandermeer.asciitable.AsciiTable;
import de.vandermeer.asciitable.CWC_FixedWidth;
import de.vandermeer.skb.interfaces.transformers.textformat.TextAlignment;
import com.gdblab.algebra.condition.Condition;

public class Utils {

    public static List<Path> iterToList(final PhysicalOperator physicalOp) {
        List<Path> l = new ArrayList<>();
        while (physicalOp.hasNext()) {
            l.add(physicalOp.next());
        }
        return l;
    }

    public static List<Edge> edgesIterToList(Iterator<Edge> edges) {
        List<Edge> l = new ArrayList<>();
        while (edges.hasNext()) {
            l.add(edges.next());
        }
        return l;
    }

    public static List<Node> nodesIterToList(Iterator<Node> nodes) {
        List<Node> l = new ArrayList<>();
        while (nodes.hasNext()) {
            l.add(nodes.next());
        }
        return l;
    }

    public static String getTime(long start, long end) {
        long duration = end - start;
        double durationInSeconds = (double) duration / 1_000_000_000.0;
        return String.format("%.3f", durationInSeconds);
    }

    public static Path NodeLink(Path pathA, Path pathB) {

        if (pathA.isNodeLinkable(pathB) && pathA.getSumEdges(pathB) <= Context.getInstance().getMaxPathsLength()) {

            switch (Context.getInstance().getSemantic()) {
                case 2:
                    if (!pathA.isTrail(pathB)) {
                        return null;
                    }
                    break;
                case 3:
                    if (!pathA.isAcyclic(pathB)) {
                        return null;
                    }
                    break;
                case 4:
                    if (!pathA.isSimplePath(pathB)) {
                        return null;
                    }
                    break;
            }

            Path join_path = new Path("", (pathA.getEdgeLength() + pathB.getEdgeLength()));

            if (pathA.getNodesAmount() == 1 && pathB.getNodesAmount() == 1) {
                join_path.insertNode(pathA.first());
            } else {
                join_path.setSequence(pathA.getSequence());
                join_path.appendSequence(pathB.getSequence());
            }

            return join_path;
        }

        return null;
    }

    public static int printAndCountPaths(PhysicalOperator po) {
        Integer counterLP = 1;

        ArrayList<ReturnContent> returnContentList = Context.getInstance().getReturnedVariables();
        Integer limitCalculatePaths = Context.getInstance().getLimit();

        AsciiTable table = new AsciiTable();
        List<String> columnNames = returnContentList.stream()
                .map(ReturnContent::getReturnName)
                .toList();
        ArrayList<String> columnNamesWithLength = new ArrayList<>(columnNames);
        columnNamesWithLength.add(0, "#");
        CWC_FixedWidth cwc = new CWC_FixedWidth();
        cwc.add(10);
        for (int i = 0; i < columnNames.size(); i++) {
            cwc.add(30);
        }
        table.getRenderer().setCWC(cwc);
        table.addRule();
        table.addRow(columnNamesWithLength).setTextAlignment(TextAlignment.CENTER);
        table.addRule();
        while (counterLP <= limitCalculatePaths && po.hasNext()) {

            Path p = po.next();

            List<String> row = new ArrayList<>();
            row.add(String.valueOf(counterLP));
            for (ReturnContent returnContent : returnContentList) {
                String content = returnContent.getContent(p);
                row.add(content);
            }
            table.addRow(row).setTextAlignment(TextAlignment.CENTER);
            table.addRule();
            counterLP++;
        }

        System.out.println();
        System.out.println(table.render());
        return counterLP;
    }

     /**
     * Drains a physical plan, releasing non-sensitive paths exactly and a noisy count of the rest.
     * Works regardless of plan shape/depth: `condition` is evaluated against each COMPLETE result
     * Path, the same way any WHERE-clause condition already is elsewhere in PathDB.
     */
    public static int printAndCountPathsDP(final PhysicalOperator po, final Condition condition, double epsilon) {
        int counter = 1;
        long sensitiveCount = 0;
        while (po.hasNext()) {
            Path p = po.next();
            if (condition.eval(p)) {
                sensitiveCount++;
                continue;
            }
            System.out.println(counter + ": " + p);
            counter++;
        }
        double u = new java.util.Random().nextDouble() - 0.5;
        double noise = -(1.0 / epsilon) * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
        System.out.println("(+ " + String.format("%.2f", sensitiveCount + noise) + " sensitive paths, protected)");
        return counter;
    }

    /**
     * Selective-DP SUM over a numeric node property (e.g. Message.length) at a given 1-indexed
     * node position in each result path. `clipBound` bounds the contribution of one sensitive path;
     * non-sensitive contributions are exact and unclipped.
     */
    public static double sumWithDP(final PhysicalOperator po, final Condition sensitivityCondition,
                                    final String valueProperty, final int valueNodePos,
                                    final double clipBound, final double epsilon) {
        double exactSum = 0, clippedSensitiveSum = 0;
        while (po.hasNext()) {
            Path p = po.next();
            double value = Double.parseDouble(p.getNodeAt(valueNodePos - 1).getProperty(valueProperty));
            if (sensitivityCondition.eval(p)) {
                clippedSensitiveSum += Math.min(Math.max(value, 0), clipBound);
            } else {
                exactSum += value;
            }
        }
        double u = new java.util.Random().nextDouble() - 0.5;
        double noise = -(clipBound / epsilon) * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
        return exactSum + clippedSensitiveSum + noise;
    }

}
