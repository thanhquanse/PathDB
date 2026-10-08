package com.gdblab.execution;

import java.io.IOException;
import java.util.ArrayList;

import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.jline.reader.LineReader;
import org.jline.reader.LineReaderBuilder;
import org.jline.reader.UserInterruptException;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;

import com.gdblab.algebra.condition.Condition;
import com.gdblab.algebra.condition.First;
import com.gdblab.algebra.condition.Last;
import com.gdblab.algebra.parser.RPQErrorListener;
import com.gdblab.algebra.parser.RPQExpression;
import com.gdblab.algebra.parser.RPQGrammarListener;
import com.gdblab.algebra.parser.error.SyntaxErrorException;
import com.gdblab.algebra.parser.error.VariableNotFoundException;
import com.gdblab.algebra.parser.impl.RPQtoAlgebraVisitor;
import com.gdblab.algebra.queryplan.logical.LogicalOperator;
import com.gdblab.algebra.queryplan.logical.impl.LogicalOpSelection;
import com.gdblab.algebra.queryplan.logical.visitor.LogicalToBFPhysicalVisitor;
import com.gdblab.algebra.queryplan.logical.visitor.PredicatePushdownLogicalPlanVisitor;
import com.gdblab.algebra.queryplan.physical.PhysicalOperator;
import com.gdblab.algebra.queryplan.util.Utils;
import com.gdblab.graph.Graph;
import com.gdblab.graph.schema.Edge;
import com.gdblab.privacy.SelectiveDPRedactor;
import com.gdblab.privacy.SensitivePatternPolicy;
import com.gdlab.parser.RPQGrammarLexer;
import com.gdlab.parser.RPQGrammarParser;

public final class Execute {
    // Built once the graph is loaded (see interactive()), reused for the whole REPL
    // session so the privacy accountant composes across every /sdp query the user runs.
    private static SelectiveDPRedactor sdpRedactor = null;

    // EDGE-focused sensitivity policy (separate from sdpRedactor's attribute-key
    // policy F) - configured live via /sdp-config. Persists across graph
    // reloads (a /load doesn't reset which node names/statuses/edge labels the
    // analyst has declared sensitive), unlike sdpRedactor which must be rebuilt
    // per-graph since its domain index is graph-derived.
    private static SensitivePatternPolicy sdpPattern = new SensitivePatternPolicy();

    /** "/sdp-config [subcommand] [arg]" - configures sdpPattern, the
     *  EDGE-focused sensitivity policy: an edge is sensitive if its own label is
     *  flagged, or if either endpoint node matches a flagged id/name/status. Same
     *  live-editable style as handleSdpSensitiveCommand, different policy object. */
    private static void handleSdpSensitiveEdgeCommand(final String rest) {
        String[] parts = rest.isEmpty() ? new String[0] : rest.split("\\s+", 2);
        String sub = parts.length > 0 ? parts[0].toLowerCase() : "show";
        String arg = parts.length > 1 ? parts[1].trim() : "";

        switch (sub) {
            case "show", "" -> {
                System.out.println("Edge sensitivity pattern - current configuration:");
                System.out.println("  sensitive node ids:     " + sdpPattern.getSensitiveNodeIds());
                System.out.println("  sensitive node names:    " + sdpPattern.getSensitiveNodeNames());
                System.out.println("  sensitive node statuses: " + sdpPattern.getSensitiveStatusValues());
                System.out.println("  sensitive edge labels:   " + sdpPattern.getSensitiveEdgeLabels());
                System.out.println("  an edge is sensitive if its own label is flagged, OR either endpoint node matches.\n");
            }
            case "add-node-name" -> {
                if (arg.isEmpty()) { System.out.println("Usage: /sdp-config add-node-name <name>\n"); }
                else { sdpPattern.addSensitiveNodeName(arg); System.out.println("Added sensitive node name: " + arg + "\n"); }
            }
            case "remove-node-name" -> {
                if (arg.isEmpty()) { System.out.println("Usage: /sdp-config remove-node-name <name>\n"); }
                else { sdpPattern.removeSensitiveNodeName(arg); System.out.println("Removed sensitive node name: " + arg + "\n"); }
            }
            case "add-node-id" -> {
                if (arg.isEmpty()) { System.out.println("Usage: /sdp-config add-node-id <id>\n"); }
                else { sdpPattern.addSensitiveNodeId(arg); System.out.println("Added sensitive node id: " + arg + "\n"); }
            }
            case "remove-node-id" -> {
                if (arg.isEmpty()) { System.out.println("Usage: /sdp-config remove-node-id <id>\n"); }
                else { sdpPattern.removeSensitiveNodeId(arg); System.out.println("Removed sensitive node id: " + arg + "\n"); }
            }
            case "add-status" -> {
                String[] kv = arg.split("=", 2);
                if (kv.length != 2) { System.out.println("Usage: /sdp-config add-status <propertyKey>=<value>\n"); }
                else { sdpPattern.addSensitiveStatus(kv[0].trim(), kv[1].trim());
                       System.out.println("Added sensitive status: " + kv[0].trim() + "=" + kv[1].trim() + "\n"); }
            }
            case "remove-status" -> {
                if (arg.isEmpty()) { System.out.println("Usage: /sdp-config remove-status <propertyKey>\n"); }
                else { sdpPattern.removeSensitiveStatus(arg); System.out.println("Removed sensitive status key: " + arg + "\n"); }
            }
            case "add-edge-label" -> {
                if (arg.isEmpty()) { System.out.println("Usage: /sdp-config add-edge-label <label>\n"); }
                else { sdpPattern.addSensitiveEdgeLabel(arg); System.out.println("Added sensitive edge label: " + arg + "\n"); }
            }
            case "remove-edge-label" -> {
                if (arg.isEmpty()) { System.out.println("Usage: /sdp-config remove-edge-label <label>\n"); }
                else { sdpPattern.removeSensitiveEdgeLabel(arg); System.out.println("Removed sensitive edge label: " + arg + "\n"); }
            }
            default -> System.out.println("Unknown /sdp-config subcommand: " + sub
                    + "\nUsage: /sdp-config [show | add-node-name <n> | remove-node-name <n> | "
                    + "add-node-id <id> | remove-node-id <id> | add-status <key>=<value> | remove-status <key> | "
                    + "add-edge-label <label> | remove-edge-label <label>]\n");
        }
    }

    public static void EvalRPQWithAlgebra() {
        long start = System.nanoTime();
        int counter = 1;

        PhysicalOperator po = null;

        try {
            RPQGrammarLexer lexer = new RPQGrammarLexer(CharStreams.fromString(Context.getInstance().getCompleteQuery()));
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

            LogicalOperator lo = visitor.getRoot();

            Condition condition = Context.getInstance().getCondition();
            lo = checkAndAddFilter(lo, condition);

            LogicalToBFPhysicalVisitor visitor2 = new LogicalToBFPhysicalVisitor();
            lo.acceptVisitor(visitor2);
            po = visitor2.getPhysicalPlan().getRootOperator();

            counter = Utils.printAndCountPaths(po);

            long end = System.nanoTime();
            System.out.println("\nTotal paths: " + (counter - 1) + " paths");
            System.out.println("Execution time: " + Utils.getTime(start, end) + " seconds");
            System.out.println("");

            Tools.resetContext();
            // return Context.getInstance().getCompleteQuery() + Utils.getTime(start, end);
        } catch (SyntaxErrorException | RecognitionException syntaxError) {
            Tools.resetContext();
            System.out.println(syntaxError.toString());
            // return Context.getInstance().getCompleteQuery() + "999.999";
        } catch (OutOfMemoryError e) {
            System.gc();
            Tools.resetContext();
            System.out.println("Out of memory error. Try again with more memory.\n");
            // return Context.getInstance().getCompleteQuery() + "999.999";
        }
    }

    public static void interactive(String[] args) {

        try {
            Terminal terminal = TerminalBuilder.terminal();
            LineReader reader = LineReaderBuilder.builder().terminal(terminal).build();

            reader.getHistory().add("");

            Tools.clearConsole();

            if (args.length == 0) {
                Tools.showUsageNoArgs();
                Tools.loadDefaultGraph();
            } else {
                Tools.showUsageArgsLoadingCustomGraph(args[0], args[1]);
                Tools.loadCustomGraphFiles(args[0], args[1]);
            }

            // Must be built AFTER the graph is loaded: its constructor scans the graph
            // to index candidate values for the categorical (Exponential Mechanism) DP.
            sdpRedactor = SelectiveDPRedactor.withDefaults();

            String prompt = "PathDB> ";

            // ServerSocket ss = new ServerSocket(12000);
            // System.out.println("Server started on port 12000. Waiting for client connections...");
            // while (true) {
            //     try (Socket clientSocket = ss.accept(); BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream())); PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)) {
            //         String input = in.readLine();
            //         System.out.println("Received: " + input);
            //         Context.getInstance().setCompleteQuery(input);
            //         String res = EvalRPQWithAlgebra();
            //         out.println(res);
            //     }
            // }
            while (true) {
                String line = reader.readLine(prompt);
                reader.getHistory().add(line);
                if (line.equalsIgnoreCase("/h") || line.equalsIgnoreCase("/help")) {
                    Tools.showHelp();
                    System.out.println("                      Load a dataset, same flag style as the PathDB CLI. Rebuilds sdpRedactor.");
                    System.out.println("  /sdp-config [show|add-node-name ..|add-status k=v|add-edge-label ..|...]");
                    System.out.println("                      Inspect/edit the EDGE-focused sensitivity pattern (e.g. 'President'):");
                    System.out.println("                      an edge is sensitive if either endpoint matches, or its own label does.");
                    System.out.println("  /sdp-execute <query>;  Run the sensitive-EDGE-pattern operator-level benchmark (gradual");
                    System.out.println("                      per-operator summing, then selective vs uniform DP) on the loaded");
                    System.out.println("                      graph, for any query - see Evaluator.java.");
                    System.out.println();
                } else if (line.equalsIgnoreCase("/in") || line.equalsIgnoreCase("/information")) {
                    Tools.showInformation();
                } else if (line.equalsIgnoreCase("/la") || line.equalsIgnoreCase("/labels")) {
                    System.out.println("Samples: ");
                    ArrayList<Edge> edges = Graph.getGraph().getSampleOfEachlabel();
                    for (Edge e : edges) {
                        System.out.println(e.getId() + ": " + e.getSource().getId() + "," + e.getLabel() + "," + e.getTarget().getId());
                    }
                    System.out.println("");
                } else if (line.startsWith("/explain ")) {
                    String q = line.substring("/explain ".length());
                    Context.getInstance().setCompleteQuery(q);
                    IntermediateResultsExplainer.explain(q, 20);
                    Tools.resetContext();
                } else if (line.equalsIgnoreCase("/q") || line.equalsIgnoreCase("/quit")) {
                    System.out.println("Exiting...");
                    System.exit(0);
		        } else if (line.toLowerCase().startsWith("/sdp-config")) {
                    handleSdpSensitiveEdgeCommand(line.substring("/sdp-config".length()).trim());
                } else if (line.toLowerCase().startsWith("/sdp-execute ")) {
                    // "/sdp-execute <query>;" - runs the sensitive-EDGE-pattern
                    // operator-level benchmark against WHATEVER graph is currently
                    // loaded (via /load) using sdpPattern (configured via
                    // /sdp-config), for any query PathDB supports - no file
                    // paths here, load the dataset separately with /load first.
                    String q = line.substring("/sdp-execute ".length()).trim();
                    if (q.isEmpty() || !q.endsWith(";")) {
                        System.out.println("Usage: /sdp-execute <query>;  (load a dataset first with /load -n ... -e ...)\n");
                    } else {
                        try {
                            // Evaluator.run(q, sdpPattern, sdpRedactor.getEpsilonPerRelease(), 600);
                            Evaluator.compare(q, sdpPattern, new double[]{0.1, 0.5, 1, 2, 5, 10}, 600.0, 50, 42L);
                        } catch (Exception e) {
                            System.out.println(e);
                        }
                    }
                } else if (line.endsWith(";")) {
                    try {
                        Context.getInstance().setCompleteQuery(line);
                        EvalRPQWithAlgebra();
                    } catch (OutOfMemoryError e) {
                        System.out.println("Out of memory error. Try again with more memory.\n");
                    } catch (VariableNotFoundException e) {
                        System.out.println(e.toString());
                    } catch (Exception e) {
                        System.out.println(e);
                    }
                } else {
                    System.out.println("Invalid command. Type /h or /help for help.\n");
                }
            }
        } catch (IOException e) {
            System.out.println(e.toString());
        } catch (UserInterruptException e) {
            System.out.println("\nExiting...");
            System.exit(0);
        }
    }

    private static LogicalOperator checkAndAddFilter(LogicalOperator lo, Condition condition) {
        if (condition == null) {
            // Caso sin condiciones
            return lo;
        }
        // if (condition instanceof And) {
        //     // Significa que es un And compuesto
        //     Condition leftCond = ((And) condition).getC1();
        //     Condition rightCond = ((And) condition).getC2();
        //     if (leftCond instanceof First && rightCond instanceof First) {
        //         lo = new LogicalOpSelection(lo, condition);
        //         PredicatePushdownLogicalPlanVisitor v = new PredicatePushdownLogicalPlanVisitor();
        //         lo.acceptVisitor(v);
        //         lo = v.getRoot();
        //         return lo;
        //     }
        //     if (leftCond instanceof First && rightCond instanceof Last) {
        //         lo = new LogicalOpSelection(lo, leftCond);
        //         PredicatePushdownLogicalPlanVisitor v = new PredicatePushdownLogicalPlanVisitor();
        //         lo.acceptVisitor(v);
        //         lo = v.getRoot();
        //         return new LogicalOpSelection(lo, rightCond);
        //     }
        //     if (leftCond instanceof Last && rightCond instanceof First) {
        //         lo = new LogicalOpSelection(lo, rightCond);
        //         PredicatePushdownLogicalPlanVisitor v = new PredicatePushdownLogicalPlanVisitor();
        //         lo.acceptVisitor(v);
        //         lo = v.getRoot();
        //         return new LogicalOpSelection(lo, leftCond);
        //     }
        //     if (leftCond instanceof Last && rightCond instanceof Last) {
        //         lo = new LogicalOpSelection(lo, condition);
        //         PredicatePushdownLogicalPlanVisitor v = new PredicatePushdownLogicalPlanVisitor();
        //         lo.acceptVisitor(v);
        //         lo = v.getRoot();
        //         return lo;
        //     }
        // }
        if (condition instanceof First || condition instanceof Last) {
            // Significa que es un First o Last y simplemente se baja
            lo = new LogicalOpSelection(lo, condition);
            PredicatePushdownLogicalPlanVisitor v = new PredicatePushdownLogicalPlanVisitor();
            lo.acceptVisitor(v);
            lo = v.getRoot();
            return lo;
        }

        // Caso en que sea cualquiera otra condicion no se puede bajar
        return new LogicalOpSelection(lo, condition);
    }

}
