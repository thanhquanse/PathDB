package com.gdblab.graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import com.gdblab.graph.interfaces.InterfaceGraph;
import com.gdblab.graph.schema.Edge;
import com.gdblab.graph.schema.Node;

public final class DefaultGraph {

    private static InterfaceGraph graph = Graph.getGraph();

    private static final int PERSON_COUNT = 100;
    private static final int MESSAGE_COUNT = 100;
    private static final long RANDOM_SEED = 42L;

    private static final String[] PERSON_NAMES = {
            "Moe",
            "Bart",
            "Lisa",
            "Apu",
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Eve",
            "Frank",
            "Grace",
            "Helen",
            "Ivy",
            "Jack",
            "Karen",
            "Leo",
            "Maria",
            "Nina",
            "Oscar",
            "Paul",
            "Quinn",
            "Rachel",
            "Sam",
            "Tom",
            "Uma",
            "Victor",
            "Wendy",
            "Xavier",
            "Yara",
            "Zack"
    };

    private DefaultGraph() {
    }

    public static Node[] getDefaultNodes() {
        Node[] nodes = new Node[PERSON_COUNT + MESSAGE_COUNT];

        for (int i = 1; i <= PERSON_COUNT; i++) {
            HashMap<String, String> personProperties = new HashMap<>();

            String personName;

            if (i == 1) {
                personName = "President";
            } else if (i == 2) {
                personName = "CEO";
            } else {
                personName = PERSON_NAMES[(i - 3) % PERSON_NAMES.length];
            }

            personProperties.put("name", personName);

            nodes[i - 1] = new Node(
                    "p" + i,
                    "Person",
                    personProperties
            );
        }

        for (int i = 1; i <= MESSAGE_COUNT; i++) {
            HashMap<String, String> messageProperties = new HashMap<>();

            String messageText = "Msg" + i;

            messageProperties.put("txt", messageText);
            messageProperties.put(
                    "length",
                    String.valueOf(messageText.length())
            );

            nodes[PERSON_COUNT + i - 1] = new Node(
                    "m" + i,
                    "Message",
                    messageProperties
            );
        }

        return nodes;
    }

    public static Edge[] getDefaultEdges() {
        Random random = new Random(RANDOM_SEED);
        List<Edge> edges = new ArrayList<>();

        int edgeId = 1;

        /*
         * Person -> Person
         * Three knows edges per person.
         */
        for (int sourceIndex = 1; sourceIndex <= PERSON_COUNT; sourceIndex++) {
            for (int i = 0; i < 3; i++) {
                int targetIndex;

                do {
                    targetIndex = 1 + random.nextInt(PERSON_COUNT);
                } while (targetIndex == sourceIndex);

                edges.add(new Edge(
                        "E" + edgeId++,
                        "knows",
                        graph.getNode("p" + sourceIndex),
                        graph.getNode("p" + targetIndex),
                        new HashMap<String, String>()
                ));
            }
        }

        /*
         * Person -> Message
         * Two likes edges per person.
         */
        for (int personIndex = 1; personIndex <= PERSON_COUNT; personIndex++) {
            for (int i = 0; i < 2; i++) {
                int messageIndex = 1 + random.nextInt(MESSAGE_COUNT);

                edges.add(new Edge(
                        "E" + edgeId++,
                        "likes",
                        graph.getNode("p" + personIndex),
                        graph.getNode("m" + messageIndex),
                        new HashMap<String, String>()
                ));
            }
        }

        /*
         * Message -> Person
         * Every Message has one creator.
         */
        for (int messageIndex = 1; messageIndex <= MESSAGE_COUNT; messageIndex++) {
            int creatorIndex = 1 + random.nextInt(PERSON_COUNT);

            edges.add(new Edge(
                    "E" + edgeId++,
                    "hasCreator",
                    graph.getNode("m" + messageIndex),
                    graph.getNode("p" + creatorIndex),
                    new HashMap<String, String>()
            ));
        }

        return edges.toArray(new Edge[0]);
    }
}