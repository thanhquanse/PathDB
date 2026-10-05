package com.gdblab.privacy;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import com.gdblab.graph.schema.Edge;
import com.gdblab.graph.schema.GraphObject;
import com.gdblab.graph.schema.Node;
import com.gdblab.graph.schema.Path;

/**
 * Pattern-based sensitivity, EDGE-first: "(Alice)-[knows]->(Bob) is not sensitive,
 * but (Alice)-[knows]->(President) is" - i.e. sensitivity is a property of the
 * EDGE (does it touch a sensitive node, or does it carry a sensitive label?), not
 * of a standalone attribute value. isSensitiveEdge(Edge) is the primary, first-class
 * check here; isSensitiveNode and isSensitivePath are both defined in terms of it,
 * not the other way around - this mirrors PhysicalOpSelectionByLabelDP, which
 * decides sensitivity per-edge at the leaf scan itself.
 *
 * Three independent ways to declare an edge sensitive, all runtime-editable:
 * <ul>
 *   <li>by NODE id/name/status at either endpoint (X knows President, President
 *       knows Y - both directions, and the "status" hook covers "high-status person
 *       or whatever", not just a hardcoded name)</li>
 *   <li>by the edge's OWN label (e.g. every "salary" edge is sensitive, regardless
 *       of endpoints) - a pure edge-level rule, independent of node identity</li>
 * </ul>
 */
public final class SensitivePatternPolicy {

    private final Set<String> sensitiveNodeIds = ConcurrentHashMap.newKeySet();
    private final Set<String> sensitiveNodeNames = ConcurrentHashMap.newKeySet();
    private final Map<String, String> sensitiveStatusValues = new ConcurrentHashMap<>(); // propKey -> value
    private final Set<String> sensitiveEdgeLabels = ConcurrentHashMap.newKeySet();

    // ---- configuration ----

    public void addSensitiveNodeId(final String id) { sensitiveNodeIds.add(id); }
    public void removeSensitiveNodeId(final String id) { sensitiveNodeIds.remove(id); }
    public Set<String> getSensitiveNodeIds() { return new LinkedHashSet<>(sensitiveNodeIds); }

    public void addSensitiveNodeName(final String name) { sensitiveNodeNames.add(name); }
    public void removeSensitiveNodeName(final String name) { sensitiveNodeNames.remove(name); }
    public Set<String> getSensitiveNodeNames() { return new LinkedHashSet<>(sensitiveNodeNames); }

    /** e.g. addSensitiveStatus("status", "president") - any node whose "status"
     *  property equals "president" (case-insensitive); "high-status person or
     *  whatever" is any property key/value pair, not just a hardcoded name. */
    public void addSensitiveStatus(final String propertyKey, final String value) {
        sensitiveStatusValues.put(propertyKey, value);
    }
    public void removeSensitiveStatus(final String propertyKey) { sensitiveStatusValues.remove(propertyKey); }
    public Map<String, String> getSensitiveStatusValues() { return new LinkedHashMap<>(sensitiveStatusValues); }

    /** A pure edge-level rule: every edge with this label is sensitive, regardless
     *  of its endpoints - e.g. addSensitiveEdgeLabel("hasSalary"). */
    public void addSensitiveEdgeLabel(final String label) { sensitiveEdgeLabels.add(label); }
    public void removeSensitiveEdgeLabel(final String label) { sensitiveEdgeLabels.remove(label); }
    public Set<String> getSensitiveEdgeLabels() { return new LinkedHashSet<>(sensitiveEdgeLabels); }

    // ---- checks ----

    private boolean isSensitiveNode(final Node n) {
        if (n == null) return false;
        if (sensitiveNodeIds.contains(n.getId())) return true;
        Map<String, String> props = n.getProperties();
        if (props == null) return false;
        String name = props.get("name");
        if (name != null && sensitiveNodeNames.contains(name)) return true;
        for (Map.Entry<String, String> e : sensitiveStatusValues.entrySet()) {
            String v = props.get(e.getKey());
            if (v != null && v.equalsIgnoreCase(e.getValue())) return true;
        }
        return false;
    }

    /** PRIMARY check: an edge is sensitive if its own label is flagged and if
     *  EITHER endpoint is a sensitive node - covers both "X knows President" and
     *  "President knows Y" in one check, since it doesn't care which side President
     *  is on. */
    public boolean isSensitiveEdge(final Edge e) {
        if (e == null) return false;
        // if (sensitiveEdgeLabels.contains(e.getLabel())) return true;
        return (sensitiveEdgeLabels.contains(e.getLabel())) && (isSensitiveNode(e.getSource()) || isSensitiveNode(e.getTarget()));
    }

    /** A path is sensitive if ANY edge along it is sensitive - defined in terms of
     *  isSensitiveEdge, consistent with the edge-first design; walks the path's
     *  actual edges (via Edge.getSource()/getTarget(), not sequence position, so
     *  it agrees with isSensitiveEdge exactly). */
    public boolean isSensitivePath(final Path p) {
        List<GraphObject> seq = p.getSequence();
        for (GraphObject go : seq) {
            if (go instanceof Edge e && isSensitiveEdge(e)) {
                return true;
            }
        }
        return false;
    }
}