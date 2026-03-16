OTP-PD: Object Type Perspective-Based Process Discovery
A process mining toolkit for OCEL (Object-Centric Event Logs) that discovers per–object-type lifecycle models and renders interactive visualizations.

Overview
OTP-PD extracts one Object Type Perspective (OTP) per object type in an OCEL log. For each perspective it:

Mines the correct lifecycle (activity) order using a directly-follows graph and topological sort — the standard process mining heuristic.
Computes inter-object-type relationships (direct, co-occurrence, or both).
Renders an annotated lifecycle diagram per object type showing event boxes, partner ellipses, shared event lanes, and end-of-lifecycle markers.
