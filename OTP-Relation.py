"""
improved_extract_all_otps.py

Improved extraction of Object Type Perspectives (OTPs) from an OCEL.

This script provides a single high-level function `extract_all_otps(ocel, ...)`
that returns a rich dictionary of OTP information for every object type and
prints a concise human-readable summary matching the user's 6 requirements:

  1) print the name of the selected object type
  2) the total number of objects belonging to it
  3) name all of the event types that have E2O relation
  4) total number of events belonging to every event type in perspective
  5) name of object types that have O2O (direct, co-occurrence in events, or both)
  6) number of objects allocated with object type of O2O relation (for each partner)

The O2O calculation follows the method the user provided: it filters the OCEL
for each pair of object types (A,B), writes/loads JSON, then computes
- direct relations by counting `relationships` on filtered objects (with
  unique-edge de-duplication), and
- shared-event co-occurrence by counting for each event how many objects of A
  and B participate and summing num_A * num_B (plus reporting which objects
  actually co-occurred).

The function is defensive: it will try to use pm4py filtering to generate the
pair-filtered OCEL, and if that isn't available it will write the whole OCEL
JSON and then filter in-memory.

New features in this extended version:
 - Counts object attribute names per object type (OA_ot_count)
 - Categorizes object-type attributes using heuristics (datatype + semantics):
     * numeric, string, datetime, boolean, other
     * id_like (name contains 'id' or uniqueness==num_objects)
     * temporal (name contains 'time'/'date' or dtype datetime)
     * categorical (low cardinality)
 - Counts total events per perspective (total_events)
 - Counts and categorizes event attributes per object type perspective
 - Adds these categorized lists into the returned results and prints a brief
   summary when verbose=True.

Dependencies: pm4py, pandas

Return value: dict keyed by object-type. Each value contains detailed fields
(see function docstring inside file).

Example usage:
    from improved_extract_all_otps import extract_all_otps
    results = extract_all_otps(ocel, verbose=True)

"""

from collections import defaultdict
import itertools
import json
import tempfile
import os
from typing import Dict, Any, Tuple

try:
    import pandas as pd
    import pm4py
except Exception as e:
    # We don't raise here to keep the file importable in editors; errors will
    # be raised when functions that require pm4py are invoked.
    pm4py = None


def _write_ocel_to_json(ocel, path: str):
    """Write an OCEL to JSON using pm4py if available.

    If pm4py.write.write_ocel2_json fails, this function will raise the
    underlying exception.
    """
    if pm4py is None:
        raise RuntimeError("pm4py is required for OCEL JSON serialization but not available")
    # pm4py exposes write_ocel2_json in many installations; keeping call direct
    pm4py.write.write_ocel2_json(ocel, path)


def _compute_pair_o2o_metrics(ocel, typeA: str, typeB: str, tmp_dir: str = None) -> Dict[str, Any]:
    """Compute O2O metrics for the pair (typeA, typeB) using the user's JSON method.

    Returns a dictionary with keys:
      - relationship_count: raw sum of relationships lists lengths (may double count)
      - unique_direct_pairs_count: number of unique undirected object-object edges
      - direct_oids_A, direct_oids_B: sets of object ids of typeA/typeB that are directly connected
      - event_relation_cnt: shared-event co-occurrence count (sum num_A * num_B across events)
      - coocc_oids_A, coocc_oids_B: sets of object ids of typeA/typeB that co-occur in >=1 event
    """
    # first try to use pm4py built-in filtering to create a filtered OCEL
    use_pm4py_filter = True
    filtered_json_data = None

    tmp_dir = tmp_dir or tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=tmp_dir) as tmpf:
        tmp_path = tmpf.name

    try:
        if pm4py is None:
            raise RuntimeError("pm4py not available")

        # try filter_ocel_object_types (user-provided snippet uses this)
        try:
            filtered_ocel = pm4py.filtering.filter_ocel_object_types(ocel, [typeA, typeB])
            # write filtered ocEL
            _write_ocel_to_json(filtered_ocel, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as fh:
                filtered_json_data = json.load(fh)
        except Exception:
            # fallback: write full ocel JSON and filter in-memory
            _write_ocel_to_json(ocel, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as fh:
                full = json.load(fh)
            # filter objects of requested types
            filtered_objects = [o for o in full.get("objects", []) if o.get("type") in {typeA, typeB}]
            allowed_oids = {o.get("id") for o in filtered_objects}
            # filter events and restrict relationships to allowed_oids
            filtered_events = []
            for e in full.get("events", []):
                # each relationship is an object like {"objectId": ..., ...}
                rels = [r for r in e.get("relationships", []) if r.get("objectId") in allowed_oids]
                if rels:
                    ev = dict(e)
                    ev["relationships"] = rels
                    filtered_events.append(ev)
            filtered_json_data = {"objects": filtered_objects, "events": filtered_events}

        # Now compute metrics from filtered_json_data
        data = filtered_json_data or {"objects": [], "events": []}

        # object id -> type mapping for filtered objects
        omap = {o["id"]: o.get("type") for o in data.get("objects", [])}

        relationship_count = 0
        unique_edges = set()
        direct_oids_A = set()
        direct_oids_B = set()

        for o in data.get("objects", []):
            rels = o.get("relationships", [])
            relationship_count += len(rels)
            oid = o.get("id")
            otype = o.get("type")
            for r in rels:
                rid = r.get("objectId")
                if rid in omap:
                    # undirected unique edge
                    unique_edges.add(frozenset({oid, rid}))
                    # mark which side this object belongs to
                    if otype == typeA and omap[rid] == typeB:
                        direct_oids_A.add(oid)
                        direct_oids_B.add(rid)
                    if otype == typeB and omap[rid] == typeA:
                        direct_oids_B.add(oid)
                        direct_oids_A.add(rid)

        unique_direct_pairs_count = len(unique_edges)

        # now shared-event co-occurrence counts
        event_relation_cnt = 0
        coocc_oids_A = set()
        coocc_oids_B = set()

        for ev in data.get("events", []):
            # count how many relationships in this event belong to each type
            rels = ev.get("relationships", [])
            oids_A = [r.get("objectId") for r in rels if omap.get(r.get("objectId")) == typeA]
            oids_B = [r.get("objectId") for r in rels if omap.get(r.get("objectId")) == typeB]
            na = len(oids_A)
            nb = len(oids_B)
            if na > 0 and nb > 0:
                event_relation_cnt += na * nb
                coocc_oids_A.update(oids_A)
                coocc_oids_B.update(oids_B)

        # Cleanup tmp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return {
            "relationship_count": relationship_count,
            "unique_direct_pairs_count": unique_direct_pairs_count,
            "direct_oids_A": direct_oids_A,
            "direct_oids_B": direct_oids_B,
            "event_relation_cnt": event_relation_cnt,
            "coocc_oids_A": coocc_oids_A,
            "coocc_oids_B": coocc_oids_B,
        }

    except Exception as exc:
        # ensure cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _categorize_object_attributes(objects_df: "pd.DataFrame", obj_ids: set) -> Dict[str, Any]:
    """Heuristic categorization of object attributes for a selected set of object ids.

    Returns a dict containing:
      - oa_names: set of attribute names
      - oa_count: number of attributes
      - categorized: dict of categories -> list of attribute names

    Categories applied (best-effort):
      - numeric, string, datetime, boolean, other
      - id_like (contains 'id' or unique values == num_objects)
      - temporal (contains 'time' or 'date' or dtype datetime)
      - categorical (low cardinality: unique <= 0.2 * num_objects and >1)

    The function is defensive: it will ignore attributes that are part of the
    OCEL plumbing columns (ocel:oid, ocel:type) and only consider user attributes
    stored as additional columns in objects_df.
    """
    import pandas as pd
    res = {}
    # restrict to rows for these object ids
    sub = objects_df[objects_df["ocel:oid"].isin(obj_ids)]
    if sub.empty:
        return {"oa_names": set(), "oa_count": 0, "categorized": {}}

    # candidate attribute columns: all except ocel:oid and ocel:type
    candidates = [c for c in sub.columns if c not in {"ocel:oid", "ocel:type"}]

    oa_names = set(candidates)
    num_objects = len(sub)

    categorized = {
        "numeric": [],
        "string": [],
        "datetime": [],
        "boolean": [],
        "other": [],
        "id_like": [],
        "temporal": [],
        "categorical": [],
    }

    for col in candidates:
        series = sub[col].dropna()
        # detect dtype
        dtype = series.dtype if not series.empty else objects_df[col].dtype if col in objects_df.columns else None
        is_datetime = pd.api.types.is_datetime64_any_dtype(dtype)
        is_numeric = pd.api.types.is_numeric_dtype(dtype)
        is_bool = pd.api.types.is_bool_dtype(dtype)

        # basic datatype bucket
        if is_datetime:
            categorized["datetime"].append(col)
        elif is_numeric:
            categorized["numeric"].append(col)
        elif is_bool:
            categorized["boolean"].append(col)
        else:
            categorized["string"].append(col)

        # semantic heuristics
        lname = col.lower()
        unique_vals = series.unique() if not series.empty else []
        unique_count = len(unique_vals)

        # id-like: name contains 'id' or unique count equals num_objects
        if "id" in lname or unique_count == num_objects:
            categorized["id_like"].append(col)

        # temporal: name hints or datetime dtype
        if "time" in lname or "date" in lname or is_datetime:
            categorized["temporal"].append(col)

        # categorical: low cardinality but >1 (not constant)
        if 1 < unique_count <= max(1, int(0.2 * num_objects)):
            categorized["categorical"].append(col)

        # fallback to other
        if col not in sum([categorized[k] for k in ["numeric", "string", "datetime", "boolean"]], []):
            categorized["other"].append(col)

    res["oa_names"] = oa_names
    res["oa_count"] = len(oa_names)
    res["categorized"] = categorized
    return res


def _categorize_event_attributes(events_df: "pd.DataFrame", event_attrs_df, related_eids: set, ET_ot: set) -> Dict[str, Any]:
    """Heuristic categorization of event attributes for a given set of event ids.

    Returns a dict containing:
      - ea_names: set of attribute names
      - ea_count: number of attributes
      - categorized: dict of categories -> list of attribute names

    Categories mirror _categorize_object_attributes:
      - numeric, string, datetime, boolean, other
      - id_like (name contains 'id' or unique values == num_events)
      - temporal (name contains 'time' or 'date' or dtype datetime)
      - categorical (low cardinality: unique <= 0.2 * num_events and >1)

    OCEL plumbing columns (ocel:eid, ocel:activity, ocel:timestamp) are excluded.
    """
    import pandas as pd

    PLUMBING = {"ocel:eid", "ocel:activity", "ocel:timestamp"}

    # Restrict events to those related to this object type's perspective
    sub = events_df[events_df["ocel:eid"].isin(related_eids)]

    if sub.empty:
        return {"ea_names": set(), "ea_count": 0, "categorized": {}}

    # If a dedicated event_attributes dataframe is available, use it to find
    # attribute names for the relevant activity types; otherwise fall back to
    # extra columns on events_df.
    if event_attrs_df is not None:
        try:
            ea_names = set(
                event_attrs_df[event_attrs_df["ocel:activity"].isin(ET_ot)]["ocel:attr:name"].unique()
            )
        except Exception:
            ea_names = set(sub.columns) - PLUMBING
    else:
        ea_names = set(sub.columns) - PLUMBING

    num_events = len(sub)

    categorized = {
        "numeric": [],
        "string": [],
        "datetime": [],
        "boolean": [],
        "other": [],
        "id_like": [],
        "temporal": [],
        "categorical": [],
    }

    # We can only categorize dtype/cardinality for attributes that are actual
    # columns on events_df. For attributes that only exist in event_attrs_df,
    # we record the name but skip dtype analysis.
    analysable_cols = [c for c in ea_names if c in sub.columns]
    name_only_cols = [c for c in ea_names if c not in sub.columns]

    # Name-only attributes (from event_attrs_df but not a column): put in "other"
    categorized["other"].extend(name_only_cols)

    for col in analysable_cols:
        series = sub[col].dropna()
        dtype = series.dtype if not series.empty else sub[col].dtype
        is_datetime = pd.api.types.is_datetime64_any_dtype(dtype)
        is_numeric = pd.api.types.is_numeric_dtype(dtype)
        is_bool = pd.api.types.is_bool_dtype(dtype)

        # basic datatype bucket
        if is_datetime:
            categorized["datetime"].append(col)
        elif is_numeric:
            categorized["numeric"].append(col)
        elif is_bool:
            categorized["boolean"].append(col)
        else:
            categorized["string"].append(col)

        # semantic heuristics
        lname = col.lower()
        unique_vals = series.unique() if not series.empty else []
        unique_count = len(unique_vals)

        if "id" in lname or unique_count == num_events:
            categorized["id_like"].append(col)

        if "time" in lname or "date" in lname or is_datetime:
            categorized["temporal"].append(col)

        if 1 < unique_count <= max(1, int(0.2 * num_events)):
            categorized["categorical"].append(col)

        if col not in sum([categorized[k] for k in ["numeric", "string", "datetime", "boolean"]], []):
            categorized["other"].append(col)

    return {
        "ea_names": ea_names,
        "ea_count": len(ea_names),
        "categorized": categorized,
    }


def extract_all_otps(ocel, verbose: bool = True, tmp_dir: str = None) -> Dict[str, Dict[str, Any]]:
    """Extract Object Type Perspectives for every object type in `ocel`.

    Returns a dict keyed by object-type. Each value is a dict with (at least)
    the following keys:
      - total_objects: int
      - total_events: int  -- NEW: total number of events in this OT perspective
      - OA_ot: set of object attribute names (best-effort)
      - OA_ot_count: int
      - OA_ot_categorized: dict (see _categorize_object_attributes)
      - EA_ot: set of event attribute names (best-effort)
      - EA_ot_count: int  -- NEW: total number of distinct event attributes
      - EA_ot_categorized: dict  -- NEW: categorized event attributes
      - ET_ot: set of event/activity types where this object-type participates
      - event_type_counts: dict {activity: count_of_events_involving_this_object_type}
      - OT_ot: set of related object-types (union of direct and co-occurrence)
      - direct_relations: dict {other_type: {relationship_count, unique_direct_pairs_count,
          direct_oids_count_per_side: (count_A, count_B)}}
      - coocc_relations: dict {other_type: {event_relation_cnt, coocc_oids_count_per_side: (count_A, count_B)}}
      - direct_only, coocc_only, both: sets for reporting

    The function will attempt to use o2o dataframe if present, but will always
    compute pairwise results using the JSON method so results are consistent
    with the user's preferred calculation.
    """
    # lazy imports
    import pandas as pd
    from collections import defaultdict

    events_df = ocel.events.copy()
    objects_df = ocel.objects.copy()
    relations_df = ocel.relations.copy()

    event_attrs_df = getattr(ocel, "event_attributes", None)
    object_attrs_df = getattr(ocel, "object_attributes", None)
    o2o_df = getattr(ocel, "o2o", None)

    # basic mappings
    types_all = sorted(objects_df["ocel:type"].unique())
    obj_id_to_type = dict(zip(objects_df["ocel:oid"], objects_df["ocel:type"]))
    eid_to_activity = dict(zip(events_df["ocel:eid"], events_df["ocel:activity"]))

    # build E2O mapping: {eid: [oid, ...]}
    E2O = defaultdict(list)
    for _, r in relations_df.iterrows():
        E2O[r["ocel:eid"]].append(r["ocel:oid"])

    # precompute pairwise O2O metrics (combinations of object types)
    pair_metrics = {}
    for a, b in itertools.combinations(types_all, 2):
        metrics = _compute_pair_o2o_metrics(ocel, a, b, tmp_dir=tmp_dir)
        pair_metrics[(a, b)] = metrics

    results = {}

    for ot in types_all:
        # object attributes
        if object_attrs_df is not None:
            try:
                OA_ot = set(object_attrs_df[object_attrs_df["ocel:type"] == ot]["ocel:attr:name"].unique())
            except Exception:
                OA_ot = set(objects_df[objects_df["ocel:type"] == ot].columns) - {"ocel:oid", "ocel:type"}
        else:
            OA_ot = set(objects_df[objects_df["ocel:type"] == ot].columns) - {"ocel:oid", "ocel:type"}

        # count and categorize object attributes using dataframe rows for this object type
        obj_ids_of_type = set(objects_df[objects_df["ocel:type"] == ot]["ocel:oid"].unique())
        oa_info = _categorize_object_attributes(objects_df, obj_ids_of_type)

        # events involving objects of this type
        related_eids = set(relations_df[relations_df["ocel:oid"].isin(obj_ids_of_type)]["ocel:eid"].unique())

        # ── NEW: total events in this perspective ──────────────────────────────
        total_events = len(related_eids)

        ET_ot = set(eid_to_activity[e] for e in related_eids if e in eid_to_activity)

        # event attributes (names, count, categorization)
        # We call the dedicated helper which mirrors _categorize_object_attributes
        ea_info = _categorize_event_attributes(events_df, event_attrs_df, related_eids, ET_ot)
        EA_ot = ea_info["ea_names"]

        # event type counts
        ev_sub = events_df[events_df["ocel:eid"].isin(related_eids)]
        if not ev_sub.empty:
            event_type_counts = ev_sub.groupby("ocel:activity").size().to_dict()
        else:
            event_type_counts = {}

        # O2O relations to other object types
        direct_relations = {}
        coocc_relations = {}

        for other in types_all:
            if other == ot:
                continue
            key = (ot, other) if (ot, other) in pair_metrics else (other, ot)
            metrics = pair_metrics.get(key)
            if metrics is None:
                # shouldn't happen, but skip defensively
                continue

            # map metrics to correct side: metrics were computed for (a,b)
            if key[0] == ot:
                direct_oids_here = metrics["direct_oids_A"]
                direct_oids_other = metrics["direct_oids_B"]
                coocc_oids_here = metrics["coocc_oids_A"]
                coocc_oids_other = metrics["coocc_oids_B"]
            else:
                direct_oids_here = metrics["direct_oids_B"]
                direct_oids_other = metrics["direct_oids_A"]
                coocc_oids_here = metrics["coocc_oids_B"]
                coocc_oids_other = metrics["coocc_oids_A"]

            direct_present = metrics["unique_direct_pairs_count"] > 0
            coocc_present = metrics["event_relation_cnt"] > 0

            direct_relations[other] = {
                "relationship_count_raw": metrics["relationship_count"],
                "unique_direct_pairs_count": metrics["unique_direct_pairs_count"],
                "direct_oids_count_this_type": len(direct_oids_here),
                "direct_oids_count_other_type": len(direct_oids_other),
                "direct_oids_this_type": direct_oids_here,
                "direct_oids_other_type": direct_oids_other,
            }

            coocc_relations[other] = {
                "event_relation_cnt": metrics["event_relation_cnt"],
                "coocc_oids_count_this_type": len(coocc_oids_here),
                "coocc_oids_count_other_type": len(coocc_oids_other),
                "coocc_oids_this_type": coocc_oids_here,
                "coocc_oids_other_type": coocc_oids_other,
            }

        # classify partner types into direct_only, coocc_only, both
        direct_set = {t for t, info in direct_relations.items() if info["unique_direct_pairs_count"] > 0}
        coocc_set = {t for t, info in coocc_relations.items() if info["event_relation_cnt"] > 0}
        both = direct_set & coocc_set
        direct_only = direct_set - coocc_set
        coocc_only = coocc_set - direct_set
        OT_ot = direct_set | coocc_set

        results[ot] = {
            "total_objects": len(obj_ids_of_type),
            "total_events": total_events,                    # ── NEW
            "OA_ot": OA_ot,
            "OA_ot_count": oa_info.get("oa_count", 0),
            "OA_ot_categorized": oa_info.get("categorized", {}),
            "EA_ot": EA_ot,
            "EA_ot_count": ea_info.get("ea_count", 0),      # ── NEW
            "EA_ot_categorized": ea_info.get("categorized", {}),  # ── NEW
            "ET_ot": ET_ot,
            "event_type_counts": event_type_counts,
            "OT_ot": OT_ot,
            "direct_relations": direct_relations,
            "coocc_relations": coocc_relations,
            "both": both,
            "direct_only": direct_only,
            "coocc_only": coocc_only,
            "related_eids": related_eids,
        }

        if verbose:
            print(f"=== Object Type: {ot} ===")
            print(f"1) Object Type name: {ot}")
            print(f"2) Total number of objects: {len(obj_ids_of_type)}")
            # ── NEW: total events printed right after total objects ────────────
            print(f"   Total number of events in perspective: {total_events}")
            print(f"3) Event types that have E2O relation (ET_ot): {len(ET_ot)} -> {sorted(list(ET_ot))}")
            print("4) Total number of events per event type in this perspective:")
            for et, cnt in sorted(event_type_counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"   - {et}: {cnt}")
            if not event_type_counts:
                print("   (no events found for this perspective)")

            print(f"5) O2O related object types (direct_only={len(direct_only)}, coocc_only={len(coocc_only)}, both={len(both)}):")
            if direct_only:
                print(f"   direct_only: {sorted(list(direct_only))}")
            if coocc_only:
                print(f"   coocc_only: {sorted(list(coocc_only))}")
            if both:
                print(f"   both: {sorted(list(both))}")
            if not OT_ot:
                print("   (no O2O relations detected)")

            print("6) Number of objects allocated with object type of O2O relation (per partner):")
            partners = sorted(set(list(direct_relations.keys()) + list(coocc_relations.keys())))
            for p in partners:
                dinfo = direct_relations.get(p, {})
                cinfo = coocc_relations.get(p, {})
                dcount = dinfo.get("direct_oids_count_this_type", 0)
                ccount = cinfo.get("coocc_oids_count_this_type", 0)
                print(f"   - Partner {p}: direct_objects={dcount}, cooccurrence_objects={ccount}")

            # ── Object attribute counts and categories ─────────────────────────
            print(f"Object attributes (OA_ot_count): {oa_info.get('oa_count', 0)}")
            oa_cat = oa_info.get("categorized", {})
            if oa_cat:
                print("   Object attribute categories summary:")
                for k in ["id_like", "temporal", "numeric", "categorical", "string", "boolean", "other"]:
                    vals = oa_cat.get(k, [])
                    if vals:
                        print(f"     - {k}: {len(vals)} -> {sorted(vals)[:10]}{'...' if len(vals) > 10 else ''}")

            # ── NEW: Event attribute counts and categories ─────────────────────
            print(f"Event attributes (EA_ot_count): {ea_info.get('ea_count', 0)}")
            ea_cat = ea_info.get("categorized", {})
            if ea_cat:
                print("   Event attribute categories summary:")
                for k in ["id_like", "temporal", "numeric", "categorical", "string", "boolean", "other"]:
                    vals = ea_cat.get(k, [])
                    if vals:
                        print(f"     - {k}: {len(vals)} -> {sorted(vals)[:10]}{'...' if len(vals) > 10 else ''}")

            print()  # blank line between object types

    return results


# If this module is run as a script, provide a tiny CLI example (non-intrusive):
if __name__ == "__main__":
    print("This module provides extract_all_otps(ocel). Import and call from your code.")
