# ============================================================
# OTP-PD (Object Type Perspective Based Process Discovery)
# Combined: Lifecycle Mining + Visualization
#
# CORRECT HEURISTIC:
# For each object type OT, build the lifecycle order by:
# 1. Filter relations to only OT's objects.
# 2. For every OT object, get its personal event sequence
#    (sorted by timestamp).
# 3. Count how often activity A directly PRECEDES activity B
#    across all object traces → "follows" matrix.
# 4. Topological sort the follows matrix to get lifecycle order.
# 5. Break ties with median first-occurrence timestamp.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, Ellipse
from matplotlib.lines import Line2D
from collections import defaultdict, deque
from textwrap import fill


# ============================================================
# LIFECYCLE MINING
# ============================================================

def _mine_lifecycle_order(ot, ocel, otp_info):
    """
    Mine the correct lifecycle (activity) order for a single object type
    using a follows-graph approach on individual object traces.

    Steps:
      1. Get all objects of this type and their events (filtered to this OT).
      2. For each object, sort its events by timestamp → personal trace.
      3. Build a 'directly follows' count matrix across all traces.
      4. Topological sort the follows graph (Kahn's algorithm).
      5. Use median-first-occurrence timestamp as a tiebreaker.

    Returns: list of activity names in lifecycle order.
    """
    events_df   = ocel.events.copy()
    objects_df  = ocel.objects.copy()
    relations_df = ocel.relations.copy()

    ot_obj_ids   = set(objects_df[objects_df["ocel:type"] == ot]["ocel:oid"])
    ot_activities = set(otp_info["event_type_counts"].keys())
    if not ot_activities:
        return []

    eid_to_ts  = dict(zip(events_df["ocel:eid"], events_df["ocel:timestamp"]))
    eid_to_act = dict(zip(events_df["ocel:eid"], events_df["ocel:activity"]))

    ot_rels = relations_df[relations_df["ocel:oid"].isin(ot_obj_ids)]

    # Build per-object traces: {oid: [(timestamp, activity), ...]}
    obj_traces = defaultdict(list)
    for _, row in ot_rels.iterrows():
        oid = row["ocel:oid"]
        eid = row["ocel:eid"]
        ts  = eid_to_ts.get(eid)
        act = eid_to_act.get(eid)
        if ts is not None and act is not None and act in ot_activities:
            obj_traces[oid].append((ts, act))

    # Sort each object's trace by timestamp and deduplicate consecutive identical activities
    for oid in obj_traces:
        obj_traces[oid].sort(key=lambda x: x[0])
        deduped = []
        for ts, act in obj_traces[oid]:
            if not deduped or deduped[-1][1] != act:
                deduped.append((ts, act))
        obj_traces[oid] = deduped

    # Build directly-follows matrix
    follows = defaultdict(lambda: defaultdict(int))
    for oid, trace in obj_traces.items():
        for (ts1, a1), (ts2, a2) in zip(trace, trace[1:]):
            if a1 != a2:
                follows[a1][a2] += 1

    # Compute median first-occurrence timestamp per activity (tiebreaker)
    act_first_ts = defaultdict(list)
    for oid, trace in obj_traces.items():
        seen_acts = set()
        for ts, act in trace:
            if act not in seen_acts:
                act_first_ts[act].append(ts)
                seen_acts.add(act)

    act_median_ts = {}
    for act in ot_activities:
        ts_list = act_first_ts.get(act, [])
        if ts_list:
            try:
                numeric = pd.to_numeric(pd.Series(ts_list).values, errors='coerce')
                act_median_ts[act] = float(np.nanmedian(numeric))
            except Exception:
                act_median_ts[act] = 0.0
        else:
            act_median_ts[act] = 0.0

    # Topological sort (Kahn's algorithm)
    all_acts    = list(ot_activities)
    max_follows = max(
        (follows[a][b] for a in follows for b in follows[a]), default=1
    )
    threshold = max(1, int(max_follows * 0.01))  # 1% noise threshold

    graph  = defaultdict(set)
    in_deg = defaultdict(int)
    nodes  = set(all_acts)

    for a in follows:
        for b, cnt in follows[a].items():
            if cnt >= threshold and a != b:
                graph[a].add(b)
                in_deg[b] += 1

    for act in nodes:
        if act not in in_deg:
            in_deg[act] = 0

    queue = sorted(
        [act for act in nodes if in_deg[act] == 0],
        key=lambda a: act_median_ts.get(a, 0.0)
    )
    topo_order = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for neighbor in sorted(graph[node], key=lambda a: act_median_ts.get(a, 0.0)):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                inserted = False
                for idx, q in enumerate(queue):
                    if act_median_ts.get(neighbor, 0.0) < act_median_ts.get(q, 0.0):
                        queue.insert(idx, neighbor)
                        inserted = True
                        break
                if not inserted:
                    queue.append(neighbor)

    # Append any remaining cycle nodes sorted by median timestamp
    remaining = [a for a in nodes if a not in topo_order]
    remaining.sort(key=lambda a: act_median_ts.get(a, 0.0))
    topo_order.extend(remaining)
    return topo_order


def convert_otp_to_plotdata(otp_results, ocel):
    """
    Converts the output of extract_all_otps() into the plotting format
    required by plot_object(). Uses _mine_lifecycle_order() to discover
    the correct per-OT event sequence via a directly-follows graph and
    topological sort.
    """
    object_data = {}

    ot_lifecycle_orders = {}
    for ot, info in otp_results.items():
        ot_lifecycle_orders[ot] = _mine_lifecycle_order(ot, ocel, info)

    # Global event order = union of all OT orders (best-effort fallback)
    all_acts    = []
    seen_global = set()
    for order in ot_lifecycle_orders.values():
        for act in order:
            if act not in seen_global:
                all_acts.append(act)
                seen_global.add(act)
    global_event_order = all_acts

    for ot, info in otp_results.items():
        total_objects = info["total_objects"]
        event_counts  = info["event_type_counts"]

        lifecycle_order = ot_lifecycle_orders[ot]
        events_sorted   = [ev for ev in lifecycle_order if ev in event_counts]

        direct_only = sorted(list(info["direct_only"]))
        both        = sorted(list(info["both"]))
        shared_only = sorted(list(info["coocc_only"]))

        partner_stats = {}
        partners = sorted(
            set(info["direct_relations"].keys()) | set(info["coocc_relations"].keys())
        )
        for p in partners:
            d_count = info["direct_relations"].get(p, {}).get("direct_oids_count_this_type", 0)
            c_count = info["coocc_relations"].get(p, {}).get("coocc_oids_count_this_type", 0)
            partner_stats[p] = (d_count, c_count)

        shared_map = {}
        for p in both + shared_only:
            partner_event_counts = otp_results[p]["event_type_counts"].keys()
            shared_events = [ev for ev in events_sorted if ev in partner_event_counts]
            shared_map[p] = shared_events

        object_data[ot] = {
            "total_objects": total_objects,
            "events":        event_counts,
            "direct_only":   direct_only,
            "both":          both,
            "shared_only":   shared_only,
            "shared_map":    shared_map,
            "partner_stats": partner_stats,
            "_event_order":  events_sorted,
        }

    return object_data, global_event_order


# ============================================================
# VISUALIZATION
# ============================================================

def annotate_arrow(ax, x1, y1, x2, y2, text, color, rad=0.0, ratio=0.5):
    if not text:
        return
    dx, dy = x2 - x1, y2 - y1
    dist   = np.sqrt(dx**2 + dy**2)
    mx, my = x1 + dx * ratio, y1 + dy * ratio
    if dist > 0:
        px, py = -dy / dist, dx / dist
    else:
        px, py = 0, 0
    offset_magnitude = abs(rad) * dist * 0.5 * (4 * ratio * (1 - ratio))
    if rad > 0:
        offset_x, offset_y = -px * offset_magnitude, -py * offset_magnitude
    else:
        offset_x, offset_y =  px * offset_magnitude,  py * offset_magnitude
    ax.text(mx + offset_x, my + offset_y, text, color="black", fontsize=7,
            ha='center', va='center', weight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.2))


def compute_relation_cardinalities(relations):
    cardinality_map = {}
    for (src_type, qualifier, tgt_type), pairs in relations.items():
        fromA, fromB = defaultdict(set), defaultdict(set)
        for a, b in pairs:
            fromA[a].add(b)
            fromB[b].add(a)
        maxA = max(len(v) for v in fromA.values()) if fromA else 0
        maxB = max(len(v) for v in fromB.values()) if fromB else 0
        if   maxA <= 1 and maxB <= 1: forward, reverse = "1 : 1", "1 : 1"
        elif maxA >  1 and maxB <= 1: forward, reverse = "1 : *", "* : 1"
        elif maxA <= 1 and maxB >  1: forward, reverse = "* : 1", "1 : *"
        else:                         forward, reverse = "* : *", "* : *"
        cardinality_map[(src_type, tgt_type)] = forward
        cardinality_map[(tgt_type, src_type)] = reverse
    return cardinality_map


def generate_object_colors(object_types):
    try:
        cmap = matplotlib.colormaps["tab10"]
    except Exception:
        cmap = plt.get_cmap("tab10")
    return {ot: mcolors.to_hex(cmap(i % cmap.N)) for i, ot in enumerate(object_types)}


def wrapped_text(text, width=18):
    return fill(text.replace("-", "-\n"), width=width)


def _get_event_order(obj):
    """Return the mined per-OT lifecycle order; falls back to events dict keys."""
    return obj.get("_event_order") or list(obj["events"].keys())


def expand_shared_map(object_data):
    """
    Expands shared_map using each OT's own mined lifecycle order (_event_order)
    instead of the global_event_order.
    """
    expanded = {}
    for obj_name, obj in object_data.items():
        main_events = _get_event_order(obj)
        expanded[obj_name] = {}
        for rel, shared_events in obj.get("shared_map", {}).items():
            partner_event_set = set(object_data[rel]["events"].keys())
            shared_indices = [
                main_events.index(ev) for ev in shared_events if ev in main_events
            ]
            if not shared_indices:
                continue
            start_idx = min(shared_indices)
            result = []
            for ev in main_events[start_idx:]:
                if ev in partner_event_set:
                    result.append(ev)
            seen = set()
            expanded[obj_name][rel] = [e for e in result if not (e in seen or seen.add(e))]
    return expanded


def plot_object(object_type, data, relation_cardinalities=None):
    """
    Render the full OTP diagram for one object type.

    Parameters
    ----------
    object_type : str
    data : dict  — the object_data dict from convert_otp_to_plotdata()
    relation_cardinalities : dict, optional — from compute_relation_cardinalities()
    """
    rel_cards      = relation_cardinalities or {}
    object_types   = list(data.keys())
    object_colors  = generate_object_colors(object_types)
    expanded_shared = expand_shared_map(data)

    obj          = data[object_type]
    events       = _get_event_order(obj)
    event_counts = obj["events"]
    events       = [ev for ev in events if ev in event_counts]

    direct_only    = obj["direct_only"]
    both_relations = obj.get("both", [])
    shared_only    = obj.get("shared_only", [])
    shared_map     = expanded_shared[object_type]
    partner_stats  = obj["partner_stats"]
    total_objects  = obj["total_objects"]

    # Layout
    box_w, box_h = 0.16, 0.10
    gap     = 0.05
    start_x = 0.30
    event_xs        = [start_x + i * (box_w + gap) for i in range(len(events))]
    last_event_x    = event_xs[-1] if event_xs else start_x
    event_width_req = last_event_x + 0.6

    ellipse_base_width  = 0.16
    min_spacing         = 0.04
    top_objs            = direct_only
    bottom_objs         = both_relations + shared_only
    max_circles         = max(len(top_objs), len(bottom_objs)) if (top_objs or bottom_objs) else 0
    circle_row_needed_width = (
        max_circles * ellipse_base_width + max(0, max_circles - 1) * min_spacing
    )

    if max_circles > 0:
        center_of_event_flow = start_x + (last_event_x - start_x) / 2
        circle_row_start_x   = max(
            start_x - ellipse_base_width / 2,
            center_of_event_flow - circle_row_needed_width / 2
        )
    else:
        circle_row_start_x = start_x

    circle_width_req = circle_row_start_x + circle_row_needed_width + 0.2
    plot_xlim = max(2.0, event_width_req, circle_width_req)
    fig_width = 20 * (plot_xlim / 2.0)

    fig, ax = plt.subplots(figsize=(fig_width, 9))
    ax.set_xlim(0, plot_xlim)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect('equal', adjustable='box')

    main_x, main_y      = 0.10, 0.55
    event_y              = 0.55
    rel_top_y            = 0.85
    rel_bottom_y         = 0.25
    main_ellipse_width   = 0.13
    main_ellipse_height  = 0.10
    main_color = object_colors[object_type]

    # Main object ellipse
    main_patch = Ellipse((main_x, main_y), main_ellipse_width, main_ellipse_height,
                         color=main_color, ec="black", lw=2)
    ax.add_patch(main_patch)
    ax.text(main_x, main_y, wrapped_text(object_type, width=8),
            fontsize=10, weight="bold", ha="center", va="center")
    ax.text(main_x, main_y - 0.085, f"Total: {total_objects}", ha="center", fontsize=10)

    # Event boxes
    centers, event_patches = {}, {}
    for x, ev in zip(event_xs, events):
        bp = Rectangle((x - box_w / 2, event_y - box_h / 2), box_w, box_h,
                       fc="#B0E0E6", ec="black")
        ax.add_patch(bp)
        event_patches[ev] = bp
        ax.text(x, event_y, wrapped_text(ev, width=10),
                ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x, event_y + 0.065, f"{event_counts[ev]}", ha="center", fontsize=9)
        centers[ev] = (x, event_y)

    # Lifecycle arrows
    if events:
        first = events[0]
        ax.annotate("", xy=centers[first], xytext=(main_x, main_y),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=main_color,
                                   patchA=main_patch, patchB=event_patches[first],
                                   shrinkA=0, shrinkB=0))
        for ev1, ev2 in zip(events, events[1:]):
            ax.annotate("", xy=centers[ev2], xytext=centers[ev1],
                        arrowprops=dict(arrowstyle="-|>", lw=2, color=main_color,
                                       patchA=event_patches[ev1], patchB=event_patches[ev2],
                                       shrinkA=0, shrinkB=0))

    # End-of-lifecycle marker (open circle + filled square)
    if events:
        last  = events[-1]
        end_x = centers[last][0] + 0.18
        ax.add_patch(Circle((end_x, event_y), 0.045, ec=main_color, lw=3, fill=False))
        sq = 0.045 * 1.2
        ax.add_patch(Rectangle((end_x - sq / 2, event_y - sq / 2), sq, sq,
                                fc=main_color, ec=main_color, lw=2))
        ax.annotate("", xy=(end_x - 0.045 + 0.003, event_y),
                    xytext=(centers[last][0] + box_w / 2, event_y),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color=main_color,
                                   shrinkA=5, shrinkB=5))

    def ellipse_xs(n):
        if n == 0:
            return []
        return [circle_row_start_x + i * (ellipse_base_width + min_spacing) for i in range(n)]

    def arrow_offset(idx, total, span=0.08):
        if total <= 1:
            return 0
        return -(span / 2) + idx * span / (total - 1)

    # Top row: direct-only partners
    if top_objs:
        for i, (x, rel) in enumerate(zip(ellipse_xs(len(top_objs)), top_objs)):
            d, c = partner_stats.get(rel, (0, 0))
            rc   = object_colors.get(rel, "#FFD8A8")
            rp   = Ellipse((x, rel_top_y), 0.12, 0.09, color=rc, ec="black")
            ax.add_patch(rp)
            ax.text(x, rel_top_y, wrapped_text(rel, width=8),
                    ha="center", va="center", fontsize=9, weight="bold")
            ax.text(x, rel_top_y - 0.065, f"d={d}, c={c}", ha="center", fontsize=8)
            tx   = main_x + arrow_offset(i, len(top_objs))
            ty   = main_y + main_ellipse_height / 2
            curv = 0.3 + 0.15 * i
            ax.annotate("", xy=(tx, ty), xytext=(x, rel_top_y),
                        arrowprops=dict(arrowstyle="-", lw=1.5, color=rc, alpha=0.5,
                                       patchA=rp, shrinkA=0, shrinkB=0,
                                       connectionstyle=f"arc3,rad={curv}"))
            annotate_arrow(ax, x, rel_top_y, tx, ty,
                           rel_cards.get((object_type, rel), ""), rc, rad=curv)

    # Bottom row: both + shared-only partners
    relation_order_cache = {}
    if bottom_objs:
        for i, (x, rel) in enumerate(zip(ellipse_xs(len(bottom_objs)), bottom_objs)):
            d, c  = partner_stats.get(rel, (0, 0))
            rc    = object_colors.get(rel, "#FFD8A8")
            rp    = Ellipse((x, rel_bottom_y), 0.12, 0.09, color=rc, ec="black")
            ax.add_patch(rp)
            ax.text(x, rel_bottom_y, wrapped_text(rel, width=8),
                    ha="center", va="center", fontsize=9, weight="bold")
            ax.text(x, rel_bottom_y - 0.065, f"d={d}, c={c}", ha="center", fontsize=8)
            is_labeled = False

            # Direct connection line
            if rel in both_relations:
                tx   = main_x + arrow_offset(i, len(bottom_objs))
                ty   = main_y - main_ellipse_height / 2
                curv = -0.3 - 0.1 * i
                ax.annotate("", xy=(tx, ty), xytext=(x, rel_bottom_y),
                            arrowprops=dict(arrowstyle="-", lw=1.5, color=rc, alpha=0.5,
                                           patchA=rp, shrinkA=0, shrinkB=0,
                                           connectionstyle=f"arc3,rad={curv}"))
                annotate_arrow(ax, x, rel_bottom_y, tx, ty,
                               rel_cards.get((object_type, rel), ""), rc, rad=curv)
                is_labeled = True

            # Shared event lane (dashed)
            if rel in shared_map:
                shared_evs = [ev for ev in shared_map[rel] if ev in centers]
                if not shared_evs:
                    continue
                dim      = 0.5
                lane_off = arrow_offset(i, len(bottom_objs), span=0.10)
                fx, fy   = centers[shared_evs[0]]
                tx       = fx + lane_off
                ty       = fy - box_h / 2
                dx_      = fx - x
                curv     = (0.2 + abs(dx_) * 0.05) * (1 if dx_ >= 0 else -1)
                ax.annotate("", xy=(tx, ty), xytext=(x, rel_bottom_y),
                            arrowprops=dict(arrowstyle="-|>", lw=2, linestyle="--",
                                           color=rc, alpha=dim, patchA=rp,
                                           shrinkA=0, shrinkB=0,
                                           connectionstyle=f"arc3,rad={curv}"))
                if not is_labeled:
                    annotate_arrow(ax, x, rel_bottom_y, tx, ty,
                                   rel_cards.get((object_type, rel), ""), rc, rad=curv)
                    is_labeled = True

                for ev_a, ev_b in zip(shared_evs, shared_evs[1:]):
                    sx = centers[ev_a][0] + lane_off
                    sy = centers[ev_a][1] + box_h / 2
                    ex = centers[ev_b][0] + lane_off
                    ey = centers[ev_b][1] + box_h / 2
                    gap_steps = abs(events.index(ev_b) - events.index(ev_a))
                    base_r    = -0.5 - 0.1 * i
                    r = base_r / (gap_steps ** 0.4) if gap_steps > 2 else base_r
                    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                                arrowprops=dict(arrowstyle="-|>", lw=2, linestyle="--",
                                               color=rc, alpha=dim,
                                               shrinkA=0, shrinkB=0,
                                               connectionstyle=f"arc3,rad={r}"))

                # Fan-out end marker
                if object_type not in relation_order_cache:
                    rel_order = sorted(
                        [(r, events.index(shared_map[r][-1]))
                         for r in shared_map
                         if shared_map[r] and shared_map[r][-1] in events],
                        key=lambda x: x[1]
                    )
                    relation_order_cache[object_type] = {
                        r: idx for idx, (r, _) in enumerate(rel_order)
                    }
                rank   = relation_order_cache[object_type].get(rel, 0)
                lx, ly = centers[shared_evs[-1]]
                theta  = np.deg2rad(30 + rank * 20)
                fan_x  = lx + 0.35 * np.cos(theta)
                fan_y  = ly + 0.35 * np.sin(theta)
                ax.add_patch(Circle((fan_x, fan_y), 0.025, ec=rc, lw=2, fill=False, alpha=dim))
                sq2 = 0.016
                ax.add_patch(Rectangle((fan_x - sq2, fan_y - sq2), sq2 * 2, sq2 * 2,
                                       fc=rc, ec=rc, lw=1.4, alpha=dim))
                ax.annotate("", xy=(fan_x - 0.025, fan_y),
                            xytext=(lx + lane_off, ly + box_h / 2),
                            arrowprops=dict(arrowstyle="-|>", lw=2, linestyle="--",
                                           color=rc, alpha=dim,
                                           shrinkA=0, shrinkB=0))

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#B0E0E6", edgecolor="black", label="Event type"),
        Ellipse((0, 0), 0.13, 0.10, facecolor=main_color, edgecolor="black",
                label=f"Main: {object_type}"),
    ]
    for on, clr in object_colors.items():
        legend_items.append(
            Ellipse((0, 0), 0.12, 0.09, facecolor=clr, edgecolor="black", label=on)
        )
    legend_items += [
        Line2D([0], [0], color="black", lw=2, label="Direct"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Shared"),
    ]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(0.0, 1.15),
              ncol=3, fontsize=9, frameon=False)
    plt.title(f"Selected OTP: {object_type}", fontsize=18, weight="bold", pad=100, loc="left")
    plt.tight_layout()
    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

def run_otp_pd(otp_results, ocel, relations=None):
    """
    Full pipeline: mine lifecycle orders, convert to plot data, render diagrams.

    Parameters
    ----------
    otp_results : dict       — output of extract_all_otps()
    ocel        : OCEL object — loaded OCEL log
    relations   : dict, optional — for cardinality labels on arcs
    """
    object_data, global_event_order = convert_otp_to_plotdata(otp_results, ocel)

    rel_cards = compute_relation_cardinalities(relations) if relations else {}

    for obj_type in object_data.keys():
        plot_object(obj_type, object_data, relation_cardinalities=rel_cards)

    return object_data, global_event_order


# ── Quick run when executed as a notebook cell ────────────────────────────────
if "otp_results" in dir() and "ocel" in dir():
    _relations = globals().get("relations", None)
    run_otp_pd(otp_results, ocel, relations=_relations)
