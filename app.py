"""
Commute Time Map — Streamlit web app
Visualise travel-time isochrones (car, bicycle, walking, public transit) and
as-the-crow-flies distance circles.  Each zone has its own origin.

Zones are combined visually using a step-based pipeline builder:
  Step 1: Zone A  union    Zone B  →  Step 1 result
  Step 2: Step 1  cut      Zone C  →  Step 2 result
  ...
Any step's result can be used as the input to a later step, allowing
arbitrary set-algebra without writing any expressions.

API: Geoapify (https://www.geoapify.com/)
     Free tier: 3 000 requests / day, no credit card required.
     Sign up at https://myprojects.geoapify.com/ to get an API key.
"""

from __future__ import annotations

import os
import uuid

import folium
import pyproj
import requests
import streamlit as st
from dotenv import load_dotenv
from shapely.geometry import Point, mapping, shape
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

load_dotenv()

GEOAPIFY_KEY: str = os.getenv("GEOAPIFY_API_KEY", "")
ISOLINE_URL = "https://api.geoapify.com/v1/isoline"
GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"

TRANSPORT_MODES: dict[str, str] = {
    "Car": "drive",
    "Bicycle": "bicycle",
    "Walking": "walk",
    "Public Transit": "transit",
}

DEFAULT_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]

ZONE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

OP_LABELS: dict[str, str] = {
    "union":     "∪ union",
    "intersect": "∩ intersect",
    "cut":       "∖ cut (A − B)",
}

DEFAULT_LAT: float = 51.505
DEFAULT_LON: float = -0.09
MAP_ZOOM: int = 12

# Colours for the step-result overlay
RESULT_COLORS = [
    "#1a1a2e", "#16213e", "#0f3460", "#533483",
    "#e94560", "#2b2d42", "#8d99ae", "#ef233c",
]


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults: dict = {
        "zones": [],             # list of zone dicts
        "display_mode": "Show All",  # "Show All" | "Build Operation"
        # ── operation pipeline ──
        "op_steps": [],          # list of step dicts (see schema below)
        "op_next_id": 1,         # monotonic counter for stable step IDs
        "op_display_step": None, # step id to show on map, or None → last
        "op_color": "#1a1a2e",   # overlay colour for the chosen result
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Zone dict schema ─────────────────────────────────────────────────────────
# {
#   "id":             str          — uuid4
#   "zone_type":      str          — "Isochrone" | "Distance Circle"
#   "mode":           str          — "Car" | "Bicycle" | "Walking" | "Public Transit"
#   "minutes":        int          — ideal inner time limit (isochrone only)
#   "radius_km":      float        — ideal inner radius (distance circle only)
#   "color":          str          — hex colour
#   "lat":            float | None
#   "lon":            float | None
#   "search_text":    str
#   "geojson":        dict | None  — inner (ideal) boundary GeoJSON Feature
#   "error":          str | None
#   # Range (borderline outer zone) — optional
#   "has_range":      bool         — whether to show an outer "acceptable" boundary
#   "range_minutes":  int          — outer time limit (isochrone, > minutes)
#   "range_km":       float        — outer radius (distance circle, > radius_km)
#   "range_geojson":  dict | None  — outer boundary GeoJSON Feature
#   "range_error":    str | None
# }
#
# When has_range is True the rendered zone shows:
#   inner boundary  — solid fill (ideal area)
#   outer ring      — outer.difference(inner), dashed border + lighter fill (borderline)
#
# ── Step dict schema ─────────────────────────────────────────────────────────
# {
#   "id":    str  — stable "step_N" string, N from op_next_id counter
#   "left":  str | None  — zone label ("A") or step id ("step_2") or None
#   "op":    str  — "union" | "intersect" | "cut"
#   "right": str | None  — same as left
# }


# ---------------------------------------------------------------------------
# Geoapify API helpers
# ---------------------------------------------------------------------------

def geocode_location(text: str) -> tuple[float, float] | None:
    if not GEOAPIFY_KEY:
        st.error("GEOAPIFY_API_KEY is not set — geocoding unavailable.")
        return None
    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"text": text, "apiKey": GEOAPIFY_KEY, "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            return None
        lon, lat = features[0]["geometry"]["coordinates"]
        return float(lat), float(lon)
    except requests.RequestException as exc:
        st.error(f"Geocoding request failed: {exc}")
        return None


def fetch_isochrone(
    lat: float, lon: float, api_mode: str, minutes: int
) -> dict | None:
    if not GEOAPIFY_KEY:
        return None
    try:
        resp = requests.get(
            ISOLINE_URL,
            params={
                "lat": lat, "lon": lon,
                "type": "time", "mode": api_mode,
                "range": minutes * 60,
                "apiKey": GEOAPIFY_KEY,
            },
            timeout=30,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        return features[0] if features else None
    except requests.RequestException as exc:
        raise RuntimeError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def make_distance_circle(
    lat: float, lon: float, radius_km: float, resolution: int = 64
) -> dict:
    utm_zone = int((lon + 180) / 6) + 1
    utm_crs = pyproj.CRS.from_dict(
        {"proj": "utm", "zone": utm_zone, "south": lat < 0, "ellps": "WGS84"}
    )
    wgs84 = pyproj.CRS.from_epsg(4326)
    to_utm = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    from_utm = pyproj.Transformer.from_crs(utm_crs, wgs84, always_xy=True)
    x, y = to_utm.transform(lon, lat)
    circle = Point(x, y).buffer(radius_km * 1000, resolution=resolution)
    coords = [list(from_utm.transform(px, py)) for px, py in circle.exterior.coords]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {},
    }


def _to_shapely(feature: dict):
    geom = feature if feature.get("type") != "Feature" else feature["geometry"]
    return shape(geom)


def _to_geojson_feature(geom) -> dict:
    return {"type": "Feature", "geometry": mapping(geom), "properties": {}}


# ---------------------------------------------------------------------------
# Operation pipeline: evaluation
# ---------------------------------------------------------------------------

# Each "geometry value" in the pipeline is a (inner, outer) pair.
# Zones without a range use  outer = inner  so the ring is always empty.
# Operations are applied independently to inner and outer; the borderline
# ring is always  outer_result − inner_result  at render time.

def _apply_op_pair(op: str, left: tuple, right: tuple) -> tuple:
    """Apply a set operation to two (inner, outer) geometry pairs."""
    iA, oA = left
    iB, oB = right
    if op == "union":
        return (iA.union(iB), oA.union(oB))
    if op == "intersect":
        return (iA.intersection(iB), oA.intersection(oB))
    if op == "cut":
        return (iA.difference(iB), oA.difference(oB))
    raise ValueError(f"Unknown operation: {op!r}")


def evaluate_operations(
    zones: list[dict], steps: list[dict]
) -> dict[str, tuple]:
    """
    Evaluate all steps in order.
    Returns {step_id: ((inner_geom, outer_geom) | None, error_str | None)}.
    Each geometry value is a (inner, outer) pair; outer == inner when no range.
    """
    # Build (inner, outer) pairs for each computed zone
    zone_geom_pairs: dict[str, tuple] = {}
    for i, z in enumerate(zones):
        if i >= len(ZONE_LABELS) or not z.get("geojson"):
            continue
        try:
            inner = _to_shapely(z["geojson"])
            outer = (
                _to_shapely(z["range_geojson"])
                if z.get("has_range") and z.get("range_geojson")
                else inner
            )
            zone_geom_pairs[ZONE_LABELS[i]] = (inner, outer)
        except Exception:  # noqa: BLE001
            pass

    results: dict[str, tuple] = {}

    for step in steps:
        sid = step["id"]
        left_ref  = step.get("left")
        right_ref = step.get("right")

        if left_ref is None or right_ref is None:
            results[sid] = (None, "Both inputs must be selected.")
            continue

        left_pair  = _resolve_ref(left_ref,  zone_geom_pairs, results)
        right_pair = _resolve_ref(right_ref, zone_geom_pairs, results)

        if left_pair is None:
            results[sid] = (None, f"Left input ({_ref_label(left_ref)}) has no result.")
            continue
        if right_pair is None:
            results[sid] = (None, f"Right input ({_ref_label(right_ref)}) has no result.")
            continue

        try:
            op = step.get("op", "union")
            inner, outer = _apply_op_pair(op, left_pair, right_pair)

            if inner.is_empty and outer.is_empty:
                results[sid] = (None, "Result is empty.")
            else:
                results[sid] = ((inner, outer), None)
        except Exception as exc:  # noqa: BLE001
            results[sid] = (None, f"Geometry error: {exc}")

    return results


def _resolve_ref(ref: str, zone_geoms: dict, step_results: dict):
    """Resolve a left/right reference to a Shapely geometry, or None."""
    if ref in zone_geoms:
        return zone_geoms[ref]
    if ref in step_results:
        return step_results[ref][0]  # None if that step errored
    return None


def _ref_label(ref: str | None) -> str:
    """Human-readable label for a ref string."""
    if ref is None:
        return "(none)"
    if len(ref) == 1:
        return f"Zone {ref}"
    # "step_N"
    return f"Step {ref.split('_')[1]}"


# ---------------------------------------------------------------------------
# Operation pipeline: step management helpers
# ---------------------------------------------------------------------------

def _add_step() -> None:
    sid = f"step_{st.session_state.op_next_id}"
    st.session_state.op_next_id += 1
    st.session_state.op_steps.append(
        {"id": sid, "left": None, "op": "union", "right": None}
    )


def _remove_step(del_idx: int) -> None:
    steps = st.session_state.op_steps
    deleted_id = steps[del_idx]["id"]
    steps.pop(del_idx)

    # Nullify dangling references in later steps
    for s in steps:
        if s["left"] == deleted_id:
            s["left"] = None
        if s["right"] == deleted_id:
            s["right"] = None

    # Fix display selection
    if st.session_state.op_display_step == deleted_id:
        st.session_state.op_display_step = steps[-1]["id"] if steps else None


# ---------------------------------------------------------------------------
# Zone compute
# ---------------------------------------------------------------------------

def _compute_zone(zone: dict) -> None:
    lat = zone.get("lat")
    lon = zone.get("lon")
    if lat is None or lon is None:
        zone["error"] = "No location set — search for an address first."
        return
    zone["geojson"] = None
    zone["error"] = None
    zone["range_geojson"] = None
    zone["range_error"] = None

    if zone["zone_type"] == "Isochrone":
        if not GEOAPIFY_KEY:
            zone["error"] = "GEOAPIFY_API_KEY not set."
            return
        api_mode = TRANSPORT_MODES[zone["mode"]]
        # Inner (ideal) boundary
        try:
            feature = fetch_isochrone(lat, lon, api_mode, zone["minutes"])
            if feature is None:
                zone["error"] = "API returned no isochrone."
            else:
                zone["geojson"] = feature
        except RuntimeError as exc:
            zone["error"] = f"API error: {exc}"
        # Outer (acceptable) boundary
        if zone.get("has_range") and zone["geojson"] is not None:
            outer_min = zone.get("range_minutes", zone["minutes"] + 15)
            try:
                rf = fetch_isochrone(lat, lon, api_mode, outer_min)
                if rf is None:
                    zone["range_error"] = "API returned no outer isochrone."
                else:
                    zone["range_geojson"] = rf
            except RuntimeError as exc:
                zone["range_error"] = f"API error: {exc}"
    else:
        # Inner circle
        try:
            zone["geojson"] = make_distance_circle(lat, lon, zone["radius_km"])
        except Exception as exc:  # noqa: BLE001
            zone["error"] = f"Circle error: {exc}"
        # Outer circle
        if zone.get("has_range") and zone["geojson"] is not None:
            outer_km = zone.get("range_km", zone["radius_km"] * 1.5)
            try:
                zone["range_geojson"] = make_distance_circle(lat, lon, outer_km)
            except Exception as exc:  # noqa: BLE001
                zone["range_error"] = f"Circle error: {exc}"


# ---------------------------------------------------------------------------
# Map bounds
# ---------------------------------------------------------------------------

def _bounds_from_zones(zones: list[dict]) -> list | None:
    pts: list[tuple[float, float]] = []
    for z in zones:
        if z.get("lat") and z.get("lon"):
            pts.append((z["lat"], z["lon"]))
        for gj_key in ("geojson", "range_geojson"):
            if z.get(gj_key):
                try:
                    b = _to_shapely(z[gj_key]).bounds
                    pts += [(b[1], b[0]), (b[3], b[2])]
                except Exception:  # noqa: BLE001
                    pass
    if not pts:
        return None
    min_lat = min(p[0] for p in pts)
    max_lat = max(p[0] for p in pts)
    min_lon = min(p[1] for p in pts)
    max_lon = max(p[1] for p in pts)
    pad_lat = max((max_lat - min_lat) * 0.12, 0.02)
    pad_lon = max((max_lon - min_lon) * 0.12, 0.02)
    return [
        [min_lat - pad_lat, min_lon - pad_lon],
        [max_lat + pad_lat, max_lon + pad_lon],
    ]


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------

def build_map(render_items: list[dict], bounds: list | None) -> folium.Map:
    center = (
        [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
        if bounds
        else [DEFAULT_LAT, DEFAULT_LON]
    )
    m = folium.Map(location=center, zoom_start=MAP_ZOOM, tiles="CartoDB positron")

    for item in render_items:
        color = item["color"]
        feature = item["feature"]
        fill_opacity = item.get("fill_opacity", 0.25)
        weight = item.get("weight", 2)
        opacity = item.get("opacity", 0.8)
        label = item.get("label", "")

        if feature.get("type") != "Feature":
            feature = {"type": "Feature", "geometry": feature, "properties": {}}

        dash_array = item.get("dash_array")
        folium.GeoJson(
            feature,
            tooltip=label,
            style_function=lambda _f, c=color, fo=fill_opacity, w=weight, o=opacity, da=dash_array: {
                "fillColor": c, "color": c,
                "weight": w, "fillOpacity": fo, "opacity": o,
                **( {"dashArray": da} if da else {} ),
            },
        ).add_to(m)

        if "origin" in item:
            lat, lon = item["origin"]
            folium.CircleMarker(
                location=[lat, lon],
                radius=5, color=color,
                fill=True, fill_color=color, fill_opacity=0.9,
                tooltip=f"{label} origin" if label else "Origin",
            ).add_to(m)

    if bounds:
        m.fit_bounds(bounds)
    return m


# ---------------------------------------------------------------------------
# Render items builder
# ---------------------------------------------------------------------------

def _zone_render_items(z: dict, idx: int, ghost: bool = False) -> list[dict]:
    """
    Return 1–2 render items for a zone.
    If the zone has a computed range, emits:
      • outer ring  (dashed, lighter fill) — rendered first (bottom layer)
      • inner zone  (solid fill)           — rendered second (top layer)
    Otherwise emits just the inner zone.
    ghost=True reduces opacity for background display in Build-Operation mode.
    """
    if not z.get("geojson"):
        return []

    label = f"Zone {ZONE_LABELS[idx]}" if idx < len(ZONE_LABELS) else f"Zone {idx}"
    color = z["color"]
    origin = (z["lat"], z["lon"]) if z.get("lat") and z.get("lon") else None

    if ghost:
        inner_style = {"fill_opacity": 0.06, "weight": 1.0, "opacity": 0.30}
        ring_style  = {"fill_opacity": 0.03, "weight": 1.0, "opacity": 0.20}
    else:
        inner_style = {"fill_opacity": 0.28, "weight": 2,   "opacity": 0.85}
        ring_style  = {"fill_opacity": 0.10, "weight": 2,   "opacity": 0.60, "dash_array": "7 5"}

    items: list[dict] = []

    # Outer ring — rendered before inner so inner sits on top
    if z.get("has_range") and z.get("range_geojson"):
        try:
            inner_geom = _to_shapely(z["geojson"])
            outer_geom = _to_shapely(z["range_geojson"])
            ring_geom  = outer_geom.difference(inner_geom)
            if not ring_geom.is_empty:
                ring_item: dict = {
                    "color": color,
                    "feature": _to_geojson_feature(ring_geom),
                    "label": f"{label} (borderline)",
                    **ring_style,
                }
                if origin:
                    ring_item["origin"] = origin
                items.append(ring_item)
        except Exception:  # noqa: BLE001
            pass  # silently skip a bad ring rather than crashing

    # Inner (ideal) zone
    inner_item: dict = {
        "color": color,
        "feature": z["geojson"],
        "label": label,
        **inner_style,
    }
    if origin:
        inner_item["origin"] = origin
    items.append(inner_item)

    return items


def _get_render_items() -> list[dict]:
    zones = st.session_state.zones

    if st.session_state.display_mode == "Show All":
        items: list[dict] = []
        for i, z in enumerate(zones):
            items.extend(_zone_render_items(z, i, ghost=False))
        return items

    # ── Build Operation mode ──────────────────────────────────────────────
    ghost_items: list[dict] = []
    for i, z in enumerate(zones):
        ghost_items.extend(_zone_render_items(z, i, ghost=True))

    steps = st.session_state.op_steps
    if not steps:
        return ghost_items

    step_results = evaluate_operations(zones, steps)

    # Determine which step to display
    display_sid = st.session_state.op_display_step
    if display_sid is None or display_sid not in step_results:
        # Default: last step
        display_sid = steps[-1]["id"]

    geom_pair, error = step_results.get(display_sid, (None, "Step not evaluated."))
    if geom_pair is None:
        return ghost_items  # error shown in sidebar; map shows ghosts only

    step_pos = {s["id"]: i + 1 for i, s in enumerate(steps)}
    pos = step_pos.get(display_sid, "?")
    color = st.session_state.op_color
    inner_geom, outer_geom = geom_pair
    result_items: list[dict] = []

    # Outer (borderline) ring — rendered first so inner sits on top
    try:
        ring = outer_geom.difference(inner_geom) if not inner_geom.is_empty else outer_geom
        if not ring.is_empty:
            result_items.append({
                "color": color,
                "feature": _to_geojson_feature(ring),
                "fill_opacity": 0.12,
                "weight": 2.5,
                "opacity": 0.70,
                "dash_array": "7 5",
                "label": f"Step {pos} result (borderline)",
            })
    except Exception:  # noqa: BLE001
        pass

    # Inner (ideal) zone
    if not inner_geom.is_empty:
        result_items.append({
            "color": color,
            "feature": _to_geojson_feature(inner_geom),
            "fill_opacity": 0.45,
            "weight": 3,
            "opacity": 1.0,
            "label": f"Step {pos} result",
        })

    return ghost_items + result_items


# ---------------------------------------------------------------------------
# Sidebar: zone editor
# ---------------------------------------------------------------------------

def _render_zone_editor(idx: int, zone: dict) -> None:
    z_id = zone["id"]

    # Per-zone location search
    search = st.text_input(
        "Address",
        value=zone.get("search_text", ""),
        placeholder="e.g. Canary Wharf, London",
        key=f"search_{z_id}",
        label_visibility="collapsed",
    )
    if st.button("Search", key=f"btn_search_{z_id}", use_container_width=True):
        if search.strip():
            with st.spinner("Geocoding…"):
                result = geocode_location(search.strip())
            if result:
                zone["lat"], zone["lon"] = result
                zone["search_text"] = search.strip()
                zone["geojson"] = None
                zone["error"] = None
                st.rerun()
            else:
                st.warning("No results found.")
        else:
            st.warning("Enter an address first.")

    if zone.get("lat") is not None:
        st.caption(f"📍 {zone['lat']:.4f}, {zone['lon']:.4f}")
    else:
        st.caption("📍 No location set.")

    new_type = st.selectbox(
        "Type",
        ["Isochrone", "Distance Circle"],
        index=["Isochrone", "Distance Circle"].index(zone["zone_type"]),
        key=f"type_{z_id}",
    )
    if new_type != zone["zone_type"]:
        zone["zone_type"] = new_type
        zone["geojson"] = None
        zone["error"] = None

    if zone["zone_type"] == "Isochrone":
        new_mode = st.selectbox(
            "Transport mode",
            list(TRANSPORT_MODES.keys()),
            index=list(TRANSPORT_MODES.keys()).index(zone["mode"]),
            key=f"mode_{z_id}",
        )
        if new_mode != zone["mode"]:
            zone["mode"] = new_mode
            zone["geojson"] = None
            zone["error"] = None

        new_min = st.slider(
            "Travel time (minutes)", 5, 120, zone["minutes"], step=5,
            key=f"min_{z_id}",
        )
        if new_min != zone["minutes"]:
            zone["minutes"] = new_min
            zone["geojson"] = None
            zone["error"] = None
    else:
        new_radius = st.number_input(
            "Radius (km)", min_value=0.1, max_value=500.0,
            value=zone["radius_km"], step=0.5,
            key=f"radius_{z_id}", format="%.1f",
        )
        if float(new_radius) != zone["radius_km"]:
            zone["radius_km"] = float(new_radius)
            zone["geojson"] = None
            zone["error"] = None

    zone["color"] = st.color_picker("Colour", zone["color"], key=f"color_{z_id}")

    # ── Range (outer "acceptable" boundary) ──────────────────────────────
    new_has_range = st.checkbox(
        "Add outer 'acceptable' boundary",
        value=zone.get("has_range", False),
        key=f"has_range_{z_id}",
        help="Show a second, larger zone with a dashed border — e.g. 'ideally 20 min, up to 30 if necessary'.",
    )
    if new_has_range != zone.get("has_range", False):
        zone["has_range"] = new_has_range
        zone["range_geojson"] = None
        zone["range_error"] = None

    if zone.get("has_range"):
        if zone["zone_type"] == "Isochrone":
            inner_min = zone["minutes"]
            range_default = max(zone.get("range_minutes", inner_min + 15), inner_min + 5)
            new_range_min = st.slider(
                "Acceptable up to (minutes)",
                min_value=inner_min + 5,
                max_value=180,
                value=range_default,
                step=5,
                key=f"range_min_{z_id}",
            )
            if new_range_min != zone.get("range_minutes"):
                zone["range_minutes"] = new_range_min
                zone["range_geojson"] = None
                zone["range_error"] = None
        else:
            inner_km = zone["radius_km"]
            range_default = max(zone.get("range_km", round(inner_km * 1.5, 1)), inner_km + 0.5)
            new_range_km = st.number_input(
                "Acceptable up to (km)",
                min_value=inner_km + 0.1,
                max_value=1000.0,
                value=float(range_default),
                step=0.5,
                key=f"range_km_{z_id}",
                format="%.1f",
            )
            if float(new_range_km) != zone.get("range_km"):
                zone["range_km"] = float(new_range_km)
                zone["range_geojson"] = None
                zone["range_error"] = None

    # ── Compute / Remove ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Compute", key=f"compute_{z_id}", use_container_width=True):
            with st.spinner("Computing…"):
                _compute_zone(zone)
            st.rerun()
    with col2:
        if st.button("Remove", key=f"remove_{z_id}", use_container_width=True):
            st.session_state.zones = [z for z in st.session_state.zones if z["id"] != z_id]
            st.rerun()

    # ── Status ────────────────────────────────────────────────────────────
    inner_ok = zone.get("geojson") is not None
    range_ok = not zone.get("has_range") or zone.get("range_geojson") is not None

    if inner_ok and range_ok:
        st.success("Ready ✓", icon="✅")
    elif inner_ok and zone.get("has_range"):
        # Inner done but outer failed or not yet computed
        if zone.get("range_error"):
            st.warning(f"Inner ✓ · Outer failed: {zone['range_error']}")
        else:
            st.info("Inner computed — press Compute to also fetch the outer boundary.")
    elif zone.get("error"):
        st.error(zone["error"])
    else:
        st.caption("Not computed yet — press Compute.")


# ---------------------------------------------------------------------------
# Sidebar: operation pipeline builder
# ---------------------------------------------------------------------------

def _build_input_options(
    step_idx: int,
    zones: list[dict],
    steps: list[dict],
) -> tuple[list, dict]:
    """
    Return (option_values, label_map) for a left/right input selectbox
    at position step_idx (0-based).
    option_values[0] is always None → "(select input)".
    """
    vals: list = [None]
    labels: dict = {None: "(select input)"}

    for i, z in enumerate(zones):
        if z.get("geojson") and i < len(ZONE_LABELS):
            lbl = ZONE_LABELS[i]
            vals.append(lbl)
            labels[lbl] = f"Zone {lbl}"

    for j in range(step_idx):  # only steps that come before this one
        sid = steps[j]["id"]
        pos = j + 1
        vals.append(sid)
        labels[sid] = f"Step {pos} result"

    return vals, labels


def _render_operation_builder() -> None:
    zones = st.session_state.zones
    steps = st.session_state.op_steps

    # Pre-evaluate for status badges
    step_results = evaluate_operations(zones, steps) if steps else {}

    if not steps:
        st.caption("Press **+ Add Step** to start combining zones.")

    for step_idx, step in enumerate(steps):
        sid = step["id"]
        pos = step_idx + 1

        st.markdown(f"**Step {pos}**")

        opt_vals, opt_labels = _build_input_options(step_idx, zones, steps)

        # Clamp stored value to valid options (guards against dangling refs)
        left_val = step["left"] if step["left"] in opt_vals else None
        right_val = step["right"] if step["right"] in opt_vals else None

        col_l, col_op, col_r, col_del = st.columns([2.8, 2.0, 2.8, 0.7])

        with col_l:
            new_left = st.selectbox(
                "left",
                options=opt_vals,
                index=opt_vals.index(left_val),
                format_func=lambda v, m=opt_labels: m.get(v, str(v)),
                key=f"op_left_{sid}",
                label_visibility="collapsed",
            )
            step["left"] = new_left if new_left != "(select input)" else None

        with col_op:
            op_keys = list(OP_LABELS.keys())
            cur_op = step.get("op", "union")
            new_op = st.selectbox(
                "op",
                options=op_keys,
                index=op_keys.index(cur_op) if cur_op in op_keys else 0,
                format_func=lambda k: OP_LABELS[k],
                key=f"op_op_{sid}",
                label_visibility="collapsed",
            )
            step["op"] = new_op

        with col_r:
            new_right = st.selectbox(
                "right",
                options=opt_vals,
                index=opt_vals.index(right_val),
                format_func=lambda v, m=opt_labels: m.get(v, str(v)),
                key=f"op_right_{sid}",
                label_visibility="collapsed",
            )
            step["right"] = new_right if new_right != "(select input)" else None

        with col_del:
            if st.button("✕", key=f"op_del_{sid}", help="Remove this step"):
                _remove_step(step_idx)
                st.rerun()

        # Status badge
        if sid in step_results:
            geom_pair, err = step_results[sid]
            if geom_pair is not None:
                inner, outer = geom_pair
                has_ring = not outer.difference(inner).is_empty if not inner.is_empty else not outer.is_empty
                suffix = " + borderline ring" if has_ring else ""
                st.caption(f":green[✓ Step {pos} ready{suffix}]")
            else:
                st.caption(f":red[✗ {err}]")
        else:
            st.caption(":gray[⏸ not evaluated]")

        if step_idx < len(steps) - 1:
            st.divider()

    if st.button("＋ Add Step", key="btn_add_step", use_container_width=True):
        _add_step()
        st.rerun()

    # ── Display result selector ───────────────────────────────────────────
    if steps:
        st.divider()

        # Build list of all step IDs and their display names
        step_ids = [s["id"] for s in steps]
        step_pos_map = {s["id"]: i + 1 for i, s in enumerate(steps)}

        # Default display to last step
        current = st.session_state.op_display_step
        if current not in step_ids:
            current = step_ids[-1]
            st.session_state.op_display_step = current

        col_sel, col_col = st.columns([3, 1])
        with col_sel:
            chosen = st.selectbox(
                "Show on map",
                options=step_ids,
                index=step_ids.index(current),
                format_func=lambda sid, m=step_pos_map: f"Step {m[sid]} result",
                key="sel_display_step",
            )
            st.session_state.op_display_step = chosen

        with col_col:
            result_color = st.color_picker(
                "Colour",
                st.session_state.op_color,
                key="op_color_picker",
            )
            st.session_state.op_color = result_color

        # Show map-result status
        chosen_pair, chosen_err = step_results.get(chosen, (None, "not evaluated"))
        if chosen_pair is not None:
            st.success(f"Step {step_pos_map[chosen]} will be shown on the map.", icon="🗺️")
        elif chosen_err:
            st.error(f"Step {step_pos_map[chosen]} cannot be shown: {chosen_err}")


# ---------------------------------------------------------------------------
# Sidebar: full render
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    st.header("🗺️ Zones")

    for idx, zone in enumerate(st.session_state.zones):
        label = ZONE_LABELS[idx] if idx < len(ZONE_LABELS) else f"Z{idx}"
        dot = "🟢" if zone["geojson"] else ("🔴" if zone["error"] else "⚪")
        with st.expander(f"{dot} Zone {label}", expanded=True):
            _render_zone_editor(idx, zone)

    if st.button("＋ Add Zone", key="btn_add_zone", use_container_width=True):
        color_idx = len(st.session_state.zones) % len(DEFAULT_COLORS)
        st.session_state.zones.append({
            "id": str(uuid.uuid4()),
            "zone_type": "Isochrone",
            "mode": "Car",
            "minutes": 30,
            "radius_km": 5.0,
            "color": DEFAULT_COLORS[color_idx],
            "lat": None, "lon": None,
            "search_text": "",
            "geojson": None, "error": None,
            "has_range": False,
            "range_minutes": 45,
            "range_km": 8.0,
            "range_geojson": None, "range_error": None,
        })
        st.rerun()

    st.divider()

    # ── Combine zones ─────────────────────────────────────────────────────
    st.header("⚙️ Combine Zones")

    # Show which zones are available for operations
    ready = [
        ZONE_LABELS[i]
        for i, z in enumerate(st.session_state.zones)
        if z.get("geojson") and i < len(ZONE_LABELS)
    ]
    if ready:
        st.caption(f"Computed zones: **{', '.join(ready)}**")
    else:
        st.caption("No computed zones yet.")

    mode = st.radio(
        "Display mode",
        ["Show All", "Build Operation"],
        index=["Show All", "Build Operation"].index(st.session_state.display_mode),
        key="radio_display_mode",
        horizontal=True,
    )
    st.session_state.display_mode = mode

    if mode == "Build Operation":
        st.markdown(
            "Chain steps to combine zones. Each step's result can feed into the next.  \n"
            "**Operations:** ∪ union · ∩ intersect · ∖ cut (A minus B)"
        )
        _render_operation_builder()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Commute Time Map",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    if not GEOAPIFY_KEY:
        st.warning(
            "**No GEOAPIFY_API_KEY found.** Isochrones and address search require a key.  \n"
            "Distance circles work without one.  \n"
            "Get a free key at https://myprojects.geoapify.com/ and add it to `.env`:  \n"
            "`GEOAPIFY_API_KEY=your_key_here`",
            icon="⚠️",
        )

    with st.sidebar:
        _render_sidebar()

    st.title("🗺️ Commute Time Map")
    st.caption(
        "Each zone has its own origin and transport mode. "
        "Switch to **Build Operation** to chain steps — e.g. "
        "Step 1: A ∪ B, Step 2: Step 1 ∖ C."
    )

    render_items = _get_render_items()
    bounds = _bounds_from_zones(st.session_state.zones)
    m = build_map(render_items, bounds)
    st_folium(m, use_container_width=True, height=700, returned_objects=[])


if __name__ == "__main__":
    main()
