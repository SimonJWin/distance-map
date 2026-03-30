"""
Commute Time Map — Streamlit web app
Visualise travel-time isochrones (car, bicycle, walking, public transit) and
as-the-crow-flies distance circles from a chosen origin.  Each zone has its
own origin.  Zones can be combined with arbitrary set expressions such as
  (A union B) cut C
using union / intersect / cut operators.

API: Geoapify (https://www.geoapify.com/)
     Free tier: 3,000 requests / day, no credit card required.
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

# Visually distinct palette
DEFAULT_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # grey
]

# Zones are labelled A, B, C, … in the UI and in expressions
ZONE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_LAT: float = 51.505
DEFAULT_LON: float = -0.09
MAP_ZOOM: int = 12


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults: dict = {
        "zones": [],           # list of zone dicts (see schema below)
        "display_mode": "Show All",  # "Show All" | "Expression"
        "expr_text": "",       # raw expression string
        "expr_geom": None,     # evaluated Shapely geometry (or None)
        "expr_color": "#e41a1c",
        "expr_error": None,    # error message from last evaluation
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Zone dict schema:
# {
#   "id":          str          — uuid4
#   "zone_type":   str          — "Isochrone" | "Distance Circle"
#   "mode":        str          — "Car" | "Bicycle" | "Walking" | "Public Transit"
#   "minutes":     int          — travel time (isochrone only)
#   "radius_km":   float        — radius (distance circle only)
#   "color":       str          — hex colour
#   "lat":         float | None — origin latitude
#   "lon":         float | None — origin longitude
#   "search_text": str          — last searched string
#   "geojson":     dict | None  — GeoJSON Feature
#   "error":       str | None   — last error message
# }


# ---------------------------------------------------------------------------
# Geoapify API helpers
# ---------------------------------------------------------------------------

def geocode_location(text: str) -> tuple[float, float] | None:
    """Return (lat, lon) for the first geocoding result, or None."""
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
    """
    Call the Geoapify Isoline API and return the first GeoJSON Feature, or raise
    RuntimeError on HTTP/network failure.
    """
    if not GEOAPIFY_KEY:
        return None
    try:
        resp = requests.get(
            ISOLINE_URL,
            params={
                "lat": lat,
                "lon": lon,
                "type": "time",
                "mode": api_mode,
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
    """
    Return a GeoJSON Feature (Polygon) representing a geodesic circle.
    Uses local UTM projection for accuracy.
    """
    utm_zone = int((lon + 180) / 6) + 1
    utm_crs = pyproj.CRS.from_dict(
        {"proj": "utm", "zone": utm_zone, "south": lat < 0, "ellps": "WGS84"}
    )
    wgs84 = pyproj.CRS.from_epsg(4326)
    to_utm = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    from_utm = pyproj.Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    x, y = to_utm.transform(lon, lat)
    circle_utm = Point(x, y).buffer(radius_km * 1000, resolution=resolution)
    coords = [list(from_utm.transform(px, py)) for px, py in circle_utm.exterior.coords]

    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {},
    }


def _to_shapely(feature: dict):
    """Convert a GeoJSON Feature or bare Geometry to a Shapely geometry."""
    geom = feature if feature.get("type") != "Feature" else feature["geometry"]
    return shape(geom)


def _to_geojson_feature(geom) -> dict:
    """Convert a Shapely geometry to a minimal GeoJSON Feature dict."""
    return {"type": "Feature", "geometry": mapping(geom), "properties": {}}


# ---------------------------------------------------------------------------
# Expression parser
#
# Grammar (all binary operators are left-associative, equal precedence;
# use parentheses to control evaluation order):
#
#   expr    := primary (OP primary)*
#   primary := ZONE | '(' expr ')'
#   OP      := 'union' | '|' | 'intersect' | '&' | 'cut' | '-' | 'diff'
#   ZONE    := single letter A–Z (case-insensitive)
# ---------------------------------------------------------------------------

def _tokenize(expr: str) -> list[tuple]:
    """Convert an expression string to a flat list of typed tokens."""
    tokens: list[tuple] = []
    i = 0
    keywords = {
        "union": "UNION",
        "intersect": "INTERSECT",
        "intersection": "INTERSECT",
        "cut": "DIFF",
        "diff": "DIFF",
        "difference": "DIFF",
        "minus": "DIFF",
    }
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(("LPAREN",))
            i += 1
            continue
        if ch == ")":
            tokens.append(("RPAREN",))
            i += 1
            continue
        if ch == "|":
            tokens.append(("UNION",))
            i += 1
            continue
        if ch == "&":
            tokens.append(("INTERSECT",))
            i += 1
            continue
        if ch == "-":
            tokens.append(("DIFF",))
            i += 1
            continue
        if ch.isalpha():
            # Greedily read the full word, then classify
            j = i
            while j < len(expr) and expr[j].isalpha():
                j += 1
            word = expr[i:j].lower()
            if word in keywords:
                tokens.append((keywords[word],))
            elif len(word) == 1:
                tokens.append(("ZONE", word.upper()))
            else:
                # Multi-letter word that isn't a keyword is an unknown zone name
                raise ValueError(
                    f"Unknown keyword or zone name: {expr[i:j]!r}. "
                    "Zone names must be single letters (A, B, C, …). "
                    f"Operators: union, intersect, cut."
                )
            i = j
            continue
        raise ValueError(f"Unexpected character: {ch!r} at position {i}")
    return tokens


class _Parser:
    """Recursive-descent parser that evaluates a zone expression directly."""

    def __init__(self, tokens: list[tuple], zone_geoms: dict):
        self.tokens = tokens
        self.pos = 0
        self.zone_geoms = zone_geoms

    def _peek(self) -> str | None:
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def _consume(self) -> tuple:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse_expr(self):
        """Parse: primary (OP primary)*  — left-associative."""
        left = self._parse_primary()
        while self._peek() in ("UNION", "INTERSECT", "DIFF"):
            op = self._consume()[0]
            right = self._parse_primary()
            if op == "UNION":
                left = left.union(right)
            elif op == "INTERSECT":
                left = left.intersection(right)
            elif op == "DIFF":
                left = left.difference(right)
        return left

    def _parse_primary(self):
        tok_type = self._peek()
        if tok_type is None:
            raise ValueError("Unexpected end of expression — expected a zone name or '('.")
        if tok_type == "LPAREN":
            self._consume()
            result = self.parse_expr()
            if self._peek() != "RPAREN":
                raise ValueError("Missing closing ')'.")
            self._consume()
            return result
        if tok_type == "ZONE":
            label = self._consume()[1]
            if label not in self.zone_geoms:
                available = ", ".join(sorted(self.zone_geoms)) or "none"
                raise ValueError(
                    f"Zone '{label}' has no computed geometry. "
                    f"Computed zones: {available}."
                )
            return self.zone_geoms[label]
        raise ValueError(
            f"Expected a zone name or '(' but got token type: {tok_type!r}."
        )


def evaluate_expression(expr_str: str, zone_geoms: dict):
    """
    Parse and evaluate a boolean zone expression.
    zone_geoms: {label: shapely_geometry}
    Returns a Shapely geometry, or raises ValueError.
    """
    tokens = _tokenize(expr_str.strip())
    if not tokens:
        raise ValueError("Expression is empty.")
    parser = _Parser(tokens, zone_geoms)
    result = parser.parse_expr()
    if parser.pos < len(tokens):
        leftover = tokens[parser.pos:]
        raise ValueError(
            f"Unexpected extra tokens after expression: "
            f"{' '.join(t[0] for t in leftover)}"
        )
    return result


# ---------------------------------------------------------------------------
# Zone compute
# ---------------------------------------------------------------------------

def _compute_zone(zone: dict) -> None:
    """Fetch / compute the zone geometry; mutates zone in place."""
    lat = zone.get("lat")
    lon = zone.get("lon")
    if lat is None or lon is None:
        zone["error"] = "No location set — search for an address first."
        return

    zone["geojson"] = None
    zone["error"] = None

    if zone["zone_type"] == "Isochrone":
        if not GEOAPIFY_KEY:
            zone["error"] = "GEOAPIFY_API_KEY not set."
            return
        api_mode = TRANSPORT_MODES[zone["mode"]]
        try:
            feature = fetch_isochrone(lat, lon, api_mode, zone["minutes"])
            if feature is None:
                zone["error"] = "API returned no isochrone for this location/mode."
            else:
                zone["geojson"] = feature
        except RuntimeError as exc:
            zone["error"] = f"API error: {exc}"
    else:
        try:
            zone["geojson"] = make_distance_circle(lat, lon, zone["radius_km"])
        except Exception as exc:  # noqa: BLE001
            zone["error"] = f"Circle computation failed: {exc}"


# ---------------------------------------------------------------------------
# Map bounds helpers
# ---------------------------------------------------------------------------

def _bounds_from_zones(zones: list[dict]) -> list | None:
    """Return [[min_lat, min_lon], [max_lat, max_lon]] covering all zones, or None."""
    pts: list[tuple[float, float]] = []
    for z in zones:
        if z.get("lat") and z.get("lon"):
            pts.append((z["lat"], z["lon"]))
        if z.get("geojson"):
            try:
                b = _to_shapely(z["geojson"]).bounds  # (minx, miny, maxx, maxy)
                pts.append((b[1], b[0]))
                pts.append((b[3], b[2]))
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
    """
    render_items: list of dicts with keys:
        color, feature (GeoJSON Feature or bare Geometry), fill_opacity,
        weight, opacity, label (optional), origin (optional lat/lon tuple)
    """
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

        # Normalise to GeoJSON Feature
        if feature.get("type") != "Feature":
            feature = {"type": "Feature", "geometry": feature, "properties": {}}

        folium.GeoJson(
            feature,
            tooltip=label,
            style_function=lambda _f, c=color, fo=fill_opacity, w=weight, o=opacity: {
                "fillColor": c,
                "color": c,
                "weight": w,
                "fillOpacity": fo,
                "opacity": o,
            },
        ).add_to(m)

        # Small dot at the zone's origin
        if "origin" in item:
            lat, lon = item["origin"]
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=f"{label} origin" if label else "Origin",
            ).add_to(m)

    if bounds:
        m.fit_bounds(bounds)

    return m


# ---------------------------------------------------------------------------
# Render items builder
# ---------------------------------------------------------------------------

def _get_render_items() -> list[dict]:
    zones = st.session_state.zones

    def _zone_item(z: dict, idx: int, **overrides) -> dict:
        label = f"Zone {ZONE_LABELS[idx]}" if idx < len(ZONE_LABELS) else f"Zone {idx}"
        item: dict = {
            "color": z["color"],
            "feature": z["geojson"],
            "fill_opacity": 0.25,
            "weight": 2,
            "opacity": 0.8,
            "label": label,
        }
        if z.get("lat") and z.get("lon"):
            item["origin"] = (z["lat"], z["lon"])
        item.update(overrides)
        return item

    if st.session_state.display_mode == "Show All":
        return [
            _zone_item(z, i)
            for i, z in enumerate(zones)
            if z.get("geojson")
        ]

    # Expression mode
    ghost_items = [
        _zone_item(z, i, fill_opacity=0.06, weight=1.0, opacity=0.35)
        for i, z in enumerate(zones)
        if z.get("geojson")
    ]

    if st.session_state.expr_geom is None:
        return ghost_items  # no result yet: show ghosts only

    result_item: dict = {
        "color": st.session_state.expr_color,
        "feature": _to_geojson_feature(st.session_state.expr_geom),
        "fill_opacity": 0.45,
        "weight": 3,
        "opacity": 1.0,
        "label": "Expression result",
    }
    return ghost_items + [result_item]


# ---------------------------------------------------------------------------
# Expression evaluation (sidebar action)
# ---------------------------------------------------------------------------

def _evaluate_expression() -> None:
    """Evaluate the current expression text; update session state in place."""
    expr_text = st.session_state.expr_text.strip()
    if not expr_text:
        st.session_state.expr_error = "Enter an expression first."
        st.session_state.expr_geom = None
        return

    # Build zone geometry map from computed zones
    zone_geoms: dict = {}
    for i, z in enumerate(st.session_state.zones):
        if z.get("geojson") is not None and i < len(ZONE_LABELS):
            label = ZONE_LABELS[i]
            try:
                zone_geoms[label] = _to_shapely(z["geojson"])
            except Exception as exc:  # noqa: BLE001
                st.session_state.expr_error = (
                    f"Zone {label} has invalid geometry: {exc}"
                )
                st.session_state.expr_geom = None
                return

    try:
        result = evaluate_expression(expr_text, zone_geoms)
    except ValueError as exc:
        st.session_state.expr_error = str(exc)
        st.session_state.expr_geom = None
        return
    except Exception as exc:  # noqa: BLE001
        st.session_state.expr_error = f"Geometry operation failed: {exc}"
        st.session_state.expr_geom = None
        return

    if result.is_empty:
        st.session_state.expr_error = (
            "Result is empty — zones may not overlap, or the cut removed everything."
        )
        st.session_state.expr_geom = None
    else:
        st.session_state.expr_geom = result
        st.session_state.expr_error = None


# ---------------------------------------------------------------------------
# Sidebar: zone editor
# ---------------------------------------------------------------------------

def _render_zone_editor(idx: int, zone: dict) -> None:
    """Render all editing widgets for a single zone (inside an expander)."""
    z_id = zone["id"]
    label = ZONE_LABELS[idx] if idx < len(ZONE_LABELS) else f"Z{idx}"

    # ── Per-zone location search ──────────────────────────────────────────
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        search = st.text_input(
            "Address / place",
            value=zone.get("search_text", ""),
            placeholder="e.g. Canary Wharf, London",
            key=f"search_{z_id}",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("Search", key=f"btn_search_{z_id}", use_container_width=True):
            if search.strip():
                with st.spinner("Geocoding…"):
                    result = geocode_location(search.strip())
                if result:
                    zone["lat"], zone["lon"] = result
                    zone["search_text"] = search.strip()
                    zone["geojson"] = None
                    zone["error"] = None
                    # Expression result may be stale now
                    st.session_state.expr_geom = None
                    st.rerun()
                else:
                    st.warning("No results found.")
            else:
                st.warning("Enter an address first.")

    if zone.get("lat") is not None:
        st.caption(f"📍 {zone['lat']:.4f}, {zone['lon']:.4f}")
    else:
        st.caption("📍 No location set — search above.")

    # ── Zone type & parameters ────────────────────────────────────────────
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
        st.session_state.expr_geom = None

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
            st.session_state.expr_geom = None

        new_min = st.slider(
            "Travel time (minutes)",
            min_value=5,
            max_value=120,
            value=zone["minutes"],
            step=5,
            key=f"min_{z_id}",
        )
        if new_min != zone["minutes"]:
            zone["minutes"] = new_min
            zone["geojson"] = None
            zone["error"] = None
            st.session_state.expr_geom = None
    else:
        new_radius = st.number_input(
            "Radius (km)",
            min_value=0.1,
            max_value=500.0,
            value=zone["radius_km"],
            step=0.5,
            key=f"radius_{z_id}",
            format="%.1f",
        )
        if float(new_radius) != zone["radius_km"]:
            zone["radius_km"] = float(new_radius)
            zone["geojson"] = None
            zone["error"] = None
            st.session_state.expr_geom = None

    zone["color"] = st.color_picker("Colour", zone["color"], key=f"color_{z_id}")

    # ── Compute / Remove ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Compute", key=f"compute_{z_id}", use_container_width=True):
            with st.spinner("Computing…"):
                _compute_zone(zone)
            st.session_state.expr_geom = None  # invalidate expression result
            st.rerun()
    with col2:
        if st.button("Remove", key=f"remove_{z_id}", use_container_width=True):
            st.session_state.zones = [z for z in st.session_state.zones if z["id"] != z_id]
            st.session_state.expr_geom = None
            st.rerun()

    # Status
    if zone["geojson"] is not None:
        st.success("Ready ✓", icon="✅")
    elif zone["error"]:
        st.error(zone["error"])
    else:
        st.caption("Not computed yet — press Compute.")


# ---------------------------------------------------------------------------
# Sidebar: combine-zones panel
# ---------------------------------------------------------------------------

def _render_combine_panel() -> None:
    st.header("⚙️ Combine Zones")

    ready_labels = [
        ZONE_LABELS[i]
        for i, z in enumerate(st.session_state.zones)
        if z.get("geojson") is not None and i < len(ZONE_LABELS)
    ]

    if ready_labels:
        st.caption(f"Computed zones: **{', '.join(ready_labels)}**")
    else:
        st.caption("No computed zones yet.")

    # Display mode toggle
    mode = st.radio(
        "Display mode",
        ["Show All", "Expression"],
        index=["Show All", "Expression"].index(st.session_state.display_mode),
        key="radio_display_mode",
        horizontal=True,
    )
    st.session_state.display_mode = mode

    if mode == "Expression":
        st.markdown(
            "**Operators:** `union` `intersect` `cut`  \n"
            "**Symbols:** `|` &nbsp; `&` &nbsp; `-`  \n"
            "**Examples:**  \n"
            "`A union B`  \n"
            "`A intersect B`  \n"
            "`(A union B) cut C`  \n"
            "`(A union B) intersect (C union D)`"
        )

        expr_text = st.text_input(
            "Expression",
            value=st.session_state.expr_text,
            placeholder="e.g. (A union B) cut C",
            key="expr_input",
        )
        st.session_state.expr_text = expr_text

        col1, col2 = st.columns([1, 1])
        with col1:
            result_color = st.color_picker(
                "Result colour",
                st.session_state.expr_color,
                key="expr_color_picker",
            )
            st.session_state.expr_color = result_color
        with col2:
            st.write("")  # vertical spacer
            if st.button("Evaluate", key="btn_evaluate", use_container_width=True):
                _evaluate_expression()
                st.rerun()

        if st.session_state.expr_error:
            st.error(st.session_state.expr_error)
        elif st.session_state.expr_geom is not None:
            st.success("Expression evaluated ✓", icon="✅")


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
        st.session_state.zones.append(
            {
                "id": str(uuid.uuid4()),
                "zone_type": "Isochrone",
                "mode": "Car",
                "minutes": 30,
                "radius_km": 5.0,
                "color": DEFAULT_COLORS[color_idx],
                "lat": None,
                "lon": None,
                "search_text": "",
                "geojson": None,
                "error": None,
            }
        )
        st.rerun()

    st.divider()
    _render_combine_panel()


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
        "Switch to **Expression** mode to combine zones with "
        "`union`, `intersect`, and `cut` — e.g. `(A union B) cut C`."
    )

    render_items = _get_render_items()
    bounds = _bounds_from_zones(st.session_state.zones)
    m = build_map(render_items, bounds)
    st_folium(m, use_container_width=True, height=700, returned_objects=[])


if __name__ == "__main__":
    main()
