"""
Commute Time Map — Streamlit web app
Visualise travel-time isochrones (car, bicycle, walking, public transit) and
as-the-crow-flies distance circles from a chosen origin.  Supports union,
intersection, and difference boolean operations on zones.

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
from shapely.ops import unary_union
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

# Visually distinct palette (colour-blind-friendly-ish)
DEFAULT_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
]

DEFAULT_LAT: float = 51.505
DEFAULT_LON: float = -0.09
MAP_ZOOM: int = 12

BOOL_OPS = ["Show All", "Union", "Intersection", "Difference"]


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    if "zones" not in st.session_state:
        st.session_state.zones: list[dict] = []
    if "center_lat" not in st.session_state:
        st.session_state.center_lat: float = DEFAULT_LAT
    if "center_lon" not in st.session_state:
        st.session_state.center_lon: float = DEFAULT_LON
    if "bool_op" not in st.session_state:
        st.session_state.bool_op: str = "Show All"
    if "diff_a" not in st.session_state:
        st.session_state.diff_a: str | None = None
    if "diff_b" not in st.session_state:
        st.session_state.diff_b: str | None = None


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
    Call the Geoapify Isoline API and return the first GeoJSON Feature, or None.
    api_mode: one of drive | bicycle | walk | transit
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
    Return a GeoJSON Feature (Polygon) for a geodesic circle.
    Uses local UTM projection via pyproj for accuracy.
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

    coords = [
        list(from_utm.transform(px, py))
        for px, py in circle_utm.exterior.coords
    ]
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
    raw = mapping(geom)
    return {"type": "Feature", "geometry": raw, "properties": {}}


def apply_boolean_op(
    zones: list[dict],
    op: str,
    diff_a_id: str | None,
    diff_b_id: str | None,
) -> list[dict]:
    """
    Return a list of {"color": str, "feature": dict} render items.
    Zones whose geojson is None are silently skipped.
    """
    ready = [z for z in zones if z.get("geojson") is not None]
    if not ready:
        return []

    if op == "Show All":
        return [{"color": z["color"], "feature": z["geojson"]} for z in ready]

    try:
        if op == "Union":
            geoms = [_to_shapely(z["geojson"]) for z in ready]
            result = unary_union(geoms)
            if result.is_empty:
                st.warning("Union result is empty.")
                return []
            return [{"color": ready[0]["color"], "feature": _to_geojson_feature(result)}]

        if op == "Intersection":
            if len(ready) < 2:
                st.warning("Intersection requires at least 2 computed zones.")
                return []
            geoms = [_to_shapely(z["geojson"]) for z in ready]
            result = geoms[0]
            for g in geoms[1:]:
                result = result.intersection(g)
            if result.is_empty:
                st.warning("The intersection of all zones is empty.")
                return []
            return [{"color": ready[0]["color"], "feature": _to_geojson_feature(result)}]

        if op == "Difference":
            zone_a = next((z for z in ready if z["id"] == diff_a_id), None)
            zone_b = next((z for z in ready if z["id"] == diff_b_id), None)
            if zone_a is None or zone_b is None:
                st.info("Select Zone A and Zone B for the Difference operation.")
                return []
            if diff_a_id == diff_b_id:
                st.warning("Zone A and Zone B must be different.")
                return []
            geom_a = _to_shapely(zone_a["geojson"])
            geom_b = _to_shapely(zone_b["geojson"])
            result = geom_a.difference(geom_b)
            if result.is_empty:
                st.warning("Difference result is empty (Zone B fully covers Zone A).")
                return []
            return [{"color": zone_a["color"], "feature": _to_geojson_feature(result)}]

    except Exception as exc:  # noqa: BLE001
        st.warning(f"Geometry operation failed: {exc}")
        return []

    return []


# ---------------------------------------------------------------------------
# Zone compute
# ---------------------------------------------------------------------------

def _compute_zone(zone: dict) -> None:
    """Fetch / compute the zone geometry; mutates zone in place."""
    lat = st.session_state.center_lat
    lon = st.session_state.center_lon
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
# Map rendering
# ---------------------------------------------------------------------------

def build_map(
    center_lat: float,
    center_lon: float,
    render_items: list[dict],
) -> folium.Map:
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=MAP_ZOOM,
        tiles="CartoDB positron",
    )

    # Origin marker
    folium.Marker(
        location=[center_lat, center_lon],
        tooltip="Origin",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)

    for item in render_items:
        color = item["color"]
        feature = item["feature"]
        # Ensure it's a Feature (not a bare geometry)
        if feature.get("type") != "Feature":
            feature = {"type": "Feature", "geometry": feature, "properties": {}}

        folium.GeoJson(
            feature,
            style_function=lambda _f, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 2,
                "fillOpacity": 0.25,
                "opacity": 0.8,
            },
        ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Sidebar UI helpers
# ---------------------------------------------------------------------------

def _render_zone_editor(idx: int, zone: dict) -> None:
    """Render editing widgets for a single zone inside an expander."""
    z_id = zone["id"]

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
        if new_radius != zone["radius_km"]:
            zone["radius_km"] = float(new_radius)
            zone["geojson"] = None
            zone["error"] = None

    zone["color"] = st.color_picker("Colour", zone["color"], key=f"color_{z_id}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Compute", key=f"compute_{z_id}", use_container_width=True):
            with st.spinner("Computing…"):
                _compute_zone(zone)
            st.rerun()
    with col2:
        if st.button("Remove", key=f"remove_{z_id}", use_container_width=True):
            st.session_state.zones = [
                z for z in st.session_state.zones if z["id"] != z_id
            ]
            st.rerun()

    # Status feedback
    if zone["geojson"] is not None:
        st.success("Ready ✓", icon="✅")
    elif zone["error"]:
        st.error(zone["error"])
    else:
        st.caption("Not computed yet — press Compute.")


def _render_sidebar() -> None:
    # ── Location search ──────────────────────────────────────────────────
    st.header("📍 Location")
    search_text = st.text_input(
        "Search address or place",
        placeholder="e.g. London Bridge, London",
        key="search_input",
    )
    if st.button("Search", key="btn_search", use_container_width=True):
        if search_text.strip():
            with st.spinner("Geocoding…"):
                result = geocode_location(search_text.strip())
            if result:
                st.session_state.center_lat, st.session_state.center_lon = result
                # Invalidate all zones — origin has moved
                for z in st.session_state.zones:
                    z["geojson"] = None
                    z["error"] = None
                st.rerun()
            else:
                st.warning("No results found for that query.")
        else:
            st.warning("Enter an address or place name first.")

    st.caption(
        f"Origin: {st.session_state.center_lat:.4f}, "
        f"{st.session_state.center_lon:.4f}"
    )

    st.divider()

    # ── Zone management ───────────────────────────────────────────────────
    st.header("🗺️ Zones")

    for idx, zone in enumerate(st.session_state.zones):
        label = zone["label"]
        status_dot = "🟢" if zone["geojson"] else ("🔴" if zone["error"] else "⚪")
        with st.expander(f"{status_dot} {label}", expanded=True):
            _render_zone_editor(idx, zone)

    if st.button("＋ Add Zone", key="btn_add_zone", use_container_width=True):
        color_idx = len(st.session_state.zones) % len(DEFAULT_COLORS)
        st.session_state.zones.append(
            {
                "id": str(uuid.uuid4()),
                "label": f"Zone {len(st.session_state.zones) + 1}",
                "zone_type": "Isochrone",
                "mode": "Car",
                "minutes": 30,
                "radius_km": 5.0,
                "color": DEFAULT_COLORS[color_idx],
                "geojson": None,
                "error": None,
            }
        )
        st.rerun()

    st.divider()

    # ── Boolean operations ────────────────────────────────────────────────
    st.header("⚙️ Boolean Operation")

    bool_op = st.radio(
        "Display mode",
        BOOL_OPS,
        index=BOOL_OPS.index(st.session_state.bool_op),
        key="radio_bool_op",
        help=(
            "**Show All** — each zone in its own colour.\n\n"
            "**Union** — merge all zones.\n\n"
            "**Intersection** — area reachable by *all* zones.\n\n"
            "**Difference** — Zone A minus Zone B."
        ),
    )
    st.session_state.bool_op = bool_op

    if bool_op == "Difference":
        ready_zones = [z for z in st.session_state.zones if z["geojson"] is not None]
        if len(ready_zones) < 2:
            st.info("You need at least 2 computed zones for Difference.")
        else:
            zone_options = {z["id"]: z["label"] for z in ready_zones}
            ids = list(zone_options.keys())

            # Default selections: first two ready zones
            default_a = st.session_state.diff_a if st.session_state.diff_a in ids else ids[0]
            default_b = st.session_state.diff_b if st.session_state.diff_b in ids else ids[1]

            st.session_state.diff_a = st.selectbox(
                "Zone A (keep)",
                options=ids,
                format_func=lambda i: zone_options[i],
                index=ids.index(default_a),
                key="sel_diff_a",
            )
            st.session_state.diff_b = st.selectbox(
                "Zone B (subtract)",
                options=ids,
                format_func=lambda i: zone_options[i],
                index=ids.index(default_b),
                key="sel_diff_b",
            )


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

    # ── API-key banner (shown once if key missing) ────────────────────────
    if not GEOAPIFY_KEY:
        st.warning(
            "**No GEOAPIFY_API_KEY found.** "
            "Isochrones and address search are disabled. "
            "Distance circles (as-the-crow-flies) work without a key.\n\n"
            "Get a free key at https://myprojects.geoapify.com/ and add it to "
            "a `.env` file: `GEOAPIFY_API_KEY=your_key_here`",
            icon="⚠️",
        )

    with st.sidebar:
        _render_sidebar()

    # ── Main area ─────────────────────────────────────────────────────────
    st.title("🗺️ Commute Time Map")
    st.caption(
        "Add zones in the sidebar, press **Compute**, then choose a boolean "
        "operation to combine them."
    )

    render_items = apply_boolean_op(
        st.session_state.zones,
        st.session_state.bool_op,
        st.session_state.diff_a,
        st.session_state.diff_b,
    )

    m = build_map(
        st.session_state.center_lat,
        st.session_state.center_lon,
        render_items,
    )

    st_folium(m, use_container_width=True, height=700, returned_objects=[])


if __name__ == "__main__":
    main()
