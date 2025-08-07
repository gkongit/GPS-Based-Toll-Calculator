import pydeck as pdk
import openrouteservice
import pandas as pd
import requests
import streamlit as st
from openrouteservice import convert
import time
from shapely.geometry import Point, LineString
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyproj import Transformer
import json
import os
from dotenv import load_dotenv
from geopy.distance import geodesic
from twilio.rest import Client as TwilioClient
from streamlit_searchbox import st_searchbox

load_dotenv()
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM = os.getenv('TWILIO_FROM')
TWILIO_TO = os.getenv('TWILIO_TO')

CAR_ICON_ATLAS = "https://cdn-icons-png.flaticon.com/512/744/744465.png"
CAR_ICON_MAPPING = {
    "car": {
        "x": 0,
        "y": 0,
        "width": 512,
        "height": 512,
        "anchorX": 256,
        "anchorY": 256
    }
}


def get_projected_transformer():
    return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def get_coordinates(city_name):
    url = f"https://nominatim.openstreetmap.org/search?city={city_name}&format=json"
    headers = {"User-Agent": "TollRouteApp/1.0 (gagan@example.com)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"Error fetching coordinates for {city_name}: {e}")
    return None

def get_route(client, coords):
    try:
        route = client.directions(coords, options={"avoid_features": []})
        geometry = route["routes"][0]["geometry"]
        decoded = convert.decode_polyline(geometry)
        return [(pt[1], pt[0]) for pt in decoded["coordinates"]]
    except Exception as e:
        print(f"Error fetching route: {e}")
        return None

def send_sms(message):
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(body=message, from_=TWILIO_FROM, to=TWILIO_TO)
    except Exception as e:
        st.error(f"Error sending SMS: {e}")

def load_toll_metadata():
    try:
        df = pd.read_csv("india-tollplaza-data-nhai.csv")
        metadata = []
        for _, row in df.iterrows():
            car_rate = row["Car Rate Single"]
            if pd.isna(car_rate):
                car_rate = "N/A"
            metadata.append({
                "name": row["Tollname"],
                "lat": row["Latitude"],
                "lon": row["Longitude"],
                "car_rate_single": car_rate 
            })
        return metadata
    except Exception as e:
        print("Error loading toll metadata:", e)
        return []


def get_static_map_layers(route_coords, matched_booths):
    path_df = pd.DataFrame([{ "path": [[lon, lat] for lat, lon in route_coords] }])

    toll_df = pd.DataFrame([{
        "lat": booth["lat"],
        "lon": booth["lon"],
        "name": booth["name"],
        "icon_data": {
            "url": "https://i.ibb.co/qfJKF1V/information.png",
            "width": 128,
            "height": 128,
            "anchorY": 128,
        }
    } for booth in matched_booths])

    start_icon_df = pd.DataFrame([{
        "lat": route_coords[0][0],
        "lon": route_coords[0][1],
        "icon_data": {
            "url": "https://i.ibb.co/b59h8H0K/play.png",
            "width": 128,
            "height": 128,
            "anchorY": 128
        }
    }])

    end_icon_df = pd.DataFrame([{
        "lat": route_coords[-1][0],
        "lon": route_coords[-1][1],
        "icon_data": {
            "url": "https://i.ibb.co/QjJtvjk1/star.png",
            "width": 128,
            "height": 128,
            "anchorY": 128
        }
    }])

    return [
        pdk.Layer("PathLayer", data=path_df, get_path="path", get_color=[0, 0, 255], width_scale=20, width_min_pixels=3),
        pdk.Layer("IconLayer", data=toll_df, get_icon="icon_data", get_position='[lon, lat]', get_size=4, size_scale=10, pickable=True),
        pdk.Layer("IconLayer", data=start_icon_df, get_icon="icon_data", get_position="[lon, lat]", get_size=4, size_scale=10, pickable=False),
        pdk.Layer("IconLayer", data=end_icon_df, get_icon="icon_data", get_position="[lon, lat]", get_size=4, size_scale=10, pickable=False),
    ]

def get_car_layer(car_location, current_zoom):
    lat, lon = car_location
    icon_size = max(2, min(current_zoom, 30)) / 2
    return [
        pdk.Layer(
            "IconLayer",
            data=[{
                "lat": lat,
                "lon": lon,
                "name": "car",
                "size": icon_size
            }],
            icon_atlas=CAR_ICON_ATLAS,
            icon_mapping=CAR_ICON_MAPPING,
            get_icon="name",
            get_position="[lon, lat]",
            get_size="size",
            size_scale=10,  
            pickable=False
        )
    ]

def simulate_live_drive(route_coords, matched_booths, speed_kmph):
    speed_mps = speed_kmph * 1000 / 3600
    interval = 0.01

    if "sent_booths" not in st.session_state:
        st.session_state["sent_booths"] = set()

    if "step_index" not in st.session_state:
        st.session_state.step_index = 0

    static_layers = get_static_map_layers(route_coords, matched_booths)
    placeholder = st.empty()

    for i in range(st.session_state.step_index + 1, len(route_coords)):
        start = route_coords[i - 1]
        end = route_coords[i]
        dist = geodesic(start, end).meters
        steps = max(1, int(dist / 10))

        for step in range(steps):
            lat = start[0] + (end[0] - start[0]) * (step / steps)
            lon = start[1] + (end[1] - start[1]) * (step / steps)
            st.session_state.car_location = (lat, lon)
            st.session_state.step_index = i

            for booth in matched_booths:
                if booth["name"] not in st.session_state["sent_booths"]:
                    distance_to_booth = geodesic((lat, lon), (booth["lat"], booth["lon"])).meters
                    if distance_to_booth < 400:
                        st.session_state["sent_booths"].add(booth["name"])
                        msg = f"Passed toll booth: {booth['name']} | Toll: ₹{booth.get('car_rate_single', 'N/A')}"

                        send_sms(msg)
                        st.success(msg)

            car_layers = get_car_layer((lat, lon), 12)
            with placeholder:
                st.pydeck_chart(pdk.Deck(
                    map_style="mapbox://styles/mapbox/streets-v11",
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=10, pitch=0),
                    layers=static_layers + car_layers
                ))

            time.sleep(interval)

    st.success("Destination reached. Simulation complete.")
    for key in ("step_index", "car_location", "sent_booths"):
        st.session_state.pop(key, None)

def get_toll_booths(route_coords, step=10, max_requests=100):
    toll_booths = set()
    overpass_url = "https://overpass.kumi.systems/api/interpreter"
    coords_to_check = route_coords[::step][:max_requests]
    progress_bar = st.progress(0)

    def find_nearest_metadata_booth(lat, lon, metadata_list, max_distance_km=5):
        nearest = None
        min_dist = float("inf")
        for booth in metadata_list:
            dist = geodesic((lat, lon), (booth["lat"], booth["lon"])).km
            if dist < min_dist and dist <= max_distance_km:
                min_dist = dist
                nearest = booth
        return nearest

    def query_toll_nodes(lat, lon):
        query = f"""
        [out:json][timeout:30];
        (
          node["barrier"="toll_booth"](around:50000,{lat},{lon});
          node["highway"="toll_gantry"](around:50000,{lat},{lon});
          node["amenity"="toll_booth"](around:50000,{lat},{lon});
          way["barrier"="toll_booth"](around:50000,{lat},{lon});
          way["highway"="toll_gantry"](around:50000,{lat},{lon});
          way["amenity"="toll_booth"](around:50000,{lat},{lon});
        );
        out center;
        """
        for _ in range(3):
            try:
                resp = requests.get(overpass_url, params={"data": query}, timeout=15)
                data = resp.json()
                return {(el.get("lat", el.get("center", {}).get("lat")),
                                  el.get("lon", el.get("center", {}).get("lon")),
                                  el.get("tags", {}).get("name", "Toll Booth"))
                                 for el in data["elements"]}
            except:
                continue
        return set()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(query_toll_nodes, lat, lon): idx for idx, (lat, lon) in enumerate(coords_to_check)}
        for i, future in enumerate(as_completed(futures)):
            toll_booths.update(future.result())
            progress_bar.progress((i + 1) / len(coords_to_check))

    progress_bar.empty()

    transformer = get_projected_transformer()
    projected_route = LineString([transformer.transform(lon, lat) for lat, lon in route_coords])
    raw_points = [Point(transformer.transform(lon, lat)) for lat, lon, _ in toll_booths]

    booths_with_pos = [(lat, lon, name, projected_route.project(pt), projected_route.distance(pt))
                                 for (lat, lon, name), pt in zip(toll_booths, raw_points)]
    booths_with_pos.sort(key=lambda x: x[2])

    filtered = []
    last_proj = -1
    for lat, lon, name, proj, dist in booths_with_pos:
        if dist > 20:
            continue
        if last_proj < 0 or abs(proj - last_proj) > 5:
            filtered.append((lat, lon, name))
            last_proj = proj

    metadata = load_toll_metadata()
    matched = []
    for lat, lon, name in filtered:
        matched_meta = find_nearest_metadata_booth(lat, lon, metadata)
        toll_price = "N/A"
        if matched_meta and "car_rate_single" in matched_meta:
            try:
                toll_price = float(matched_meta["car_rate_single"])
            except (ValueError, TypeError):
                toll_price = "N/A"
        else:
            print(f"DEBUG: No metadata found for OSM toll booth: {name} at ({lat}, {lon})")
        if matched_meta and (not name or name == "Toll Booth"):
            name = matched_meta.get("name", name)
        matched.append({
            "lat": lat,
            "lon": lon,
            "name": name,
            "metadata": matched_meta,
            "car_rate_single": toll_price
        })


    return raw_points, matched

def draw_pydeck_map(route_coords, matched_booths, car_location=None, city1=None, city2=None):
    path_df = pd.DataFrame([{ "path": [[lon, lat] for lat, lon in route_coords] }])

    toll_df = pd.DataFrame([{
        "lat": booth["lat"],
        "lon": booth["lon"],
        "name": booth["name"],
        "label": f"{booth['name']} (₹{booth.get('car_rate_single', 'N/A')})",
        "icon_data": {
            "url": "https://i.ibb.co/qfJKF1V/information.png",
            "width": 90,
            "height": 90,
            "anchorY": 90,
        }
    } for booth in matched_booths])

    toll_df["label"] = toll_df["name"]    
    
    start_icon_url = "https://i.ibb.co/b59h8H0K/play.png"
    start_icon_df = pd.DataFrame([{
        "lat": route_coords[0][0],
        "lon": route_coords[0][1],
        "icon_data": {
            "url": start_icon_url,
            "width": 128,
            "height": 128,
            "anchorY": 128
        }
    }])
    
    end_icon_url = "https://i.ibb.co/QjJtvjk1/star.png"
    end_icon_df = pd.DataFrame([{
        "lat": route_coords[-1][0],
        "lon": route_coords[-1][1],
        "icon_data": {
            "url": end_icon_url,
            "width": 128,
            "height": 128,
            "anchorY": 128
        }
    }])

    pins = []
    if city1:
        pins.append({"lat": route_coords[0][0], "lon": route_coords[0][1]})
    if city2:
        pins.append({"lat": route_coords[-1][0], "lon": route_coords[-1][1]})
    if car_location:
        pins.append({"lat": car_location[0], "lon": car_location[1]})

    pin_df = pd.DataFrame(pins)

    layers = [
        pdk.Layer("PathLayer", data=path_df, get_path="path", get_color=[0, 0, 255], width_scale=20, width_min_pixels=3),
        pdk.Layer("IconLayer", data=toll_df, get_icon="icon_data", get_position='[lon, lat]', get_size=4, size_scale=10, pickable=True),
        pdk.Layer("ScatterplotLayer", data=start_icon_df, get_position='[lon, lat]', get_color=[0, 255, 0], get_radius=200),
        pdk.Layer(
            "TextLayer",
            data=pin_df,
            get_position='[lon, lat]',
            get_text='label',
            get_size=16,
            get_color=[255, 0, 0],
            get_alignment_baseline='"top"',
            get_text_anchor='"middle"',  
            offset=[0, -1000]                
        ),
        pdk.Layer( 
            "IconLayer",
            data=start_icon_df,
            get_icon="icon_data",
            get_position="[lon, lat]",
            get_size=4,
            size_scale=10,
            pickable=False
        ),
        pdk.Layer(
            "IconLayer",
            data=end_icon_df,
            get_icon="icon_data",
            get_position="[lon, lat]",
            get_size=4,
            size_scale=10,
            pickable=False
        )
    ]

    view_state = pdk.ViewState(latitude=car_location[0] if car_location else route_coords[0][0],
                                 longitude=car_location[1] if car_location else route_coords[0][1],
                                 zoom=12, pitch=0)

    st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/streets-v11',
                                 initial_view_state=view_state,
                                 layers=layers))

def main():
    st.title("Toll Route Visualizer")
    city1 = st.text_input("Enter source city", placeholder="e.g., Mangalore")
    city2 = st.text_input("Enter destination city", placeholder="e.g., Bangalore")

    if st.button("Show Route with Tolls"):
        coords1 = get_coordinates(city1)
        coords2 = get_coordinates(city2)
        if not coords1 or not coords2:
            st.error("Could not fetch coordinates for one or both cities.")
            return

        client = openrouteservice.Client(key="<your_client_key>")
        route_coords = get_route(client, [(coords1[1], coords1[0]), (coords2[1], coords2[0])])
        if not route_coords:
            st.error("Could not fetch route between cities.")
            return

        raw_points, matched_booths = get_toll_booths(route_coords)
        st.session_state.update({
            "route_coords": route_coords,
            "matched_booths": matched_booths,
            "city1": city1,
            "city2": city2,
            "raw_points": raw_points
        })

        st.success(f"Found route with {len(matched_booths)} toll booths.")

    if all(k in st.session_state for k in ("route_coords", "matched_booths", "raw_points", "city1", "city2")):
        st.subheader("Toll Booths on Route and Their Prices")
        if st.session_state["matched_booths"]:
            toll_data = []
            total_toll_cost = 0 
            for booth in st.session_state["matched_booths"]:
                toll_price_display = "N/A"
                current_booth_cost = 0

                if 'car_rate_single' in booth and isinstance(booth['car_rate_single'], (int, float)):
                    toll_price_display = f"₹{booth['car_rate_single']:.2f}"
                    current_booth_cost = booth['car_rate_single']
                elif 'car_rate_single' in booth and booth['car_rate_single'] == "N/A":
                    toll_price_display = "N/A"
                elif 'car_rate_single' in booth and isinstance(booth['car_rate_single'], str):
                    try:
                        current_booth_cost = float(booth['car_rate_single'])
                        toll_price_display = f"₹{current_booth_cost:.2f}"
                    except ValueError:
                        toll_price_display = "N/A"

                toll_data.append({
                    "Toll Name": booth["name"],
                    "Car Rate (Single Trip)": toll_price_display
                })
                total_toll_cost += current_booth_cost

            st.dataframe(pd.DataFrame(toll_data))

            # Display total toll cost
            st.markdown(f"**Total Estimated Toll Cost: ₹{total_toll_cost:.2f}**")
        else:
            st.info("No toll booths found on this route.")

        draw_pydeck_map(
            st.session_state["route_coords"],
            st.session_state["matched_booths"],
            car_location=st.session_state.get("car_location"),
            city1=st.session_state["city1"],
            city2=st.session_state["city2"]
        )

        st.subheader("Simulate Vehicle Drive")
        with st.form("drive_simulation_form"):
            speed = st.slider("Speed (km/h)", 10, 20000, 60, key="speed_slider")
            submitted = st.form_submit_button("Start Simulation")

        if st.button("Reset Simulation"):
            for key in ("step_index", "car_location", "sent_booths", "start_sim"):
                st.session_state.pop(key, None)

        if submitted:
            st.session_state["start_sim"] = True
            st.session_state["speed_kmph"] = st.session_state["speed_slider"]

    if st.session_state.get("start_sim", False):
        simulate_live_drive(
            st.session_state["route_coords"],
            st.session_state["matched_booths"],
            st.session_state["speed_kmph"]
        )
        st.session_state["start_sim"] = False

if __name__ == "__main__":
    main()
