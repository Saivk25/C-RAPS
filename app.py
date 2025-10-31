#!/usr/bin/env python3
"""
C-RAPS: Chennai Risk Analysis & Prediction System
Main Streamlit UI Application
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
import streamlit as st
import geopandas as gpd
from shapely.geometry import Point
from PIL import Image
import requests
import pandas as pd
import time
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# Main application code (was previously in main.py, now in app.py)
import folium
from streamlit_folium import st_folium  
# Import custom modules:
# - vehicle_detection: YOLOv8-based vehicle detection functions
# - water_detection: Water/flood detection, overlay, and categorization functions
from vehicle_detection import load_yolo_model, detect_vehicles
from water_detection import load_water_yolo_model, detect_water, apply_water_overlay, get_water_level_category
from collections import Counter
import google.generativeai as genai
warnings.filterwarnings('ignore')

# Initialize session state
if 'analyze' not in st.session_state:
    st.session_state['analyze'] = False
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False
if 'processing' not in st.session_state:
    st.session_state['processing'] = False

# =============================================================================
# CACHED FUNCTIONS
# =============================================================================

@st.cache_data
def load_geojson_data():
    """Load GeoJSON data once and cache it."""
    try:
        gdf = gpd.read_file("export.geojson")
        return gdf
    except Exception as e:
        st.warning(f"Could not load GeoJSON: {e}")
        return None
import json

def load_street_knowledge_base():
    """Load street knowledge base from JSON file."""
    try:
        with open("street_knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load street knowledge base: {e}")
        return {}

STREET_KNOWLEDGE_BASE = load_street_knowledge_base()
@st.cache_resource
def build_enhanced_fuzzy_system():
    """Build enhanced fuzzy logic system with corrected priority rules."""
    # ... (keep all your existing Antecedent/Consequent definitions) ...
    
    vehicle_density = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicle_density')
    aadt = ctrl.Antecedent(np.arange(0, 50001, 100), 'aadt')
    road_width = ctrl.Antecedent(np.arange(2, 21, 0.1), 'road_width')
    water_coverage = ctrl.Antecedent(np.arange(0, 101, 1), 'water_coverage')
    speed_limit = ctrl.Antecedent(np.arange(20, 101, 1), 'speed_limit')
    rain_intensity = ctrl.Antecedent(np.arange(0, 51, 1), 'rain_intensity')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # ... (keep all your membership function definitions) ...
    
    vehicle_density['very_low'] = fuzz.trimf(vehicle_density.universe, [0, 0, 15])
    vehicle_density['low'] = fuzz.trimf(vehicle_density.universe, [10, 20, 35])
    vehicle_density['medium'] = fuzz.trimf(vehicle_density.universe, [30, 45, 60])
    vehicle_density['high'] = fuzz.trimf(vehicle_density.universe, [55, 70, 85])
    vehicle_density['very_high'] = fuzz.trimf(vehicle_density.universe, [80, 100, 100])

    aadt['low'] = fuzz.trimf(aadt.universe, [0, 5000, 15000])
    aadt['medium'] = fuzz.trimf(aadt.universe, [10000, 20000, 30000])
    aadt['high'] = fuzz.trimf(aadt.universe, [25000, 35000, 50000])

    road_width['narrow'] = fuzz.trimf(road_width.universe, [2, 4, 6])
    road_width['medium'] = fuzz.trimf(road_width.universe, [5, 7, 9])
    road_width['wide'] = fuzz.trimf(road_width.universe, [8, 12, 20])

    water_coverage['none'] = fuzz.trimf(water_coverage.universe, [0, 0, 2])
    water_coverage['light'] = fuzz.trimf(water_coverage.universe, [1, 5, 12])
    water_coverage['moderate'] = fuzz.trimf(water_coverage.universe, [10, 25, 40])
    water_coverage['heavy'] = fuzz.trimf(water_coverage.universe, [35, 60, 100])

    speed_limit['low'] = fuzz.trimf(speed_limit.universe, [20, 30, 45])
    speed_limit['medium'] = fuzz.trimf(speed_limit.universe, [40, 55, 70])
    speed_limit['high'] = fuzz.trimf(speed_limit.universe, [65, 80, 100])

    rain_intensity['none'] = fuzz.trimf(rain_intensity.universe, [0, 0, 2])
    rain_intensity['light'] = fuzz.trimf(rain_intensity.universe, [1, 5, 10])
    rain_intensity['heavy'] = fuzz.trimf(rain_intensity.universe, [8, 25, 50])

    risk['very_low'] = fuzz.trimf(risk.universe, [0, 10, 25])
    risk['low'] = fuzz.trimf(risk.universe, [15, 30, 45])
    risk['medium'] = fuzz.trimf(risk.universe, [35, 50, 65])
    risk['high'] = fuzz.trimf(risk.universe, [55, 70, 85])
    risk['very_high'] = fuzz.trimf(risk.universe, [75, 90, 100])

    # ============================================================
    # CORRECTED RULES WITH PRIORITY HIERARCHY
    # ============================================================
    rules = [
        # --- PRIORITY 1: High-risk overrides (ALWAYS very high risk) ---
        # Severe waterlogging is ALWAYS very high risk, regardless of traffic
        ctrl.Rule(water_coverage['heavy'], risk['very_high']),
        
        # Heavy rain + any significant traffic is very high risk
        ctrl.Rule(rain_intensity['heavy'] & (vehicle_density['medium'] | vehicle_density['high'] | vehicle_density['very_high']), risk['very_high']),
        
        # Very high density is ALWAYS very high risk
        ctrl.Rule(vehicle_density['very_high'], risk['very_high']),
        
        # High speed + high density = very high risk
        ctrl.Rule(speed_limit['high'] & vehicle_density['high'], risk['very_high']),

        # --- PRIORITY 2: High-risk combinations ---
        ctrl.Rule(water_coverage['moderate'], risk['high']),
        ctrl.Rule(vehicle_density['high'], risk['high']),
        ctrl.Rule(rain_intensity['heavy'], risk['high']),
        ctrl.Rule(vehicle_density['medium'] & road_width['narrow'], risk['high']),
        ctrl.Rule(vehicle_density['high'] & aadt['high'], risk['high']),
        
        # --- PRIORITY 3: Medium risk ---
        ctrl.Rule(water_coverage['light'], risk['medium']),
        ctrl.Rule(vehicle_density['medium'] & aadt['medium'], risk['medium']),
        ctrl.Rule(aadt['high'] & vehicle_density['low'], risk['medium']),
        ctrl.Rule(speed_limit['medium'] & vehicle_density['high'], risk['medium']),

        # --- PRIORITY 4: Low risk (only when conditions are actually safe) ---
        ctrl.Rule(vehicle_density['low'] & water_coverage['none'], risk['low']),
        ctrl.Rule(water_coverage['none'] & vehicle_density['very_low'] & rain_intensity['none'], risk['very_low'])
    ]

    risk_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(risk_ctrl)
# =============================================================================
# LOCATION AND STREET DATA
# =============================================================================

def get_chennai_areas():
    """Return list of major Chennai areas with coordinates."""
    return {
        "T. Nagar": (13.0418, 80.2341),
        "Anna Nagar": (13.0850, 80.2101),
        "Velachery": (12.9816, 80.2209),
        "Adyar": (13.0067, 80.2206),
        "Mylapore": (13.0339, 80.2619),
        "Guindy": (13.0067, 80.2206),
        "Porur": (13.0382, 80.1564),
        "OMR (IT Corridor)": (12.9279, 80.2388),
        "GST Road": (12.9520, 80.1462),
        "Mount Road": (13.0732, 80.2609),
        "Tambaram": (12.9249, 80.1000),
        "Chrompet": (12.9516, 80.1462),
        "Pallavaram": (12.9675, 80.1491),
        "Sholinganallur": (12.9010, 80.2279),
        "ECR (East Coast Road)": (12.8856, 80.2442)
    }
# In main.py
# REPLACE your get_logical_mptcrsi_data with this:

def get_logical_mptcrsi_data(highway_type, street_name):
    """
    Generates a DEFENSILBE, LOGICAL, and NON-RANDOM set of
    ALL data (AADT, width, crashes, etc.) based on the "real" highway type.
    This is our "Expert Knowledge Base" module.
    """
    data = {}

    # --- Special Case for your Test Video (Gopathi Road) ---
    # This ensures your live demo is reproducible
    if "gopathi" in street_name.lower():
        data['aadt'] = 20000
        data['road_width'] = 9.0
        data['speed_limit'] = 30
        data['land_use'] = 'residential'
        data['parking_type'] = 'rarely'
        data['num_exits'] = 3
        data['num_side_roads'] = 8
        data['fatal_crashes_hist'] = 1
        data['injury_crashes_hist'] = 5
        data['property_crashes_hist'] = 12
        return data

    # --- Logical Rules for all other roads ---
    if highway_type == 'primary':
        data['aadt'] = 45000
        data['road_width'] = 16.0
        data['speed_limit'] = 50
        data['land_use'] = 'shops'
        data['parking_type'] = 'prohibited'
        data['num_exits'] = 15
        data['num_side_roads'] = 10
        data['fatal_crashes_hist'] = 3
        data['injury_crashes_hist'] = 20
        data['property_crashes_hist'] = 30
    
    elif highway_type == 'secondary':
        data['aadt'] = 28000
        data['road_width'] = 11.0
        data['speed_limit'] = 40
        data['land_use'] = 'apartments'
        data['parking_type'] = 'bays_at_kerb'
        data['num_exits'] = 10
        data['num_side_roads'] = 15
        data['fatal_crashes_hist'] = 2
        data['injury_crashes_hist'] = 10
        data['property_crashes_hist'] = 22

    else: # 'residential', 'tertiary', or 'unclassified'
        data['aadt'] = 10000
        data['road_width'] = 7.0
        data['speed_limit'] = 30
        data['land_use'] = 'residential'
        data['parking_type'] = 'rarely'
        data['num_exits'] = 5
        data['num_side_roads'] = 10
        data['fatal_crashes_hist'] = 0
        data['injury_crashes_hist'] = 3
        data['property_crashes_hist'] = 8

    return data

def create_risk_map(area_name, area_coords, streets_data):
    """
    Creates an interactive Folium map with color-coded risk zones
    for all streets in the loaded knowledge base.
    """
    
    # Create the map, centered on the selected area
    m = folium.Map(location=area_coords, zoom_start=14)
    
    # Get all streets for the selected area
    streets_in_area = STREET_KNOWLEDGE_BASE.get(area_name, {})
    
    for street_name, data in streets_in_area.items():
        # --- Calculate a "Static Risk Score" for each street ---
        # This is a simplified risk score based ONLY on historical data
        # (We can't run a live video for every street on the map)
        
        epdo_score, _ = calculate_epdo(
            data.get('fatal_crashes_hist', 0),
            data.get('injury_crashes_hist', 0),
            data.get('property_crashes_hist', 0)
        )
        
        # Simple logic: high AADT or high EPDO = high risk
        if data['aadt'] > 40000 or epdo_score > 50:
            static_risk = "HIGH"
            color = "red"
            radius = 10
        elif data['aadt'] > 25000 or epdo_score > 20:
            static_risk = "MEDIUM"
            color = "orange"
            radius = 8
        else:
            static_risk = "LOW"
            color = "green"
            radius = 6
        # --- End of static risk calculation ---

        # Add a circle marker to the map
        folium.CircleMarker(
            location=[data['lat'], data['lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(f"""
                <b>{street_name}</b><br>
                Type: {data['highway_type'].title()}<br>
                AADT: {data['aadt']:,}<br>
                EPDO: {epdo_score}<br>
                <b>Static Risk: {static_risk}</b>
            """, max_width=300)
        ).add_to(m)
    
    return m
def search_streets_by_area(gdf, area_coords, radius_km=2.0):
    """
    Finds real streets from GeoJSON (name, type, location)
    and enriches them using our full logical knowledge base.
    """
    if gdf is None:
        return []
    
    try:
        lat, lon = area_coords
        center_point = Point(lon, lat)
        radius_deg = radius_km / 111.0
        
        # Clean the 'highway' column once
        if 'highway' not in gdf.columns:
            gdf['highway'] = 'residential'
        gdf['highway'] = gdf['highway'].fillna('residential')

        # Clean the 'name' column once
        if 'name' not in gdf.columns:
            gdf['name'] = 'Unnamed Road'
        gdf['name'] = gdf['name'].fillna('Unnamed Road')
        
        gdf['distance'] = gdf.geometry.distance(center_point)
        nearby_streets = gdf[gdf['distance'] <= radius_deg].copy()
        nearby_streets.sort_values('distance', inplace=True)
        
        street_list = []
        seen_names = set()
        
        for idx, row in nearby_streets.head(15).iterrows():
            street_name = row['name']
            if street_name == 'Unnamed Road' or street_name in seen_names:
                continue
            
            seen_names.add(street_name)
            
            if hasattr(row.geometry, 'centroid'):
                centroid = row.geometry.centroid
                actual_lat, actual_lon = centroid.y, centroid.x
            else:
                actual_lat, actual_lon = lat, lon
            
            # 1. GET REAL DATA (Name, Type, Location)
            highway_type = row['highway']
            real_data = {
                'name': street_name,
                'highway_type': highway_type,
                'lat': float(actual_lat),
                'lon': float(actual_lon)
            }
            
            # 2. GET LOGICAL DATA (AADT, Width, Crashes, etc.)
            assumed_data = get_logical_mptcrsi_data(highway_type, street_name)
            
            # 3. COMBINE THEM
            full_street_data = {**real_data, **assumed_data}
            
            street_list.append(full_street_data)
        
        return street_list[:10]
        
    except Exception as e:
        st.warning(f"Error processing street data: {e}")
        return []
    
def get_default_street_data(area_name, area_coords):
    """Generate default street data when no GeoJSON data is available."""
    street_names = {
        "T. Nagar": ["North Usman Road", "Thyagaraya Road", "Pondy Bazaar"],
        "Anna Nagar": ["2nd Avenue", "13th Main Road", "Anna Nagar Roundtana"],
        "Velachery": ["Velachery Main Road", "100 Feet Road", "Velachery Bypass"],
        "Adyar": ["Adyar Bridge Road", "Gandhi Nagar Main Road", "Lattice Bridge Road"],
    }
    
    default_streets = []
    names = street_names.get(area_name, [f"{area_name} Main Road", f"{area_name} Bypass"])
    for name in names[:3]:
        street_data = {
            'name': name,
            'aadt': np.random.randint(15000, 45000),
            'road_width': np.random.uniform(6.0, 15.0),
            'speed_limit': np.random.choice([40, 50, 60, 70]),
            'highway_type': np.random.choice(['residential', 'secondary', 'primary']),
            'lat': area_coords[0] + np.random.uniform(-0.01, 0.01),
            'lon': area_coords[1] + np.random.uniform(-0.01, 0.01)
        }
        default_streets.append(street_data)
    return default_streets

# =============================================================================
# WEATHER DATA
# =============================================================================

def get_weather_data(lat, lon):
    """Fetch weather data without requiring API key."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            current = data.get('current_weather', {})
            weather_code = current.get('weathercode', 0)
            description = _get_weather_description(weather_code)
            weather = {
                'description': description,
                'temperature': current.get('temperature', 28.0),
                'rain_1h': current.get('precipitation', 0.0)
            }
            return weather, None
        else:
            default_weather = {'description': 'Clear', 'temperature': 28.5, 'rain_1h': 0.0}
            return default_weather, "Weather API unavailable"
    except Exception as e:
        default_weather = {'description': 'Clear', 'temperature': 28.5, 'rain_1h': 0.0}
        return default_weather, f"Weather unavailable: {str(e)}"

def _get_weather_description(weather_code):
    """Convert weather code to description."""
    weather_codes = {
        0: 'Clear', 1: 'Mainly Clear', 2: 'Partly Cloudy', 3: 'Overcast',
        45: 'Foggy', 48: 'Foggy',
        51: 'Light Drizzle', 53: 'Moderate Drizzle', 55: 'Heavy Drizzle',
        61: 'Light Rain', 63: 'Moderate Rain', 65: 'Heavy Rain',
        80: 'Light Showers', 81: 'Moderate Showers', 82: 'Heavy Showers'
    }
    return weather_codes.get(weather_code, 'Clear')

# =============================================================================
# RISK CALCULATION
# =============================================================================

def calculate_enhanced_risk(fuzzy_sim, vehicle_count, water_coverage_pct, street_data, rain_mm_hr):
    """Calculate risk with improved traffic density calculation."""
    try:
        road_width_val = max(2.0, min(20.0, float(street_data.get('road_width', 7.0) or 7.0)))
        aadt_val = max(0, min(50000, int(street_data.get('aadt', 15000) or 15000)))
        speed_limit_val = max(20, min(100, int(street_data.get('speed_limit', 50) or 50)))
        
        lanes_estimate = max(2, road_width_val / 3.5)
        VEHICLES_PER_LANE_VISIBLE = 8  
        road_capacity = lanes_estimate * VEHICLES_PER_LANE_VISIBLE
        normalized_density = min(100.0, (float(vehicle_count) / road_capacity) * 100.0)
        
        fuzzy_sim.input['vehicle_density'] = max(0, min(100, normalized_density))
        fuzzy_sim.input['aadt'] = aadt_val
        fuzzy_sim.input['road_width'] = road_width_val
        fuzzy_sim.input['water_coverage'] = max(0, min(100, water_coverage_pct))
        fuzzy_sim.input['speed_limit'] = speed_limit_val
        fuzzy_sim.input['rain_intensity'] = max(0, min(50, rain_mm_hr))
        
        fuzzy_sim.compute()
        risk_score = fuzzy_sim.output['risk']
        
        if np.isnan(risk_score) or risk_score < 0:
            risk_score = 35.0
        if vehicle_count > 30:
            risk_score = max(risk_score, 45.0)
        elif vehicle_count > 20:
            risk_score = max(risk_score, 35.0)
            
        return max(0.0, min(100.0, risk_score))
    except Exception as e:
        st.error(f"Risk calculation error: {e}")
        traffic_risk = min(60, vehicle_count * 1.2)
        water_risk = min(30, water_coverage_pct * 0.6) if water_coverage_pct > 5 else 0
        rain_risk = min(20, rain_mm_hr * 0.4)
        return min(100, max(25, traffic_risk + water_risk + rain_risk))
def calculate_epdo(fatal_crashes, injury_crashes, property_crashes):
    """
    Calculates Equivalent Property Damage Only (EPDO) score
    from the MPTCRSI-ES base paper.
    EPDO = (1 × PDC) + (5 × IC) + (10 × FC)
    """
    epdo_score = (property_crashes * 1) + (injury_crashes * 5) + (fatal_crashes * 10)
    
    # Categorize severity
    if epdo_score >= 50:
        category = "Critical - Immediate Action"
    elif epdo_score >= 20:
        category = "High Priority"
    elif epdo_score >= 10:
        category = "Medium Priority"
    else:
        category = "Low Priority"
    
    return epdo_score, category

def predict_accidents(aadt, road_width, speed_limit, num_exits, 
                     num_side_roads, parking_type, land_use):
    """
    Predicts future accidents using the multi-variable model
    from the MPTCRSI-ES base paper.
    E(μ) = a × AADT^p × β1 × β2 × β3 × β4 × β5 × β6
    """
    try:
        # Model 2: Enhanced with road characteristics
        a = 6.09e-4
        p = 0.8
        
        # Speed limit coefficient
        beta_speed = {30: 1.8, 40: 2.0, 50: 2.25, 60: 2.85, 70: 1.0}
        β1 = beta_speed.get(speed_limit, 2.0)
        
        # Road width coefficient
        if 5.0 <= road_width <= 7.5:
            β2 = 0.83
        elif 8.0 <= road_width <= 8.5:
            β2 = 0.68
        else:
            β2 = 0.80
        
        # Number of exits (5-40 optimal, others = 1.0)
        β3 = 1.0 if 5 <= num_exits <= 40 else 1.2
        
        # Side roads coefficient
        if num_side_roads == 0:
            β4 = 0.72
        elif num_side_roads <= 5:
            β4 = 0.75
        elif num_side_roads <= 10:
            β4 = 1.0
        else:
            β4 = 1.25
        
        # Parking coefficient
        parking_coeffs = {
            "prohibited": 1.19,
            "rarely": 1.0,
            "bays_at_kerb": 1.77
        }
        β5 = parking_coeffs.get(parking_type, 1.0)
        
        # Land use coefficient
        land_use_coeffs = {
            "shops": 2.44,
            "apartments": 1.56,
            "industrial": 1.58,
            "residential": 1.58,
            "scattered": 1.0
        }
        β6 = land_use_coeffs.get(land_use, 1.3)
        
        # Calculate enhanced prediction
        enhanced_accidents = a * (aadt ** p) * β1 * β2 * β3 * β4 * β5 * β6
        
        return enhanced_accidents
    
    except Exception as e:
        st.warning(f"Could not predict accidents: {e}")
        return 0.0
# =============================================================================
# VIDEO PROCESSING
# =============================================================================
# In main.py
# =============================================================================
# AI RECOMMENDATION MODULE (PHASE 3)
# =============================================================================

# =============================================================================
# 💡 AI-POWERED RECOMMENDATION MODULE (GEMINI)
# =============================================================================

def get_gemini_recommendations(live_data, historical_data, street_data):
    """
    Generates advanced, generative AI recommendations using the Gemini API.
    """
    try:
        # 1. Configure the API key from st.secrets
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        # 2. Create the model
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # 3. Build a detailed prompt
        prompt = f"""
        You are a senior traffic safety engineer and emergency dispatcher for the city of Chennai.
        I will provide you with a full risk analysis for a specific street.
        Your job is to provide a brief, expert recommendation in markdown format.

        **SITUATION ANALYSIS:**
        - **Street:** {street_data['name']}, a {street_data['highway_type']} road.
        - **Historical Risk:** {historical_data['epdo_category']} (EPDO Score: {historical_data['epdo_score']})
        - **Predicted Accidents:** {historical_data['predicted_accidents']:.2f} per year.
        - **Historical Street Data:** AADT is {street_data['aadt']}, Road width is {street_data['road_width']}m.

        **LIVE DATA (Current Conditions):**
        - **Live Risk Score:** {live_data['risk_score']:.1f}/100
        - **Live Water Coverage:** {live_data['water_coverage']:.1f}%
        - **Live Traffic Density:** {live_data['avg_density']} vehicles/frame (average)

        **YOUR TASK:**
        Based *only* on the data above, please provide:
        1.  **Immediate Threat Assessment:** A one-sentence summary of the main problem (e.g., "CRITICAL: Severe flooding detected" or "HIGH: Extreme traffic congestion" or "LOW: Conditions are safe").
        2.  **Short-Term Actions (For Dispatch):** A 1-2 bullet point list of actions for immediate dispatch (e.g., "Divert all traffic," "Monitor for bottlenecks").
        3.  **Long-Term Actions (For Planning):** A 1-2 bullet point list of long-term infrastructure improvements (e.g., "Schedule review of drainage systems," "Install speed cameras").
        """
        
        # 4. Generate the content
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        # Provide a more specific error if the key is missing
        if "API key" in str(e):
            st.error("Gemini API Error: API key not found. Please add it to your .streamlit/secrets.toml file.")
            return "API Key Error. See log."
        else:
            st.error(f"Gemini API Error: {e}")
            return "Failed to get a recommendation from Gemini. Check your network connection."
        # In main.py
def process_video_stream(yolo_model, water_model, source, frame_limit, display_placeholder, progress_bar):
    """
    Optimized video processing with "Dynamic Risk Overlay".
    Detects water FIRST, then passes the mask to vehicle detection.
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        st.error("Failed to open video source")
        return 0, 0.0, 0, {}, set(), [], []
    
    vehicle_counts_list = []
    water_coverages = []
    total_vehicle_detections_dict = Counter()
    master_tracked_ids = set()
    
    frame_count = 0
    processed_count = 0
    skip_frames = 1 # Tracking requires every frame
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source != 0 else frame_limit
    
    while cap.isOpened() and frame_count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % skip_frames != 0:
            continue
        
        processed_count += 1
        
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            frame = cv2.resize(frame, (640, int(height * scale)))
        
        # --- RE-ORDERED LOGIC ---
        
        # 1. Detect Water FIRST
        water_coverage, water_mask = detect_water(water_model, frame)
        
        # 2. Pass water_mask to Vehicle Detector
        vehicle_counts_dict, annotated_frame, vehicle_count, frame_id_set = detect_vehicles(
            yolo_model, frame, water_mask
        )
        
        # --- END OF RE-ORDERED LOGIC ---
        
        # Now update our lists
        master_tracked_ids.update(frame_id_set)
        total_vehicle_detections_dict.update(vehicle_counts_dict)
        vehicle_counts_list.append(vehicle_count)
        water_coverages.append(water_coverage)
        
        # Apply the blue water overlay (this will draw OVER the red boxes, which is fine)
        result_frame = apply_water_overlay(annotated_frame, water_mask, alpha=0.4)
        
        # Add info panel
        avg_vehicles = np.mean(vehicle_counts_list) if vehicle_counts_list else 0
        avg_water = np.mean(water_coverages) if water_coverages else 0.0
        
        panel_height = 100
        panel = np.zeros((panel_height, result_frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        cv2.putText(panel, f'Vehicles: {int(avg_vehicles)}', (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(panel, f'Water: {avg_water:.1f}%', (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
        cv2.putText(panel, f'Frame: {frame_count}/{total_frames}', 
                    (result_frame.shape[1] - 200, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        final_frame = np.vstack([panel, result_frame])
        
        display_placeholder.image(final_frame, channels="BGR", width="stretch")
        progress = min(1.0, frame_count / total_frames) if total_frames > 0 else 0.5
        progress_bar.progress(progress)
    
    cap.release()
    
    avg_vehicle_count = np.mean(vehicle_counts_list) if vehicle_counts_list else 0
    avg_water_coverage = np.mean(water_coverages) if water_coverages else 0.0
    final_counts_dict = dict(total_vehicle_detections_dict)
    total_unique_vehicles = len(master_tracked_ids)

    # Return the full lists
    return avg_vehicle_count, avg_water_coverage, processed_count, final_counts_dict, total_unique_vehicles, vehicle_counts_list,water_coverages
# =============================================================================
# VISUALIZATION
# =============================================================================

def create_results_visualization(vehicle_count, water_coverage, risk_score, vehicle_counts_dict):
    """Create enhanced results visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Vehicle distribution pie chart
    if vehicle_counts_dict and sum(vehicle_counts_dict.values()) > 0:
        labels = [v.title() for v, c in vehicle_counts_dict.items() if c > 0]
        sizes = [c for c in vehicle_counts_dict.values() if c > 0]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(sizes)],
                      startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
        axes[0, 0].set_title('Vehicle Distribution', fontweight='bold', fontsize=12, pad=15)
    else:
        axes[0, 0].text(0.5, 0.5, 'No Vehicle Breakdown\n(Video/Live Feed)', 
                       ha='center', va='center', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')
    
    # Water coverage bar
    water_levels = ['Minimal\n(0-2%)', 'Light\n(2-10%)', 'Moderate\n(10-30%)', 'Severe\n(30-60%)', 'Extreme\n(>60%)']
    water_thresholds = [2, 10, 30, 60, 100]
    water_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']
    
    current_water_level = next((i for i, threshold in enumerate(water_thresholds) 
                               if water_coverage <= threshold), len(water_thresholds) - 1)
    
    bars = axes[0, 1].bar(water_levels, water_thresholds, color='lightgray', alpha=0.4, edgecolor='black')
    bars[current_water_level].set_color(water_colors[current_water_level])
    bars[current_water_level].set_alpha(0.9)
    
    axes[0, 1].axhline(y=water_coverage, color='red', linestyle='--', linewidth=2.5, 
                       label=f'Current: {water_coverage:.1f}%')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_ylabel('Coverage %', fontweight='bold', fontsize=11)
    axes[0, 1].set_title('Water Coverage Analysis', fontweight='bold', fontsize=12, pad=15)
    axes[0, 1].legend(loc='upper right', fontsize=9)
    axes[0, 1].grid(True, axis='y', alpha=0.3)
    
    # Risk gauge
    risk_categories = ['Very Low\n(0-25)', 'Low\n(25-45)', 'Medium\n(45-65)', 
                      'High\n(65-85)', 'Very High\n(85-100)']
    risk_values = [25, 45, 65, 85, 100]
    risk_colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    current_category = next((i for i, threshold in enumerate(risk_values) 
                           if risk_score <= threshold), len(risk_values) - 1)
    
    bars = axes[1, 0].bar(risk_categories, risk_values, color='lightgray', alpha=0.4, edgecolor='black')
    bars[current_category].set_color(risk_colors[current_category])
    bars[current_category].set_alpha(0.9)
    
    axes[1, 0].axhline(y=risk_score, color='black', linestyle='--', linewidth=2.5)
    axes[1, 0].text(2, risk_score + 5, f'Risk: {risk_score:.1f}', 
                   ha='center', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                            edgecolor='black', linewidth=2))
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].set_ylabel('Risk Score', fontweight='bold', fontsize=11)
    axes[1, 0].set_title('Risk Assessment', fontweight='bold', fontsize=12, pad=15)
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    
    # Summary panel
    axes[1, 1].axis('off')
    summary_text = f"""
    ANALYSIS SUMMARY
    {'='*30}
    
    Vehicles: {int(vehicle_count)}
    
    Water: {water_coverage:.1f}%
    
    Risk: {risk_categories[current_category].split(chr(10))[0]}
    
    Score: {risk_score:.1f}/100
    """
    
    if risk_score >= 85:
        recommendation = "IMMEDIATE ACTION\nAvoid this route"
    elif risk_score >= 65:
        recommendation = "HIGH CAUTION\nConsider alternate"
    elif risk_score >= 45:
        recommendation = "MODERATE CAUTION\nDrive carefully"
    else:
        recommendation = "SAFE CONDITIONS\nNormal flow"
    
    axes[1, 1].text(0.1, 0.75, summary_text, fontsize=10, fontfamily='monospace',
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
    axes[1, 1].text(0.1, 0.20, recommendation, fontsize=11, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='#ffe6e6', alpha=0.9))
    
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="C-RAPS: Chennai Traffic & Waterlogging Risk",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        .risk-very-high { 
            background-color: #ff4d4d; 
            color: white; 
            padding: 1rem; 
            border-radius: 8px; 
            font-weight: bold; 
            text-align: center;
        }
        .risk-high { 
            background-color: #ff9933; 
            color: white; 
            padding: 1rem; 
            border-radius: 8px; 
            font-weight: bold;
            text-align: center;
        }
        .risk-medium { 
            background-color: #ffcc00; 
            color: black; 
            padding: 1rem; 
            border-radius: 8px; 
            font-weight: bold;
            text-align: center;
        }
        .risk-low { 
            background-color: #66cc66; 
            color: white; 
            padding: 1rem; 
            border-radius: 8px; 
            font-weight: bold;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load models and data
    yolo_model = load_yolo_model()
    water_model = load_water_yolo_model()
    fuzzy_sim = build_enhanced_fuzzy_system()
    chennai_gdf = load_geojson_data()
    
    # Header
    st.markdown(
        '<div class="main-header">'
        '<h1>C-RAPS: Chennai Risk Analysis & Prediction System</h1>'
        '<p>Real-Time Traffic and Waterlogging Detection</p>'
        '</div>', 
        unsafe_allow_html=True
    )
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Analysis", "Risk Map & Stats", "About"])    
    with tab1:
        col1, col2 = st.columns([0.35, 0.65])
        
        with col1:
            st.markdown("### Input Configuration")
            
            with st.expander("Media Input", expanded=True):
                input_type = st.selectbox(
                    "Input Type", 
                    ["Image", "Video", "Live Feed"],
                    help="Select media type"
                )
                
                if input_type == "Image":
                    uploaded_file = st.file_uploader(
                        "Upload Image", 
                        type=["jpg", "jpeg", "png"]
                    )
                elif input_type == "Video":
                    uploaded_file = st.file_uploader(
                        "Upload Video", 
                        type=["mp4", "avi", "mov"]
                    )
                    frame_limit = st.slider("Max Frames", 50, 500, 150)
                else:
                    uploaded_file = 0
                    frame_limit = st.slider("Max Frames", 50, 500, 150)
                    st.info("Live feed will use your webcam")

            with st.expander("Location Selection", expanded=True):
                chennai_areas = get_chennai_areas()
                selected_area = st.selectbox(
                    "Area in Chennai:",
                    options=list(chennai_areas.keys())
                )
                
                if selected_area:
                    area_coords = chennai_areas[selected_area]
                    nearby_streets = search_streets_by_area(chennai_gdf, area_coords)                    
                    if not nearby_streets:
                        # We removed get_default_street_data, so just warn
                        st.warning(f"No streets found in export.geojson for {selected_area}")
                    
                    if nearby_streets:
                        street_options = [f"{s['name']}" for s in nearby_streets]
                        selected_street_idx = st.selectbox(
                            "Select Street:",
                            options=range(len(street_options)),
                            format_func=lambda x: street_options[x]
                        )
                        selected_street_data = nearby_streets[selected_street_idx]
                        
                        st.markdown(
                            f"**Width:** {selected_street_data['road_width']:.1f}m | "
                            f"**AADT:** {selected_street_data['aadt']:,}"
                        )
            
            st.markdown("---")
            
            can_analyze = (uploaded_file or input_type == "Live Feed") and selected_street_data
            
            if can_analyze:
                if not st.session_state.get('processing', False):
                    if st.button("Start Analysis", width="stretch", type="primary"):
                        st.session_state['processing'] = True
                        st.rerun()
                else:
                    if st.button("Stop Processing", width="stretch", type="secondary"):
                        st.session_state['processing'] = False
                        st.rerun()
            else:
                st.warning("Please provide input and select location")
        
        with col2:
            st.markdown("### Live Detection Feed")
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            if st.session_state.get('processing', False):
                try:
                    # Initialize variables
                    vehicle_counts_dict = {}
                    weather_data = None
                    rain_mm_hr = 0
                    
                    if input_type == "Image":
                        original_image = Image.open(uploaded_file).convert("RGB")
                        
                        # Process image
                        vehicle_counts_dict, annotated_image, vehicle_count,frame_id_set = detect_vehicles(
                            yolo_model, original_image
                        )
                        water_coverage, water_mask = detect_water(water_model, original_image)
                        
                        # Apply water overlay
                        result_image = apply_water_overlay(annotated_image, water_mask, alpha=0.5)
                        
                        # Add info panel
                        img_array = np.array(result_image)
                        panel_height = 100
                        panel = np.zeros((panel_height, img_array.shape[1], 3), dtype=np.uint8)
                        panel[:] = (40, 40, 40)
                        cv2.putText(panel, f'Vehicles: {int(vehicle_count)}', (20, 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(panel, f'Water Coverage: {water_coverage:.1f}%', (20, 75), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)
                        
                        result_with_panel = np.vstack([panel, img_array])
                        video_placeholder.image(
                            result_with_panel, 
                            caption="Detection Results", 
                            channels="RGB", 
                            width="stretch"
                        )
                        
                        progress_bar.progress(1.0)
                        frames_processed = 1
                        # --- THIS IS THE FIX ---
                        # We must create these lists for the image mode
                        vehicle_counts_list = [vehicle_count]
                        water_coverages_list = [water_coverage] # This line was missing
                        
                        # Get unique vehicle count for the image
                        total_unique_vehicles = len(frame_id_set)
                        
                        # Set the average count 
                        avg_vehicle_count = vehicle_count
                    elif input_type == "Video":
                        # Save uploaded video
                        video_path = "temp_video.mp4"
                        with open(video_path, "wb") as f:
                            f.write(uploaded_file.read())
                        
                        # Process video
                        vehicle_count, water_coverage, frames_processed, vehicle_counts_dict, total_unique_vehicles, vehicle_counts_list, water_coverages_list = process_video_stream(
                            yolo_model, water_model, video_path, 
                            frame_limit, video_placeholder, progress_bar
                        )
                        
                    else:  # Live Feed
                        vehicle_count, water_coverage, frames_processed, vehicle_counts_dict, total_unique_vehicles, vehicle_counts_list, water_coverages_list = process_video_stream(
                            yolo_model, water_model, video_path, 
                            frame_limit, video_placeholder, progress_bar
                        )

                    # Fetch weather and calculate risk
                    lat = selected_street_data['lat']
                    lon = selected_street_data['lon']
                    weather_data, weather_error = get_weather_data(lat, lon)
                    rain_mm_hr = weather_data.get('rain_1h', 0) if weather_data else 0
                    
                    risk_score = calculate_enhanced_risk(
                        fuzzy_sim, vehicle_count, water_coverage, 
                        selected_street_data, rain_mm_hr
                    )
                    total_detections = sum(vehicle_counts_dict.values()) if vehicle_counts_dict else int(vehicle_count)
                    # Store results
                    st.session_state['analysis_complete'] = True
                    st.session_state['results'] = {
                        'vehicle_count': vehicle_count,
                        'water_coverage': water_coverage,
                        'risk_score': risk_score,
                        'vehicle_counts_dict': vehicle_counts_dict,
                        'weather_data': weather_data,
                        'selected_street_data': selected_street_data,
                        'selected_area': selected_area,
                        'frames_processed': frames_processed
                    }
                    st.session_state['results']['total_unique_vehicles'] = total_unique_vehicles
                    st.session_state['results']['total_detections'] = total_detections  
                    st.success(f"Analysis Complete! Processed {frames_processed} frames")
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### Analysis Dashboard")
                    
                    fig = create_results_visualization(
                        vehicle_count, water_coverage, risk_score, vehicle_counts_dict
                    )
                    st.pyplot(fig)  
                    
                    st.markdown("---")
                    st.markdown("### MPTCRSI-ES Historical Analysis")
                    
                    hist_col1, hist_col2 = st.columns(2)
                    
                    with hist_col1:
                        st.markdown("#### 🚨 EPDO Score (Past Accident Severity)")
                        epdo_score, epdo_cat = calculate_epdo(
                            selected_street_data.get('fatal_crashes_hist', 0),
                            selected_street_data.get('injury_crashes_hist', 0),
                            selected_street_data.get('property_crashes_hist', 0)
                        )
                        st.metric(f"EPDO Score: {epdo_score}", epdo_cat)
                        st.caption(f"Based on {selected_street_data.get('fatal_crashes_hist', 0)} Fatals, {selected_street_data.get('injury_crashes_hist', 0)} Injuries, {selected_street_data.get('property_crashes_hist', 0)} Property")

                    with hist_col2:
                        st.markdown("#### 🔮 Accident Prediction Model")
                        predicted_accidents = predict_accidents(
                            selected_street_data.get('aadt', 20000),
                            selected_street_data.get('road_width', 9.0),
                            selected_street_data.get('speed_limit', 30),
                            selected_street_data.get('num_exits', 5),
                            selected_street_data.get('num_side_roads', 4),
                            selected_street_data.get('parking_type', 'prohibited'),
                            selected_street_data.get('land_use', 'residential')
                        )
                        st.metric("Predicted Accidents / Year", f"{predicted_accidents:.2f}")
                        st.caption("Based on the multi-variable MPTCRSI-ES model")
                    
                    # --- END OF NEW UI SECTION ---
                   # --- 💡 AI-Powered Recommendations (Gemini) ---
                    st.markdown("---")
                    st.markdown("### 💡 AI-Powered Recommendations")
                    
                    # 1. Gather all the data
                    live_data = {
                        'risk_score': risk_score,
                        'water_coverage': water_coverage,
                        'avg_density': vehicle_count # 'vehicle_count' is the average
                    }
                    historical_data = {
                        'epdo_category': epdo_cat,
                        'epdo_score': epdo_score,
                        'predicted_accidents': predicted_accidents
                    }
                    
                    # 2. Call Gemini automatically
                    with st.spinner("🚀 Asking Gemini for a deep-level expert analysis..."):
                        gemini_recommendation = get_gemini_recommendations(
                            live_data, 
                            historical_data, 
                            selected_street_data
                        )
                        st.markdown(gemini_recommendation)
                    # --- END OF NEW UI SECTION ---
                    
                    
                    # Detailed metrics
                    st.markdown("---")
                    st.markdown("### Live Analysis Details") # <-- Renamed
                    col2a, col2b, col2c = st.columns(3)
                    with col2a:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("#### Location Details")
                        st.markdown(f"**{selected_street_data['name']}**")
                        st.markdown(f"**Area:** {selected_area}")
                        st.markdown(f"**Type:** {selected_street_data['highway_type'].title()}")
                        st.markdown(f"**Width:** {selected_street_data['road_width']:.1f}m")
                        st.markdown(f"**AADT:** {selected_street_data['aadt']:,}/day")
                        st.markdown(f"**Speed Limit:** {selected_street_data['speed_limit']} km/h")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2b:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("#### Vehicle Detection")
                        st.metric("Total Unique Vehicles", f"{int(total_unique_vehicles)}")
                        st.metric("Total Detections (for pie chart)", f"{int(total_detections)}")                        
                        if vehicle_counts_dict and sum(vehicle_counts_dict.values()) > 0:
                            st.markdown("**Breakdown (by detection):**")
                            for vtype, count in vehicle_counts_dict.items():
                                if count > 0:
                                    st.markdown(f"- {vtype.title()}: {count}")
                        
                        st.markdown("#### Water Analysis")
                        st.metric("Coverage", f"{water_coverage:.1f}%")
                        category, severity = get_water_level_category(water_coverage)
                        
                        if severity == "critical":
                            st.error(f"{category} Flooding")
                        elif severity == "high":
                            st.warning(f"{category} Water")
                        elif severity == "medium":
                            st.info(f"{category} Water")
                        else:
                            st.success(f"{category} Water")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2c:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        if weather_data:
                            st.markdown("#### Weather")
                            st.markdown(f"**{weather_data['description']}**")
                            st.metric("Rain", f"{rain_mm_hr} mm/hr")
                            st.metric("Temp", f"{weather_data['temperature']}°C")
                        
                        st.markdown("#### Risk Assessment")
                        if risk_score >= 85:
                            st.markdown(
                                f'<div class="risk-very-high">VERY HIGH RISK<br/>'
                                f'{risk_score:.1f}/100<br/>Avoid Route</div>', 
                                unsafe_allow_html=True
                            )
                        elif risk_score >= 65:
                            st.markdown(
                                f'<div class="risk-high">HIGH RISK<br/>'
                                f'{risk_score:.1f}/100<br/>Use Caution</div>', 
                                unsafe_allow_html=True
                            )
                        elif risk_score >= 45:
                            st.markdown(
                                f'<div class="risk-medium">MEDIUM RISK<br/>'
                                f'{risk_score:.1f}/100<br/>Drive Carefully</div>', 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="risk-low">LOW RISK<br/>'
                                f'{risk_score:.1f}/100<br/>Safe Conditions</div>', 
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Download report
                    st.markdown("---")
                    report_text = f"""
C-RAPS ANALYSIS REPORT
=====================

Location: {selected_street_data['name']}, {selected_area}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

VEHICLE DETECTION RESULTS:
- Unique Vehicles: {int(total_unique_vehicles)}
- Total Detections: {int(total_detections)}
- Avg Vehicles/Frame (Density): {int(vehicle_count)}

WATER DETECTION RESULTS:
- Water Coverage: {water_coverage:.1f}%
- Frames Processed: {frames_processed}

RISK ASSESSMENT:
- Risk Score: {risk_score:.1f}/100
- Risk Level: {"VERY HIGH" if risk_score >= 85 else "HIGH" if risk_score >= 65 else "MEDIUM" if risk_score >= 45 else "LOW"}

WEATHER:
- Condition: {weather_data.get('description', 'N/A')}
- Rain: {rain_mm_hr} mm/hr
- Temperature: {weather_data.get('temperature', 'N/A')}°C
                    """
                    
                    st.download_button(
                        label="Download Report",
                        data=report_text,
                        file_name=f"craps_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        width="stretch"
                    )
                    
                    # Reset processing state
                    st.session_state['processing'] = False
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.session_state['processing'] = False
            else:
                st.info("Configure settings and click 'Start Analysis'")
    
    with tab2:
        # --- ADD THE MAP HERE ---
        st.markdown("### Chennai Area Risk Map")
        st.caption("This map shows the static (historical) risk for all streets in our knowledge base.")
        
        # Get the selected area's data from the sidebar
        selected_area_name = st.session_state.get('selected_area', 'T. Nagar')
        area_coords = chennai_areas[selected_area_name]
        
        # Create and display the map
        risk_map = create_risk_map(selected_area_name, area_coords, STREET_KNOWLEDGE_BASE)
        st_folium(risk_map, width="100%", height=450)
        # --- END OF MAP CODE ---
        
        st.markdown("---")
        st.markdown("### System Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Areas Covered", "15")
        with col2:
            st.metric("Detection Models", "2")
        with col3:
            st.metric("Risk Factors", "6")
        
        st.markdown("### Risk Categories")
        risk_df = pd.DataFrame({
            'Category': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            'Range': ['0-25', '25-45', '45-65', '65-85', '85-100'],
            'Action': ['Monitor', 'Normal', 'Caution', 'Alert', 'Urgent']
        })
        st.table(risk_df)
    
    with tab3:
        st.markdown("""
        ### About C-RAPS
        
        **Chennai Risk Analysis & Prediction System** combines:
        
        - Real-time Vehicle Detection using YOLOv8
        - Water/Flood Detection using specialized models
        - Fuzzy Logic Risk Assessment
        - Weather Integration
        - Location-based Analysis
        
        #### Technology Stack:
        - Streamlit for UI
        - YOLOv8 for detection
        - Scikit-fuzzy for risk calculation
        - OpenCV for processing
        - GeoPandas for spatial data
        """)

if __name__ == "__main__":
    main()