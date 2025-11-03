"""
Risk Calculation Module
Handles fuzzy logic, EPDO, and accident prediction
"""

import numpy as np
import streamlit as st
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_enhanced_fuzzy_system():
    """Build enhanced fuzzy logic system with corrected priority rules."""
    
    # Antecedents (inputs)
    vehicle_density = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicle_density')
    aadt = ctrl.Antecedent(np.arange(0, 50001, 100), 'aadt')
    road_width = ctrl.Antecedent(np.arange(2, 21, 0.1), 'road_width')
    water_coverage = ctrl.Antecedent(np.arange(0, 101, 1), 'water_coverage')
    speed_limit = ctrl.Antecedent(np.arange(20, 101, 1), 'speed_limit')
    rain_intensity = ctrl.Antecedent(np.arange(0, 51, 1), 'rain_intensity')
    
    # Consequent (output)
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')
    
    # Membership functions
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
    
    # Rules
    rules = [
        ctrl.Rule(water_coverage['heavy'], risk['very_high']),
        ctrl.Rule(rain_intensity['heavy'] & (vehicle_density['medium'] | vehicle_density['high'] | vehicle_density['very_high']), risk['very_high']),
        ctrl.Rule(vehicle_density['very_high'], risk['very_high']),
        ctrl.Rule(speed_limit['high'] & vehicle_density['high'], risk['very_high']),
        ctrl.Rule(water_coverage['moderate'], risk['high']),
        ctrl.Rule(vehicle_density['high'], risk['high']),
        ctrl.Rule(rain_intensity['heavy'], risk['high']),
        ctrl.Rule(vehicle_density['medium'] & road_width['narrow'], risk['high']),
        ctrl.Rule(vehicle_density['high'] & aadt['high'], risk['high']),
        ctrl.Rule(water_coverage['light'], risk['medium']),
        ctrl.Rule(vehicle_density['medium'] & aadt['medium'], risk['medium']),
        ctrl.Rule(aadt['high'] & vehicle_density['low'], risk['medium']),
        ctrl.Rule(speed_limit['medium'] & vehicle_density['high'], risk['medium']),
        ctrl.Rule(vehicle_density['low'] & water_coverage['none'], risk['low']),
        ctrl.Rule(water_coverage['none'] & vehicle_density['very_low'] & rain_intensity['none'], risk['very_low'])
    ]
    
    risk_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(risk_ctrl)


def calculate_enhanced_risk(fuzzy_sim, vehicle_count, water_coverage_pct, street_data, rain_mm_hr):
    """Calculate risk with improved traffic density calculation."""
    aadt = street_data.get('aadt') or 20000
    width = street_data.get('road_width') or 9.0
    speed = street_data.get('speed_limit') or 40
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
    """Calculate Equivalent Property Damage Only (EPDO) score."""
    epdo_score = (property_crashes * 1) + (injury_crashes * 5) + (fatal_crashes * 10)
    
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
    """Predict future accidents using the MPTCRSI-ES model."""
    try:
        a = 6.09e-4
        p = 0.8
        
        beta_speed = {30: 1.8, 40: 2.0, 50: 2.25, 60: 2.85, 70: 1.0}
        β1 = beta_speed.get(speed_limit, 2.0)
        
        if 5.0 <= road_width <= 7.5:
            β2 = 0.83
        elif 8.0 <= road_width <= 8.5:
            β2 = 0.68
        else:
            β2 = 0.80
        
        β3 = 1.0 if 5 <= num_exits <= 40 else 1.2
        
        if num_side_roads == 0:
            β4 = 0.72
        elif num_side_roads <= 5:
            β4 = 0.75
        elif num_side_roads <= 10:
            β4 = 1.0
        else:
            β4 = 1.25
        
        parking_coeffs = {
            "prohibited": 1.19,
            "rarely": 1.0,
            "bays_at_kerb": 1.77
        }
        β5 = parking_coeffs.get(parking_type, 1.0)
        
        land_use_coeffs = {
            "shops": 2.44,
            "apartments": 1.56,
            "industrial": 1.58,
            "residential": 1.58,
            "scattered": 1.0
        }
        β6 = land_use_coeffs.get(land_use, 1.3)
        
        enhanced_accidents = a * (aadt ** p) * β1 * β2 * β3 * β4 * β5 * β6
        return enhanced_accidents
    
    except Exception as e:
        st.warning(f"Could not predict accidents: {e}")
        return 0.0