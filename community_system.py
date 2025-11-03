"""
Community Video Linking System
Allows users to share analysis results for locations
"""

import json
import hashlib
from datetime import datetime, timedelta
import streamlit as st


def generate_location_key(lat, lon, street_name):
    """Create unique key for location (within ~100m radius)"""
    lat_rounded = round(lat, 3)
    lon_rounded = round(lon, 3)
    street_hash = hashlib.md5(street_name.lower().encode()).hexdigest()[:8]
    return f"loc:{lat_rounded}:{lon_rounded}:{street_hash}"


def save_video_analysis(location_data, analysis_results, video_metadata):
    """
    Save analysis results linked to a specific location
    
    Args:
        location_data: dict with name, lat, lon, area
        analysis_results: dict with risk_score, vehicle_count, water_coverage, etc.
        video_metadata: dict with frames_processed, duration, source
    
    Returns:
        tuple: (success: bool, location_key: str)
    """
    location_key = generate_location_key(
        location_data['lat'],
        location_data['lon'],
        location_data['name']
    )
    
    analysis_package = {
        'timestamp': datetime.now().isoformat(),
        'location': {
            'name': location_data['name'],
            'area': location_data.get('area', 'Unknown'),
            'lat': location_data['lat'],
            'lon': location_data['lon']
        },
        'analysis': {
            'risk_score': float(analysis_results['risk_score']),
            'vehicle_count': int(analysis_results['vehicle_count']),
            'water_coverage': float(analysis_results['water_coverage']),
            'vehicle_breakdown': analysis_results.get('vehicle_counts_dict', {}),
            'weather': analysis_results.get('weather_data', {})
        },
        'metadata': {
            'frames_processed': video_metadata['frames_processed'],
            'duration': video_metadata.get('duration', 0),
            'source': video_metadata.get('source', 'user_upload')
        },
        'historical': {
            'epdo_score': analysis_results.get('epdo_score', 0),
            'predicted_accidents': analysis_results.get('predicted_accidents', 0)
        }
    }
    
    try:
        # Note: This uses Streamlit's session state as fallback
        # In production, use actual persistent storage
        if 'community_cache' not in st.session_state:
            st.session_state.community_cache = {}
        
        st.session_state.community_cache[location_key] = analysis_package
        
        # Also maintain history
        history_key = f"{location_key}:history"
        if history_key not in st.session_state.community_cache:
            st.session_state.community_cache[history_key] = []
        
        st.session_state.community_cache[history_key].append({
            'timestamp': datetime.now().isoformat(),
            'risk_score': float(analysis_results['risk_score']),
            'contributor': 'anonymous'
        })
        
        # Keep only last 10
        st.session_state.community_cache[history_key] = \
            st.session_state.community_cache[history_key][-10:]
        
        return True, location_key
        
    except Exception as e:
        st.error(f"Could not save analysis: {e}")
        return False, None


def fetch_cached_analysis(location_data, max_age_minutes=30):
    """
    Check if there's a recent analysis for this location
    
    Args:
        location_data: dict with name, lat, lon
        max_age_minutes: int, how old the cached data can be
    
    Returns:
        dict with keys: found, age_minutes, data, is_stale
    """
    location_key = generate_location_key(
        location_data['lat'],
        location_data['lon'],
        location_data['name']
    )
    
    try:
        if 'community_cache' not in st.session_state:
            return {'found': False}
        
        if location_key not in st.session_state.community_cache:
            return {'found': False}
        
        analysis_package = st.session_state.community_cache[location_key]
        
        # Check freshness
        timestamp = datetime.fromisoformat(analysis_package['timestamp'])
        age_minutes = (datetime.now() - timestamp).total_seconds() / 60
        
        if age_minutes <= max_age_minutes:
            return {
                'found': True,
                'age_minutes': age_minutes,
                'data': analysis_package,
                'is_stale': False
            }
        else:
            return {
                'found': True,
                'age_minutes': age_minutes,
                'data': analysis_package,
                'is_stale': True
            }
    
    except Exception as e:
        st.warning(f"Could not fetch cached analysis: {e}")
        return {'found': False}


def get_analysis_history(location_data):
    """Get historical trend for a location"""
    location_key = generate_location_key(
        location_data['lat'],
        location_data['lon'],
        location_data['name']
    )
    
    history_key = f"{location_key}:history"
    
    try:
        if 'community_cache' not in st.session_state:
            return []
        
        return st.session_state.community_cache.get(history_key, [])
    except:
        return []


def display_cached_analysis_ui(cached_result):
    """
    Display cached analysis in UI
    
    Args:
        cached_result: dict from fetch_cached_analysis()
    """
    if not cached_result['found']:
        return
    
    data = cached_result['data']
    age_minutes = cached_result['age_minutes']
    
    if cached_result.get('is_stale', False):
        st.warning(f"⏰ **Cached Analysis Available** (from {age_minutes:.0f} mins ago - may be outdated)")
    else:
        st.info(f"✅ **Recent Analysis Available** (from {age_minutes:.0f} mins ago)")
    
    with st.expander("📊 View Cached Analysis", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Risk Score", 
                f"{data['analysis']['risk_score']:.0f}/100",
                delta=None
            )
        
        with col2:
            st.metric(
                "Vehicles Detected",
                data['analysis']['vehicle_count']
            )
        
        with col3:
            st.metric(
                "Water Coverage",
                f"{data['analysis']['water_coverage']:.1f}%"
            )
        
        st.caption(f"📍 Location: {data['location']['name']}, {data['location']['area']}")
        st.caption(f"🕒 Analyzed: {datetime.fromisoformat(data['timestamp']).strftime('%I:%M %p, %d %b')}")
        st.caption(f"📹 Source: {data['metadata']['source']} ({data['metadata']['frames_processed']} frames)")
        
        # Option to use cached data or reanalyze
        col_action1, col_action2 = st.columns(2)
        
        with col_action1:
            if st.button("✅ Use This Analysis", key="use_cached"):
                st.session_state['use_cached_analysis'] = True
                st.session_state['cached_data'] = data
                st.rerun()
        
        with col_action2:
            if st.button("🔄 Analyze Fresh", key="analyze_fresh"):
                st.session_state['use_cached_analysis'] = False
                st.info("👇 Upload new video or use live feed below")