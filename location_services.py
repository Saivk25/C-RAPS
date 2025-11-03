# location_services.py
# ✅ ENHANCED VERSION - Better street search
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import re


@st.cache_data(ttl=1800)
def search_places_in_chennai(query):
    """
    Smart search: Detects if query is an AREA or STREET, then uses appropriate AI method.
    """
    if not query or len(query) < 2:
        return []

    clean_query = re.sub(r'[^a-zA-Z0-9\s,]', '', query.strip())
    if not clean_query:
        return []

    # ========== STEP 1: Try Nominatim First ==========
    try:
        geolocator = Nominatim(user_agent="craps_chennai_v8_smart", timeout=10)
        results = []
        seen_locations = set()

        search_queries = [
            f"{clean_query}, Chennai",
            f"{clean_query} road, Chennai",
            f"{clean_query} street, Chennai",
        ]

        for search_q in search_queries:
            try:
                locations = geolocator.geocode(
                    search_q,
                    exactly_one=False,
                    limit=5,
                    addressdetails=True
                )
                
                if locations:
                    for loc in locations:
                        lat, lon = float(loc.latitude), float(loc.longitude)
                        if not (12.85 <= lat <= 13.25 and 80.10 <= lon <= 80.35):
                            continue
                        
                        addr = loc.raw.get('address', {})
                        road_name = (
                            addr.get('road') or 
                            addr.get('suburb') or 
                            addr.get('neighbourhood') or
                            None
                        )
                        
                        if not road_name:
                            continue
                            
                        location_key = f"{road_name.lower()}_{lat:.3f}"
                        if location_key in seen_locations:
                            continue
                        seen_locations.add(location_key)
                        
                        area = addr.get('suburb') or addr.get('neighbourhood') or 'Chennai'
                        
                        results.append({
                            'display_name': f"{road_name} → {area}",
                            'full_address': loc.address,
                            'lat': lat,
                            'lon': lon,
                            'road_name': road_name,
                            'area': area
                        })
                        
                        if len(results) >= 8:
                            break
            except:
                continue
            
            if len(results) >= 5:
                break
        
        if results:
            st.success(f"✅ Found {len(results)} location(s) from map")
            return results[:8]
        
    except Exception as e:
        st.caption(f"Map search issue: {str(e)[:60]}")

    # ========== STEP 2: AI FALLBACK - DETECT INTENT ==========
    st.caption("🤖 Asking AI...")
    
    try:
        from ai_services import ai_find_street_in_chennai, ai_list_streets_in_area
        
        # 🔍 SMART DETECTION: Is it an area or specific street?
        area_keywords = ['area', 'nagar', 'puram', 'pet', 'town', 'colony']
        is_area_query = any(keyword in clean_query.lower() for keyword in area_keywords)
        
        # Also check if query is short (likely an area name)
        word_count = len(clean_query.split())
        if word_count <= 2 and 'road' not in clean_query.lower():
            is_area_query = True
        
        # ========== BRANCH A: AREA SEARCH ==========
        if is_area_query:
            st.info(f"🗺️ Searching for streets in '{clean_query}' area...")
            
            ai_streets = ai_list_streets_in_area(clean_query)
            
            if ai_streets and len(ai_streets) > 0:
                results = []
                for street in ai_streets:
                    results.append({
                        'display_name': f"{street['street_name']} → {clean_query.title()} (AI)",
                        'full_address': f"{street['street_name']}, {clean_query.title()}, Chennai",
                        'lat': street['latitude'],
                        'lon': street['longitude'],
                        'road_name': street['street_name'],
                        'area': clean_query.title()
                    })
                
                st.success(f"✅ AI found {len(results)} major roads in {clean_query.title()}")
                return results
        
        # ========== BRANCH B: SPECIFIC STREET SEARCH ==========
        else:
            st.info(f"🛣️ AI searching for specific street: '{clean_query}'")
            
            ai_result = ai_find_street_in_chennai(clean_query)
            
            if ai_result and ai_result.get('confidence', 0) >= 5:
                st.success(f"✅ AI found: {ai_result['street_name']}")
                
                return [{
                    'display_name': f"{ai_result['street_name']} → {ai_result['area']} (AI)",
                    'full_address': ai_result.get('full_address', f"{ai_result['street_name']}, Chennai"),
                    'lat': ai_result['latitude'],
                    'lon': ai_result['longitude'],
                    'road_name': ai_result['street_name'],
                    'area': ai_result['area']
                }]
            else:
                st.warning(f"⚠️ AI found '{clean_query}' but low confidence")
                
    except Exception as ai_err:
        st.error(f"AI error: {str(ai_err)[:80]}")

    return []


def reverse_geocode(lat, lon):
    """Get address from coordinates"""
    try:
        geolocator = Nominatim(user_agent="craps_chennai_v7_streets", timeout=10)
        loc = geolocator.reverse((lat, lon), addressdetails=True)
        
        if loc:
            addr = loc.raw.get('address', {})
            return {
                'road': addr.get('road', 'Unknown Road'),
                'area': addr.get('suburb', 'Chennai'),
                'full_address': loc.address
            }
    except Exception as e:
        st.caption(f"Reverse geocode: {e}")
    
    return {
        'road': 'Unknown Road',
        'area': 'Chennai',
        'full_address': 'Chennai, Tamil Nadu'
    }


def validate_chennai_coordinates(lat, lon):
    """Check if coordinates are within Chennai metropolitan bounds"""
    if 12.85 <= lat <= 13.25 and 80.10 <= lon <= 80.35:
        return True, "✅ Valid Chennai location"
    return False, "⚠️ Outside Chennai bounds"