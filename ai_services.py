# ai_services.py
# ------------------------------------------------------------
# All Gemini API interactions – bullet-proof edition
# ------------------------------------------------------------

import streamlit as st
import google.generativeai as genai
import json
import re
from typing import Dict, Any, Optional


# ------------------------------------------------------------------
# 1. GEMINI CONFIGURATION (centralised, cached, safe)
# ------------------------------------------------------------------
@st.cache_resource
def _get_gemini_model(model_name: str = "gemini-2.5-flash"):
    """Configure Gemini once and reuse the model object."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in Streamlit secrets")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Gemini configuration failed: {e}")
        return None


# ------------------------------------------------------------------
# 2. HELPER: clean raw Gemini text → pure JSON
# ------------------------------------------------------------------
def _clean_gemini_json(raw_text: str) -> str:
    """
    Strip markdown, code fences, smart quotes, trailing commas
    and return a string that json.loads can parse.
    """
    txt = raw_text.strip()

    # Remove ```json ... ``` or ``` ... ```
    txt = re.sub(r"```[\w]*", "", txt)
    txt = txt.strip("`")

    # Replace fancy quotes
    txt = txt.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Remove trailing commas before } or ]
    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)

    # Grab the first {...} block
    json_match = re.search(r"\{.*\}", txt, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in Gemini response")
    return json_match.group(0)


# ------------------------------------------------------------------
# 3. CITIZEN RECOMMENDATIONS
# ------------------------------------------------------------------
def get_citizen_recommendations(
    live_data: Dict[str, Any],
    historical_data: Dict[str, Any],
    street_data: Dict[str, Any],
    user_profile: str = "commuter",
) -> str:
    """
    Friendly, profile-specific advice for everyday users.
    """
    model = _get_gemini_model()
    if not model:
        return "⚠️ AI recommendations unavailable – check GEMINI_API_KEY"

    profiles = {
        "commuter": """
You are a friendly traffic buddy for Chennai commuters.

Current situation on **{name}**:
- Risk score: {risk:.0f}/100
- Water coverage: {water:.0f}%
- Traffic density: {density} vehicles

Reply in this exact Markdown format (no extra text):

### 🚦 Should I Take This Route?
[YES / NO / MAYBE – one short reason]

### ⏱️ Expected Delay
[estimate]

### 🛵 Best Transport Mode
[Car / Bike / Auto / Walk]

### ⚠️ Watch Out For
- point 1
- point 2

### 🔄 Alternative Routes
[if needed, suggest 1-2 alternatives]
""",
        "delivery_driver": """
You help delivery riders on **{name}**.

Risk: {risk:.0f}/100 | Water: {water:.0f}%

Give a short, actionable list:
1. **Two-wheeler safe?** (Yes/No + reason)
2. **Parking possible?**
3. **Time impact?**
4. **Main hazard?**
""",
        "parent": """
A parent wants to know if **{name}** is safe for kids.

Risk: {risk:.0f}/100 | Traffic: {density} vehicles

Answer:
1. **SAFE for kids?** (Yes/No)
2. **Best walking time?**
3. **Extra precautions?**
""",
    }

    prompt = profiles.get(user_profile, profiles["commuter"]).format(
        name=street_data.get("name", "this street"),
        risk=live_data.get("risk_score", 0),
        water=live_data.get("water_coverage", 0),
        density=live_data.get("avg_density", 0),
    )

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Citizen AI error: {str(e)[:150]}"


# ------------------------------------------------------------------
# 4. OFFICIAL RECOMMENDATIONS
# ------------------------------------------------------------------
def get_official_recommendations(
    live_data: Dict[str, Any],
    historical_data: Dict[str, Any],
    street_data: Dict[str, Any],
) -> str:
    """Technical report for traffic engineers / corporation."""
    model = _get_gemini_model()
    if not model:
        return "⚠️ AI recommendations unavailable – check GEMINI_API_KEY"

    prompt = f"""
You are a senior traffic-safety engineer for Chennai Corporation.

Street: **{street_data.get('name','—')}** ({street_data.get('highway_type','—')})
AADT: {street_data.get('aadt',0):,}
Width: {street_data.get('road_width',0)} m

Historical:
- EPDO: {historical_data.get('epdo_score',0)} ({historical_data.get('epdo_category','—')})
- Predicted accidents/year: {historical_data.get('predicted_accidents',0):.2f}

Live:
- Risk score: {live_data.get('risk_score',0):.1f}/100
- Water coverage: {live_data.get('water_coverage',0):.1f}%
- Density: {live_data.get('avg_density',0)} vehicles/frame

Provide a concise technical report:

### 🚨 Immediate Threat Level
[CRITICAL / HIGH / MEDIUM / LOW – one-sentence justification]

### 📋 Short-term Dispatch Actions
- [action 1]
- [action 2]

### 🏗️ Long-term Infrastructure Fixes
- [fix 1]
- [fix 2]

Use bullet points only. No extra explanations.
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Official AI error: {str(e)[:150]}"


# ------------------------------------------------------------------
# 5. NATURAL-LANGUAGE QUERY PARSER
# ------------------------------------------------------------------
def parse_natural_language_query(user_query: str) -> Optional[Dict[str, Any]]:
    """
    Convert a free-form Chennai traffic question into a structured dict
    for the Nominatim search API.
    """
    model = _get_gemini_model("gemini-2.5-flash")
    if not model:
        return None

    prompt = f"""
Parse this Chennai traffic query into pure JSON (no markdown, no explanations):

"{user_query}"

Return exactly:
{{
    "primary_location": "exact area/street name",
    "secondary_location": null,
    "intent": "check_safety|plan_route|avoid_risk",
    "concerns": ["flooding","traffic","accidents"],   // array, max 3 items
    "search_query": "best query for Nominatim"
}}

Rules:
- Use only double quotes
- No trailing commas
- No code fences
"""

    try:
        response = model.generate_content(prompt)
        json_str = _clean_gemini_json(response.text)
        parsed = json.loads(json_str)

        # Basic sanity checks
        if not isinstance(parsed.get("primary_location"), str):
            return None
        return parsed

    except Exception as e:
        st.warning(f"Query parsing failed: {str(e)[:100]}")
        return None
    
def ai_list_streets_in_area(area_name: str) -> Optional[list]:
    """
    Use Gemini AI to list major streets/roads in a Chennai area.
    Returns a list of streets with their details.
    """
    model = _get_gemini_model("gemini-2.5-flash")
    if not model:
        return None

    prompt = f"""
You are a Chennai geography expert. List the 5-8 MAJOR streets/roads in: "{area_name}"

Return ONLY valid JSON array (no markdown, no explanations):

[
    {{
        "street_name": "exact official name",
        "road_type": "primary|secondary|residential",
        "latitude": 12.9520,
        "longitude": 80.1462,
        "description": "brief 5-word description"
    }},
    ...
]

Rules:
- Focus on MAIN roads (not small lanes)
- Use actual Chennai coordinates (12.85-13.25 lat, 80.10-80.35 lon)
- Include road type (primary=major arterial, secondary=connector, residential=local)
- No trailing commas, only double quotes
- Return 5-8 roads maximum

Examples:
- Chrompet: GST Road, Chromepet Main Road, Pallavaram-Chrompet Road
- T Nagar: Usman Road, Pondy Bazaar, Ranganathan Street
- Anna Nagar: 2nd Avenue, Roundtana Road, Anna Arch
"""

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # ✅ IMPROVED PARSING
        # Remove code fences first
        raw_text = re.sub(r"```[\w]*", "", raw_text).strip("`").strip()
        
        # Try to parse directly
        try:
            streets = json.loads(raw_text)
        except json.JSONDecodeError:
            # If direct parse fails, use cleaner
            json_str = _clean_gemini_json(raw_text)
            streets = json.loads(json_str)
        
        # Handle if it returned object instead of array
        if isinstance(streets, dict):
            streets = streets.get('streets', [streets])
        
        # Validate each street
        valid_streets = []
        for street in streets:
            if (isinstance(street, dict) and
                12.85 <= street.get('latitude', 0) <= 13.25 and
                80.10 <= street.get('longitude', 0) <= 80.35 and
                street.get('street_name')):
                valid_streets.append(street)
        
        return valid_streets if valid_streets else None
        
    except Exception as e:
        st.error(f"❌ AI area street listing failed: {str(e)}")
        import traceback
        st.caption(traceback.format_exc()[:500])  # Show detailed error for debugging
        return None
def ai_find_street_in_chennai(query: str) -> Optional[Dict[str, Any]]:
    """
    Use Gemini AI to intelligently find a street in Chennai with full details.
    Returns coordinates, road type, and area information.
    """
    model = _get_gemini_model("gemini-2.5-flash")
    if not model:
        return None

    prompt = f"""
You are a Chennai geography expert. Find this street/location: "{query}"

Return ONLY valid JSON (no markdown, no explanations):

{{
    "street_name": "exact official name",
    "area": "neighborhood/suburb name",
    "latitude": 13.0827,
    "longitude": 80.2707,
    "road_type": "primary|secondary|residential",
    "confidence": 8,
    "full_address": "complete address"
}}

Rules:
- Use actual Chennai coordinates (12.85-13.25 lat, 80.10-80.35 lon)
- If uncertain, set confidence to 5 or below
- Common roads: Anna Salai, GST Road, OMR, Mount Road, Poonamallee High Road
- No trailing commas, only double quotes
"""

    try:
        response = model.generate_content(prompt)
        json_str = _clean_gemini_json(response.text)
        result = json.loads(json_str)
        
        # Validate result
        if (result.get('confidence', 0) >= 5 and
            12.85 <= result.get('latitude', 0) <= 13.25 and
            80.10 <= result.get('longitude', 0) <= 80.35):
            return result
        
        return None
        
    except Exception as e:
        st.warning(f"AI street search failed: {str(e)[:100]}")
        return None