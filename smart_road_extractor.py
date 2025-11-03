import io
import json
import re
from typing import Dict, List, Optional, Tuple

import cv2
import google.generativeai as genai
import numpy as np
import streamlit as st
from PIL import Image


class SmartRoadDataExtractor:
    """Extract real road data by analyzing multiple frames intelligently"""
    
    def __init__(self, api_key: str):
        """Initialize with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def select_best_frames(self, source, num_frames: int = 3) -> List[np.ndarray]:
        """
        Select the BEST frames from video for analysis.
        Takes frames from: Start, Middle, and End (or multiple middle points)
        
        Args:
            source: Video path, 0 for webcam, or PIL Image
            num_frames: Number of frames to extract (default 3)
            
        Returns:
            List of numpy arrays (frames)
        """
        frames = []
        
        # Handle Image input
        if isinstance(source, Image.Image):
            return [np.array(source)]
        
        # Handle Video/Live Feed
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.error("Failed to open video source")
            return []
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:  # Live feed
            total_frames = 300  # Assume 10 seconds at 30fps
        
        # Calculate frame positions (start, middle, end)
        if num_frames == 1:
            positions = [total_frames // 2]  # Just middle
        elif num_frames == 2:
            positions = [total_frames // 4, 3 * total_frames // 4]
        else:
            positions = [int(total_frames * i / (num_frames + 1)) for i in range(1, num_frames + 1)]
        
        # Extract frames
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                # Resize for faster processing
                height, width = frame.shape[:2]
                if width > 800:
                    scale = 800 / width
                    frame = cv2.resize(frame, (800, int(height * scale)))
                frames.append(frame)
            
            if len(frames) >= num_frames:
                break
        
        cap.release()
        return frames
    
    def score_frame_quality(self, frame: np.ndarray) -> float:
        """
        Score a frame's quality for road analysis (0-100).
        Higher score = better for extracting road data
        
        Criteria:
        - Sharpness (Laplacian variance)
        - Brightness (not too dark/bright)
        - Information content (edge density)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, laplacian_var / 10)  # Normalize
        
        # 2. Brightness (optimal around 127)
        brightness = np.mean(gray)
        brightness_score = 100 - abs(brightness - 127) * 0.78  # Penalty for deviation
        
        # 3. Edge content (Canny edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size * 100
        edge_score = min(100, edge_density * 2)
        
        # Weighted average
        total_score = (sharpness_score * 0.5 + brightness_score * 0.3 + edge_score * 0.2)
        
        return total_score
    
    def analyze_road_image(self, image) -> Dict:
        """
        Analyze a SINGLE road image to extract measurements.
        (Same as before, but now it's used for the BEST frame)
        """
        # Convert image to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        prompt = """Analyze this road image and extract the following information as accurately as possible:

1. **Road Width**: Estimate the total road width in meters (including all lanes). Consider:
- Number of visible lanes (typically 3-4m per lane)
- Shoulder/parking space
- Common Indian road standards

2. **Number of Lanes**: Count distinct traffic lanes (not including parking/shoulders)

3. **Speed Limit**: Estimate appropriate speed limit in km/h based on:
- Road type (residential/arterial/highway)
- Lane width and quality
- Visible traffic patterns
- Indian road standards (30-70 km/h typical)

4. **Land Use**: Identify the surrounding area type:
- residential: houses, apartments
- commercial: shops, offices, malls
- industrial: factories, warehouses
- mixed: combination of above

5. **Parking Type**: Identify parking situation:
- prohibited: no parking visible/allowed
- bays_at_kerb: marked parking spaces along road
- parallel: parallel parking along sides
- rarely: occasional/informal parking

6. **Road Surface Quality**: Rate 1-10 (10 = excellent, 1 = very poor)

7. **Visible Infrastructure**: Count visible:
- Street lights
- Traffic signals
- Speed breakers/humps
- Pedestrian crossings

8. **Traffic Volume Estimate**: Based on visible vehicles and road type, estimate AADT:
- Light: 5,000-15,000
- Medium: 15,000-35,000
- Heavy: 35,000-60,000

9. **Visibility Quality**: How clear is this frame for analysis? (1-10)

Return ONLY a valid JSON object (no markdown, no explanations, no extra text):
{
    "road_width": <number in meters>,
    "num_lanes": <integer>,
    "speed_limit": <integer in km/h>,
    "land_use": "<residential|commercial|industrial|mixed>",
    "parking_type": "<prohibited|bays_at_kerb|parallel|rarely>",
    "surface_quality": <1-10>,
    "street_lights": <integer count>,
    "traffic_signals": <integer count>,
    "speed_breakers": <integer count>,
    "pedestrian_crossings": <integer count>,
    "aadt_estimate": <integer>,
    "aadt_category": "<light|medium|heavy>",
    "visibility_quality": <1-10>,
    "analysis_notes": "<brief observations>"
}"""

        try:
            response = self.model.generate_content([prompt, image])
            text = response.text.strip()

            # Remove ALL markdown code fences
            text = re.sub(r'```[^`\n]*', '', text)
            text = text.strip()

            # Extract JSON object only
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in response")
            text = json_match.group(0)

            # Clean problematic characters
            text = text.replace("'", '"')
            text = text.replace(''', '"').replace(''', '"')
            text = text.replace('"', '"').replace('"', '"')
            
            # Remove trailing commas
            text = re.sub(r',(\s*[}\]])', r'\1', text)

            # Parse
            data = json.loads(text)
            st.write("DEBUG: Parsed AI data:", data)  # ← Add this
            with st.expander("🔍 Raw AI Response (Parsed JSON)", expanded=False):
                st.json(data)
                st.caption(f"Keys returned: {list(data.keys())}")
                st.caption(f"AADT value: {data.get('aadt_estimate')} (type: {type(data.get('aadt_estimate'))})")

            validated_data = self._validate_extracted_data(data)
            return validated_data

        except json.JSONDecodeError as je:
            st.warning(f"AI returned invalid JSON: {str(je)[:100]}")
            return self._get_fallback_data()
            
        except Exception as e:
            st.warning(f"AI analysis failed: {str(e)}")
            return self._get_fallback_data()
    
    def analyze_multiple_frames(self, source, num_frames: int = 3) -> Tuple[Dict, int]:
        """
        SMART ANALYSIS: Extract frames, score them, analyze the best one.
        """
        st.info(f"Extracting {num_frames} frames for smart analysis...")
        
        # Step 1: Extract candidate frames
        frames = self.select_best_frames(source, num_frames)
        
        if not frames:
            st.error("No frames extracted!")
            return self._get_fallback_data(), -1
        
        st.success(f"Extracted {len(frames)} frames")
        
        # Step 2: Score each frame
        st.info("Scoring frame quality...")
        frame_scores = []
        for i, frame in enumerate(frames):
            quality_score = self.score_frame_quality(frame)
            frame_scores.append((i, quality_score, frame))
            st.caption(f"Frame {i+1}: Quality Score = {quality_score:.1f}/100")
        
        # Step 3: Pick the best frame
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_score, best_frame = frame_scores[0]
        
        st.success(f"Selected Frame {best_idx+1} (Quality: {best_score:.1f}/100)")
        
        # Step 4: Display the chosen frame
        st.image(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB), 
                caption=f"Best Frame (#{best_idx+1})", 
                width=400)
        
        # Step 5: Analyze the best frame with AI
        st.info("Running AI analysis on best frame...")
        ai_data = self.analyze_road_image(best_frame)
        
        # Add metadata
        ai_data['frame_used'] = best_idx + 1
        ai_data['frame_quality_score'] = best_score
        ai_data['total_frames_considered'] = len(frames)
        
        return ai_data, best_idx
    
    # ✅ FIX: Properly indent this method INSIDE the class
    def enhance_street_data_with_smart_vision(self, street_data: Dict, source, num_frames: int = 3) -> Dict:
        """
        COMPLETE WORKFLOW: Analyze multiple frames and enhance street data.
        """
        try:
            # Run smart multi-frame analysis
            ai_data, best_frame_idx = self.analyze_multiple_frames(source, num_frames)
            
            # Merge with existing data
            enhanced_data = street_data.copy()
            
            # Update physical measurements from AI
            enhanced_data.update({
                'road_width': ai_data['road_width'],
                'speed_limit': ai_data['speed_limit'],
                'land_use': ai_data['land_use'],
                'parking_type': ai_data['parking_type'],
                'aadt': ai_data['aadt_estimate'],
                'surface_quality': ai_data['surface_quality'],
                'num_lanes': ai_data['num_lanes'],
                'infrastructure': {
                    'street_lights': ai_data['street_lights'],
                    'traffic_signals': ai_data['traffic_signals'],
                    'speed_breakers': ai_data['speed_breakers'],
                    'pedestrian_crossings': ai_data['pedestrian_crossings']
                },
                'ai_confidence': ai_data['visibility_quality'],
                'ai_notes': ai_data['analysis_notes'],
                'frame_used': ai_data['frame_used'],
                'frame_quality': ai_data['frame_quality_score'],
                'frames_analyzed': ai_data['total_frames_considered'],
                'data_source': 'ai_enhanced_multi_frame'
            })

            # CRITICAL: PASS SUCCESS FLAG
            enhanced_data['ai_extraction_success'] = ai_data.get('ai_extraction_success', True)
            
            return enhanced_data
            
        except Exception as e:
            st.error(f"Multi-frame analysis failed: {e}")
            fallback = street_data.copy()
            fallback['ai_extraction_success'] = False
            fallback['data_source'] = 'fallback_kb'
            return fallback
    
    def _validate_extracted_data(self, data: Dict) -> Dict:
        raw_aadt = data.get('aadt_estimate', 20000)
        try:
            aadt_value = int(raw_aadt)
        except (ValueError, TypeError):
            aadt_value = 20000  # Default fallback
    
        validated = {
        'aadt_estimate': max(1000, min(100000, aadt_value)),
            'road_width': float(max(3.0, min(30.0, data.get('road_width', 9.0)))),
            'num_lanes': int(max(1, min(8, data.get('num_lanes', 2)))),
            'speed_limit': int(max(20, min(80, data.get('speed_limit', 40)))),
            'land_use': data.get('land_use', 'residential').lower(),
            'parking_type': data.get('parking_type', 'rarely').lower(),
            'surface_quality': int(max(1, min(10, data.get('surface_quality', 5)))),
            'street_lights': int(max(0, data.get('street_lights', 0))),
            'traffic_signals': int(max(0, data.get('traffic_signals', 0))),
            'speed_breakers': int(max(0, data.get('speed_breakers', 0))),
            'pedestrian_crossings': int(max(0, data.get('pedestrian_crossings', 0))),
            #'aadt_estimate': int(max(1000, min(100000, data.get('aadt_estimate', 20000)))),
            'aadt_category': data.get('aadt_category', 'medium').lower(),
            'visibility_quality': int(max(1, min(10, data.get('visibility_quality', 5)))),
            'analysis_notes': data.get('analysis_notes', 'AI-extracted data'),
            'ai_extraction_success': True
        }
        
        # Ensure valid enums
        if validated['land_use'] not in ['residential', 'commercial', 'industrial', 'mixed']:
            validated['land_use'] = 'residential'
        
        if validated['parking_type'] not in ['prohibited', 'bays_at_kerb', 'parallel', 'rarely']:
            validated['parking_type'] = 'rarely'
        
        return validated
    
    def _get_fallback_data(self) -> Dict:
        """Return safe fallback data if AI extraction fails"""
        return {
            'road_width': 9.0,
            'num_lanes': 2,
            'speed_limit': 40,
            'land_use': 'residential',
            'parking_type': 'rarely',
            'surface_quality': 5,
            'street_lights': 0,
            'traffic_signals': 0,
            'speed_breakers': 0,
            'pedestrian_crossings': 0,
            'aadt_estimate': 20000,
            'aadt_category': 'medium',
            'visibility_quality': 3,
            'analysis_notes': 'Fallback default values',
            'frame_used': 0,
            'frame_quality': 0,
            'frames_analyzed': 0,
            'data_source': 'fallback_kb',
            'ai_extraction_success': False
        }


def create_enhanced_comparison_ui(original_data: Dict, ai_data: Dict):
    """Enhanced comparison UI with frame selection info"""
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### Original Data")
        st.metric("Road Width", f"{original_data.get('road_width', 'N/A')} m")
        st.metric("Speed Limit", f"{original_data.get('speed_limit', 'N/A')} km/h")
        
        # ✅ FIX: Use 'aadt' (not 'aadt_estimate')
        aadt_val = original_data.get('aadt', 0)
        if isinstance(aadt_val, (int, float)) and aadt_val > 0:
            st.metric("AADT", f"{int(aadt_val):,}")
        else:
            st.metric("AADT", "N/A")
        
        st.caption("Source: Knowledge Base")
    
    with col2:
        st.markdown("### AI-Extracted Data")
        st.metric("Road Width", f"{ai_data.get('road_width', 'N/A')} m")
        st.metric("Speed Limit", f"{ai_data.get('speed_limit', 'N/A')} km/h")
        
        # ✅ FIX: Try both 'aadt' and 'aadt_estimate' keys
        aadt_val = ai_data.get('aadt') or ai_data.get('aadt_estimate', 0)
        if isinstance(aadt_val, (int, float)) and aadt_val > 0:
            st.metric("AADT", f"{int(aadt_val):,}")
        else:
            st.metric("AADT", "N/A")
        
        # Show frame metadata
        if 'frame_used' in ai_data:
            st.caption(f"Used Frame #{ai_data['frame_used']} of {ai_data.get('frames_analyzed', 1)}")
            st.caption(f"Frame Quality: {ai_data.get('frame_quality', 0):.1f}/100")
    
    with col3:
        st.markdown("### Analysis Quality")
        
        # ✅ FIX: Try both 'ai_confidence' and 'visibility_quality' keys
        confidence = ai_data.get('ai_confidence') or ai_data.get('visibility_quality', 0)
        
        if confidence >= 7:
            st.success("High Confidence")
        elif confidence >= 5:
            st.info("Medium Confidence")
        else:
            st.warning("Low Confidence")
        
        # ✅ FIX: Try both 'ai_notes' and 'analysis_notes' keys
        notes = ai_data.get('ai_notes') or ai_data.get('analysis_notes', 'No notes')
        st.caption(notes)
        
        # Show quality improvement message
        if ai_data.get('frame_quality', 0) > 60:
            st.success("Smart frame selection improved quality!")