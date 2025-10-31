"""
Vehicle Detection Module
Handles YOLO-based vehicle detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st


@st.cache_resource
def load_yolo_model():
    """Load YOLO model for vehicle detection."""
    try:
        yolo_model = YOLO('yolov8n.pt')
        yolo_model.overrides['conf'] = 0.3
        yolo_model.overrides['iou'] = 0.5
        return yolo_model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None


# In vehicle_detection.py

# In vehicle_detection.py

def detect_vehicles(yolo_model, image_or_frame, water_mask):
    """
    Enhanced vehicle detection using YOLOv8 tracking.
    NOW includes dynamic risk (checks if vehicle is in water).
    
    Returns:
        tuple: (vehicle_counts_dict, annotated_image, total_vehicles_in_frame, frame_id_set)
    """
    if yolo_model is None:
        return {}, image_or_frame, 0, set()
    
    try:
        image_np = np.array(image_or_frame)
        vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        results = yolo_model.track(
            image_np, 
            conf=0.25, 
            iou=0.5, 
            verbose=False, 
            classes=list(vehicle_classes.keys()),
            persist=True
        )
        
        vehicle_counts = {v: 0 for v in vehicle_classes.values()}
        annotated_image = image_np.copy()
        frame_id_set = set()
        
        # Check if the water_mask is valid
        has_water_mask = water_mask is not None and water_mask.shape[:2] == image_np.shape[:2]

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                    
                track_id = int(box.id[0])
                frame_id_set.add(track_id)
                
                class_id = int(box.cls[0])
                if class_id in vehicle_classes:
                    vehicle_type = vehicle_classes[class_id]
                    vehicle_counts[vehicle_type] += 1
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # --- DYNAMIC RISK "WOW" FACTOR ---
                    is_in_water = False
                    if has_water_mask:
                        # Check the bottom-center of the vehicle
                        check_x = int((x1 + x2) / 2)
                        check_y = int(y2 - 5) # 5 pixels up from the bottom
                        
                        # Ensure coordinates are in bounds
                        if 0 <= check_y < water_mask.shape[0] and 0 <= check_x < water_mask.shape[1]:
                            if water_mask[check_y, check_x] > 0:
                                is_in_water = True
                    
                    # Default colors
                    base_colors = {
                        'car': (255, 0, 0), # Blue
                        'motorcycle': (0, 255, 0), # Green
                        'bus': (255, 100, 0), # Light Blue
                        'truck': (255, 255, 0) # Yellow
                    }
                    
                    if is_in_water:
                        color = (0, 0, 255) # --- RED for DANGER
                        label = f'ID: {track_id} {vehicle_type} (FLOODED)'
                        thickness = 3
                    else:
                        color = base_colors.get(vehicle_type, (255, 255, 255))
                        label = f'ID: {track_id} {vehicle_type}'
                        thickness = 2
                    # --- END OF "WOW" FACTOR ---

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(
                        annotated_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness
                    )
        
        total_vehicles_in_frame = len(frame_id_set)
        return vehicle_counts, annotated_image, total_vehicles_in_frame, frame_id_set
    
    except Exception as e:
        st.error(f"Vehicle detection error: {e}")
        return {}, np.array(image_or_frame), 0, set()