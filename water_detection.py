"""
Water/Flood Detection Module
Handles YOLO-based water segmentation
"""

import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st


@st.cache_resource
def load_water_yolo_model():
    """Load pretrained YOLOv8n model for water segmentation."""
    try:
        water_model = YOLO('best.pt')
        return water_model
    except Exception as e:
        st.error(
            f"Failed to load water YOLO model: {e}. "
            "Please download best.pt from "
            "https://github.com/duchieu260503/Flood-detection"
        )
        return None


def detect_water(water_model, image_or_frame, min_area_threshold=100):
    """
    Use pretrained YOLOv8n for water segmentation with improved filtering.
    
    Args:
        water_model: Loaded water detection model
        image_or_frame: Input image or video frame
        min_area_threshold: Minimum contiguous area to consider as water (pixels)
        
    Returns:
        tuple: (water_coverage_percentage, water_mask)
    """
    if water_model is None:
        image_np = np.array(image_or_frame)
        return 0.0, np.zeros(image_np.shape[:2], dtype=np.uint8)
    
    try:
        image_np = np.array(image_or_frame)
        results = water_model.predict(
            image_np, 
            conf=0.35,  # Increased confidence threshold to reduce false positives
            iou=0.5, 
            verbose=False
        )
        
        water_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        
        if results and len(results) > 0 and results[0].masks is not None:
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))
                
                # Apply threshold
                binary_mask = (mask_resized > 0.6).astype(np.uint8) * 255
                
                # Remove small isolated regions (likely false positives on vehicles)
                kernel = np.ones((5, 5), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours and filter by area
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_area_threshold:
                        cv2.drawContours(water_mask, [contour], -1, 255, -1)
        
        # Calculate coverage focusing on lower portion of image (where road typically is)
        height = image_np.shape[0]
        lower_region = water_mask[int(height * 0.3):, :]  # Focus on lower 70% of image
        
        water_pixels = np.sum(lower_region > 0)
        total_pixels = lower_region.shape[0] * lower_region.shape[1]
        water_coverage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        
        # Cap unrealistic values
        water_coverage = min(water_coverage, 100.0)
        
        return water_coverage, water_mask
        
    except Exception as e:
        st.error(f"Water detection error: {e}")
        image_np = np.array(image_or_frame)
        return 0.0, np.zeros(image_np.shape[:2], dtype=np.uint8)


def apply_water_overlay(image, water_mask, alpha=0.6):
    """
    Apply blue overlay to show detected water areas.
    
    Args:
        image: Original image (numpy array)
        water_mask: Binary mask of water areas
        alpha: Transparency for overlay (0-1)
        
    Returns:
        Image with water overlay applied (numpy array)
    """
    if water_mask is None or water_mask.size == 0 or np.sum(water_mask) == 0:
        return np.array(image)
    
    img_array = np.array(image).copy()
    water_colored = np.zeros_like(img_array)
    water_colored[:, :] = [255, 150, 0]  # Blue color in BGR
    
    # Apply overlay only to water regions
    img_array[water_mask > 0] = cv2.addWeighted(
        img_array[water_mask > 0], 1 - alpha,
        water_colored[water_mask > 0], alpha, 0
    )
    
    return img_array


def get_water_level_category(water_coverage):
    """
    Categorize water coverage level.
    
    Args:
        water_coverage: Water coverage percentage (0-100)
        
    Returns:
        tuple: (category_name, severity_level)
    """
    if water_coverage > 60:
        return "EXTREME", "critical"
    elif water_coverage > 30:
        return "SEVERE", "high"
    elif water_coverage > 10:
        return "MODERATE", "medium"
    elif water_coverage > 2:
        return "LIGHT", "low"
    else:
        return "MINIMAL", "safe"