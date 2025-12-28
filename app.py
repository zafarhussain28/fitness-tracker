import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import io
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import requests
import os
import json
from google import genai
from google.genai.errors import APIError

# ----------------------------------------------------
# MediaPipe initializations (UNCHANGED)
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(layout="wide", page_title="SEGMENT FIT TRAINER", page_icon="⚡")
st.markdown(
    """
    <style>
    .typing-title {
        font-size: 48px;
        font-weight: 800;
        font-family: 'Arial', sans-serif;
        color: #F8E5BC;
        white-space: nowrap;
        overflow: hidden;
        width: 0;
        animation: typing 3s steps(30, end) forwards;
        letter-spacing: 2px;
        border-right: none !important; /* Remove blinking cursor */
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 670px } /* Bigger width → full sentence fits */
    }
    </style>

    <div style="display:flex; justify-content:center; margin-top:20px;">
        <div class="typing-title">
            SEGMENT FIT TRAINER
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ----------------------------------------------------
# 🎨 UI/UX DESIGN LAYER (NEW)
# ----------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        

        /* GOOGLE FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Teko:wght@400;600;700&family=Inter:wght@300;400;600&display=swap');
        /* THEME COLORS */
        :root {
            --primary-color: moccasin;
            --secondary-color: #1a1a1a;
            --text-main: #ffffff;
            --text-muted: #a0a0a0;
            --card-bg: rgba(255, 255, 255, 0.05);
        }

        /* GLOBAL HEADINGS */
        h1, h2, h3 {
            font-family: 'Teko', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        h1 { font-size: 4rem !important; font-weight: 700 !important; }
        h2 { font-size: 2.5rem !important; color: var(--primary-color) !important; }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #050505;
            border-right: 1px solid #222;
        }

        /* --- CUSTOM INPUT FIELDS (Age, Weight, Height, Gender) --- */
        /* Targets the internal box of number inputs and select boxes */
        div[data-baseweb="input"] > div, 
        div[data-baseweb="select"] > div {
            background-color: rgba(0, 0, 0, 0.45) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 8px !important;
            color: white !important;
        }
        
        /* Fix text color inside inputs */
        input[type="number"], div[data-baseweb="select"] span {
            color: white !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* --- FILE UPLOADER STYLING --- */
        /* This targets the drag-and-drop area */
        div[data-testid="stFileUploader"] section { 
            background-color: rgba(0, 0, 0, 0.45) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }
        
        /* Style the small 'Browse files' button inside the uploader */
        div[data-testid="stFileUploader"] button {
            background: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
               width: 120px !important;
            display: inline-table !important;
        }

        /* CARDS */
        .metric-card {
            background: var(--card-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        }

        /* METRIC BOXES */
        div[data-testid="stMetric"] {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 15px;
            border-left: 3px solid var(--primary-color);
            backdrop-filter: blur(10px);
        }

        div[data-testid="stMetricLabel"] { color: var(--text-muted); }
        div[data-testid="stMetricValue"] { 
            color: #FFE8B3; 
            font-family: 'Teko', sans-serif; 
            font-size: 2rem; 
        }

        /* BUTTONS */
        .stButton > button {
            background: linear-gradient(135deg, #D4FF00 0%, #aacc00 100%);
            color: black;
            font-family: 'Teko', sans-serif;
            font-size: 1.2rem;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            border: none;
        }

        /* TABS */
        .stTabs [data-baseweb="tab"] {
           background-color: rgba(255, 255, 255, 0.05);
           border-radius: 8px;
           color: white;
           font-family: 'Teko', sans-serif;
           font-size: 1.2rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color) !important;
            color: black !important;
        }

        /* GEMINI INTELLIGENCE SECTION */
        .gemini-card {
            background: rgba(0, 0, 0, 0.55);
            backdrop-filter: blur(15px);
            border-radius: 18px;
            padding: 30px;
            margin-top: 30px;
            margin-bottom: 40px;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }

        .gemini-card h2 {
            color: #FFE8B3 !important;
            font-family: 'Teko', sans-serif !important;
            font-size: 2.8rem !important;
            font-weight: 700;
            letter-spacing: 1px;
            text-shadow: 0 0 10px rgba(255, 220, 150, 0.5);
        }

        .gemini-card p, .gemini-card li {
            color: #f7f7f7 !important;
            font-size: 1.15rem;
            line-height: 1.7;
        }

        /* HIDE STREAMLIT MENUS & HEADER */
        #MainMenu, footer {visibility: hidden;}
        .block-container > div {
            background: transparent !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------
# 🧠 LOGIC LAYER (UNCHANGED)
# ----------------------------------------------------

@dataclass
class BodyMeasurements:
    """Store validated body measurements"""
    height_cm: float
    weight_kg: float
    age: int
    gender: str
    bmi: float
    bmi_category: str
    
def get_ideal_ranges(region_name: str, gender: str) -> Tuple[float, float]:
    """Returns ideal fat score ranges for each body region based on gender"""
    ideal_ranges = {
        "face": {"Male": (0.2, 0.4), "Female": (0.15, 0.35)},
        "chest": {"Male": (0.3, 0.5), "Female": (0.25, 0.45)},
        "stomach": {"Male": (0.25, 0.45), "Female": (0.2, 0.4)},
        "left_arm": {"Male": (0.15, 0.35), "Female": (0.1, 0.3)},
        "right_arm": {"Male": (0.15, 0.35), "Female": (0.1, 0.3)},
        "left_thigh": {"Male": (0.2, 0.4), "Female": (0.25, 0.45)},
        "right_thigh": {"Male": (0.2, 0.4), "Female": (0.25, 0.45)},
    }
    
    return ideal_ranges.get(region_name, {}).get(gender, (0.2, 0.4))

def get_ai_recommendations(analysis: Dict, body_measurements: BodyMeasurements) -> Dict:
    """
    Sends BMI + segmentation data to Gemini API (gemini-2.5-flash)
    and returns diet, exercise, lifestyle recommendations.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return {
            "priority_areas": [],
            "diet": ["❌ *GEMINI SETUP ERROR:* API Key not found. Please set the 'GEMINI_API_KEY' environment variable."],
            "exercise": [],
            "lifestyle": []
        }

    try:
        client = genai.Client()
    except Exception as e:
        return {
            "priority_areas": [],
            "diet": [f"❌ Gemini Client Initialization Error: {str(e)}"],
            "exercise": [],
            "lifestyle": []
        }

    prompt = f"""
You are an expert fitness coach and nutrition specialist.
Analyze the provided user data and regional fat accumulation scores.
CRITICAL INSTRUCTION: Respond with STRICT JSON object ONLY. NO extra text, conversation, or markdown fences.

### DATA ###
Age: {body_measurements.age}
Gender: {body_measurements.gender}
Height: {body_measurements.height_cm} cm
Weight: {body_measurements.weight_kg} kg
BMI: {body_measurements.bmi} ({body_measurements.bmi_category})
Overall Fat Score: {analysis.get('overall_score', 0)}

### REGIONAL FAT ACCUMULATION SCORES (0.0=Lean, 1.0=High Fat) ###
{json.dumps(analysis.get('regions', {}), indent=2)}

### REQUIRED JSON FORMAT ###
{{
  "priority_areas": [],
  "diet": [],
  "exercise": [],
  "lifestyle": []
}}

Generate the JSON object based ONLY on the data and the format above.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "priority_areas": {"type": "array", "items": {"type": "string"}},
                        "diet": {"type": "array", "items": {"type": "string"}},
                        "exercise": {"type": "array", "items": {"type": "string"}},
                        "lifestyle": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        )
        return json.loads(response.text)

    except APIError as e:
        return {
            "priority_areas": [],
            "diet": [f"❌ Gemini API Call Failed: Status {e.status_code} - {e.message}"],
            "exercise": [],
            "lifestyle": []
        }
    except Exception as e:
        return {
            "priority_areas": [],
            "diet": [f"❌ Unknown Error during AI generation: {str(e)}"],
            "exercise": [],
            "lifestyle": []
        }

def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h < 480 or w < 360:
        scale_h = 640 / h if h < 640 else 1.0
        scale_w = 480 / w if w < 480 else 1.0
        scale = max(scale_h, scale_w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return upscaled
    return image_bgr

def validate_image_quality(image_bgr: np.ndarray) -> Tuple[bool, str]:
    h, w = image_bgr.shape[:2]
    if h < 240 or w < 180:
        return False, "Image resolution extremely low. Please use a better quality image."
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 30:
        return False, "Image too dark. Please use better lighting."
    if mean_brightness > 230:
        return False, "Image overexposed. Reduce lighting or avoid bright backgrounds."
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return False, "Image too blurry. Use a stable camera and good focus."
    return True, "Image quality acceptable"

def validate_pose_quality(landmarks, img_w: int, img_h: int) -> Tuple[bool, str, float]:
    lm = landmarks.landmark
    def visibility_score(idx):
        return lm[idx].visibility if hasattr(lm[idx], "visibility") else 1.0
    key_points = [
        mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
    ]
    visibilities = [visibility_score(kp.value) for kp in key_points]
    avg_visibility = np.mean(visibilities)
    if avg_visibility < 0.4:
        return False, "Body not clearly visible. Ensure full body is in frame with good lighting.", avg_visibility
    left_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_width_norm = abs(right_sh.x - left_sh.x)
    if shoulder_width_norm < 0.06:
        return False, "Please face the camera directly (not sideways).", avg_visibility
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_shoulder_angle = abs((left_sh.y + right_sh.y) / 2 - (left_hip.y + right_hip.y) / 2)
    if hip_shoulder_angle < 0.12:
        return False, "Stand upright. Body appears too horizontal or bent.", avg_visibility
    return True, "Pose quality good", avg_visibility

def calculate_bmi_metrics(height_cm: float, weight_kg: float, age: int, gender: str) -> Dict:
    if height_cm <= 0 or weight_kg <= 0:
        return {}
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"; risk = "Nutritional deficiency risk"
    elif 18.5 <= bmi < 25:
        category = "Normal weight"; risk = "Low health risk"
    elif 25 <= bmi < 30:
        category = "Overweight"; risk = "Moderate health risk"
    elif 30 <= bmi < 35:
        category = "Obese Class I"; risk = "High health risk"
    elif 35 <= bmi < 40:
        category = "Obese Class II"; risk = "Very high health risk"
    else:
        category = "Obese Class III"; risk = "Extremely high health risk"
    return {
        "bmi": round(bmi, 2), "category": category, "risk": risk,
        "healthy_weight_range": (round(18.5 * height_m ** 2, 1), round(24.9 * height_m ** 2, 1)),
    }

def improved_segmentation_metrics(mask_region: np.ndarray, region_name: str) -> Dict:
    h, w = mask_region.shape
    if h == 0 or w == 0 or h < 10 or w < 10:
        return {"area_ratio": 0, "density": 0, "uniformity": 0, "compactness": 0}
    contours, _ = cv2.findContours(mask_region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area_ratio": 0, "density": 0, "uniformity": 0, "compactness": 0}
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    if not contours:
        return {"area_ratio": 0, "density": 0, "uniformity": 0, "compactness": 0}
    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask_region, dtype=np.uint8)
    cv2.drawContours(clean_mask, [largest_contour], -1, color=1, thickness=cv2.FILLED)
    total_pixels_in_box = h * w
    body_pixels = np.sum(clean_mask > 0)
    if body_pixels == 0:
        return {"area_ratio": 0, "density": 0, "uniformity": 0, "compactness": 0}
    area_ratio = body_pixels / total_pixels_in_box
    row_densities = []
    for row in clean_mask:
        if np.any(row):
            row_densities.append(np.sum(row > 0) / len(row))
    if len(row_densities) == 0:
        density = 0; uniformity = 0
    else:
        density = np.mean(row_densities)
        uniformity = max(0, 1 - np.std(row_densities)) if len(row_densities) > 1 else 0.5
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    compactness = 0
    if perimeter > 0:
        compactness = min((4 * np.pi * area) / (perimeter ** 2), 1.0)
    return {"area_ratio": float(area_ratio), "density": float(density), "uniformity": float(uniformity), "compactness": float(compactness)}

def calculate_region_fat_score(metrics: Dict, region_name: str, body_measurements: BodyMeasurements) -> float:
    area = metrics.get("area_ratio", 0)
    density = metrics.get("density", 0)
    uniformity = metrics.get("uniformity", 0)
    compactness = metrics.get("compactness", 0)
    if region_name == "stomach":
        score = 0.35 * compactness + 0.30 * density + 0.25 * area + 0.10 * (1 - uniformity)
    elif region_name == "chest":
        score = 0.30 * compactness + 0.30 * density + 0.20 * area + 0.20 * (1 - uniformity)
    elif "arm" in region_name:
        score = 0.40 * compactness + 0.30 * density + 0.20 * (1 - uniformity) + 0.10 * area
    elif "thigh" in region_name:
        score = 0.35 * compactness + 0.35 * density + 0.20 * area + 0.10 * (1 - uniformity)
    elif region_name == "face":
        score = 0.50 * compactness + 0.30 * area + 0.20 * density
    else:
        score = 0.35 * compactness + 0.30 * density + 0.25 * area + 0.10 * (1 - uniformity)
    gender_factor = 1.0
    if body_measurements.gender == "Female":
        if "thigh" in region_name or region_name == "face": gender_factor = 1.1
        elif region_name == "stomach": gender_factor = 0.9
    elif body_measurements.gender == "Male":
        if region_name == "stomach" or region_name == "chest": gender_factor = 1.1
    bmi_factor = np.clip((body_measurements.bmi - 18.5) / 20, 0, 1)
    adjusted_score = (score * gender_factor) * 0.7 + bmi_factor * 0.3
    return float(np.clip(adjusted_score, 0, 1))

def get_anatomical_regions(landmarks, img_w: int, img_h: int) -> Dict[str, Tuple[int, int, int, int]]:
    lm = landmarks.landmark
    def get_point(idx):
        pt = lm[idx]; return int(pt.x * img_w), int(pt.y * img_h)
    def get_norm_point(idx): return lm[idx].x, lm[idx].y
    def safe_bounds(x1, y1, x2, y2):
        x1 = max(0, min(int(x1), img_w - 1)); y1 = max(0, min(int(y1), img_h - 1))
        x2 = max(x1 + 10, min(int(x2), img_w)); y2 = max(y1 + 10, min(int(y2), img_h))
        return (x1, y1, x2, y2)
    nose = get_point(mp_pose.PoseLandmark.NOSE.value)
    left_shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    left_elbow = get_point(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    right_elbow = get_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    left_wrist = get_point(mp_pose.PoseLandmark.LEFT_WRIST.value)
    right_wrist = get_point(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    left_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP.value)
    right_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
    left_knee = get_point(mp_pose.PoseLandmark.LEFT_KNEE.value)
    right_knee = get_point(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    _, left_knee_y_norm = get_norm_point(mp_pose.PoseLandmark.LEFT_KNEE.value)
    _, right_knee_y_norm = get_norm_point(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    shoulder_width = max(10, abs(right_shoulder[0] - left_shoulder[0]))
    torso_top_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_avg_y = (left_hip[1] + right_hip[1]) / 2
    torso_height_pixels = max(10, abs(hip_avg_y - torso_top_y))
    regions: Dict[str, Tuple[int, int, int, int]] = {}
    face_width = max(20, int(shoulder_width * 0.60))
    face_height = max(20, int(abs(nose[1] - left_shoulder[1]) * 0.90))
    regions["face"] = safe_bounds(nose[0] - face_width // 2, nose[1] - int(face_height * 0.65), nose[0] + face_width // 2, nose[1] + int(face_height * 0.45))
    chest_top = min(left_shoulder[1], right_shoulder[1])
    chest_bottom = int(chest_top + torso_height_pixels * 0.4)
    chest_left = min(left_shoulder[0], right_shoulder[0]) - int(shoulder_width * 0.2)
    chest_right = max(left_shoulder[0], right_shoulder[0]) + int(shoulder_width * 0.2)
    regions["chest"] = safe_bounds(chest_left, chest_top, chest_right, chest_bottom)
    stomach_top = chest_bottom
    stomach_bottom = min(left_hip[1], right_hip[1])
    stomach_left = min(left_hip[0], right_hip[0]) - int(shoulder_width * 0.25)
    stomach_right = max(left_hip[0], right_hip[0]) + int(shoulder_width * 0.25)
    regions["stomach"] = safe_bounds(stomach_left, stomach_top, stomach_right, stomach_bottom)
    arm_width = max(15, int(shoulder_width * 0.30))
    regions["left_arm"] = safe_bounds(min(left_shoulder[0], left_elbow[0], left_wrist[0]) - arm_width // 2, left_shoulder[1], max(left_shoulder[0], left_elbow[0], left_wrist[0]) + arm_width // 2, max(left_elbow[1], left_wrist[1]))
    regions["right_arm"] = safe_bounds(min(right_shoulder[0], right_elbow[0], right_wrist[0]) - arm_width // 2, right_shoulder[1], max(right_shoulder[0], right_elbow[0], right_wrist[0]) + arm_width // 2, max(right_elbow[1], right_wrist[1]))
    thigh_width = max(20, int(shoulder_width * 0.35))
    estimated_knee_y_left = left_hip[1] + torso_height_pixels * 1.1 if left_knee_y_norm > 0.95 else left_knee[1]
    regions["left_thigh"] = safe_bounds(min(left_hip[0], left_knee[0]) - thigh_width // 2, left_hip[1], max(left_hip[0], left_knee[0]) + thigh_width // 2, estimated_knee_y_left)
    estimated_knee_y_right = right_hip[1] + torso_height_pixels * 1.1 if right_knee_y_norm > 0.95 else right_knee[1]
    regions["right_thigh"] = safe_bounds(min(right_hip[0], right_knee[0]) - thigh_width // 2, right_hip[1], max(right_hip[0], right_knee[0]) + thigh_width // 2, estimated_knee_y_right)
    return regions

def analyze_body_composition(image_bgr: np.ndarray, body_measurements: BodyMeasurements) -> Tuple[Optional[np.ndarray], Dict]:
    processed_image = preprocess_image(image_bgr)
    quality_ok, quality_msg = validate_image_quality(processed_image)
    if not quality_ok: return None, {"error": quality_msg}
    img_h, img_w = processed_image.shape[:2]
    annotated = processed_image.copy()
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_image)
    if not pose_results.pose_landmarks: return None, {"error": "No person detected. Ensure full body is visible in frame."}
    pose_ok, pose_msg, visibility = validate_pose_quality(pose_results.pose_landmarks, img_w, img_h)
    if not pose_ok: return None, {"error": pose_msg}
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        seg_results = seg.process(rgb_image)
    if seg_results.segmentation_mask is None: return None, {"error": "Segmentation failed. Try different lighting or background."}
    binary_mask = (seg_results.segmentation_mask > 0.6).astype(np.uint8)
    regions = get_anatomical_regions(pose_results.pose_landmarks, img_w, img_h)
    region_analyses: Dict[str, Dict] = {}
    for region_name, (x1, y1, x2, y2) in regions.items():
        if x2 <= x1 or y2 <= y1 or x1 >= img_w or y1 >= img_h: continue
        mask_region = binary_mask[y1:y2, x1:x2]
        morph_metrics = improved_segmentation_metrics(mask_region, region_name)
        fat_score = calculate_region_fat_score(morph_metrics, region_name, body_measurements)
        if fat_score >= 0.65: level = "High"; color = (0, 0, 255)
        elif fat_score >= 0.40: level = "Moderate"; color = (0, 165, 255)
        else: level = "Low"; color = (0, 255, 0)
        region_analyses[region_name] = {"score": round(fat_score, 3), "level": level, "metrics": morph_metrics, "box": (x1, y1, x2, y2), "color": color}
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{region_name.replace('_', ' ').title()}: {level}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    mp_drawing.draw_landmarks(annotated, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2))
    weights = {"stomach": 0.30, "chest": 0.20, "left_arm": 0.08, "right_arm": 0.08, "left_thigh": 0.12, "right_thigh": 0.12, "face": 0.10}
    valid_scores = [region_analyses[r]["score"] * weights[r] for r in region_analyses.keys() if r in weights]
    total_weight = sum(weights[r] for r in region_analyses.keys() if r in weights)
    overall_score = sum(valid_scores) / total_weight if total_weight > 0 else 0
    return annotated, {"regions": region_analyses, "overall_score": round(overall_score, 3), "pose_visibility": round(visibility, 2), "image_quality": quality_msg, "image_upscaled": processed_image.shape != image_bgr.shape}

import base64

def set_bg(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

# ----------------------------------------------------
# 🚀 MAIN APPLICATION UI (NEW STRUCTURE)
# ----------------------------------------------------
def main():
    inject_custom_css()
    set_bg(r"C:\Users\hiuna\Downloads\background.jpg") 

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.markdown("## SIGMENT FIT TRAINER")
        selected = st.radio(
            "Menu",
            ["Home", "Analysis", "Instructions", "About"],
            index=0,
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown(
            """
            <div style='background:rgba(255,255,255,0.05); padding:16px; border-radius:10px;'>
                <small style='color:#888'>SYSTEM STATUS</small><br>
                <span style='color:#0f0'>●</span> <b>Vision Engine Online</b>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- VIEW: HOME ---
    if selected == "Home":

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Define Layout: Text (Left), Image (Right)
        left, right = st.columns([1.3, 1]) 

        # LEFT COLUMN: TEXT & BUTTON
        with left:
            st.markdown("""
            <h1 style='line-height:1.0; font-size:4rem;'>
                UNLOCK YOUR <span style='color:#D4FF00;'>TRUE</span><br>
                PHYSIQUE
            </h1>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='hero-description-box'>
                <p class='hero-description-text'>
                    Advanced AI Computer Vision to map body segments, analyze fat distribution,
                    and build a personalized transformation roadmap.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            if st.button("START ANALYSIS NOW 🚀"):
                st.session_state.selected = "Analysis"
                st.info("Select 'Analysis' from the sidebar to begin.")

        # RIGHT COLUMN: SPHERICAL IMAGE
        with right:
            # Helper function to load local image for HTML
            def get_img_as_base64(file_path):
                with open(file_path, "rb") as f:
                    data = f.read()
                return base64.b64encode(data).decode()

            # UPDATE THIS PATH to your new dark theme image if necessary
            img_path = r"C:\Users\hiuna\Downloads\logo.jpg"
            
            try:
                img_base64 = get_img_as_base64(img_path)
                
                # HTML/CSS to make it Spherical
                st.markdown(
                    f"""
                    <style>
                        .spherical-frame {{
                            width: 350px;
                            height: 350px;
                            aspect-ratio: 1 / 1;       /* Forces a perfect square container */
                            border-radius: 50%;        /* Makes it a circle */
                            object-fit: cover;         /* Crops image to fit circle without stretching */
                          box-shadow: 0 0 25px rgba(128, 128, 128, 0.5);; /* Glow effect */
                            border: 3px solid rgba(255, 255, 255, 0.1);
                        }}
                    </style>
                    <img src="data:image/jpeg;base64,{img_base64}" class="spherical-frame">
                    """,
                    unsafe_allow_html=True
                )
            except FileNotFoundError:
                st.error("Image not found. Please check the file path.")

        

    # --- VIEW: ANALYSIS ---
    elif selected == "Analysis":
        st.markdown("<h2>📊 PHYSIQUE ANALYSIS LAB</h2>", unsafe_allow_html=True)
        
        
        # --- INPUT CONTAINER ---
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: age = st.number_input("AGE", 15, 100, 30)
        with c2: weight_kg = st.number_input("WEIGHT (KG)", 35.0, 250.0, 70.0)
        with c3: height_cm = st.number_input("HEIGHT (CM)", 120.0, 230.0, 170.0)
        with c4: gender = st.selectbox("GENDER", ["Male", "Female"])
        
        uploaded_file = st.file_uploader("UPLOAD BODY SCAN (Full body photo)", type=["jpg", "png", "webp"])
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file:
            # Process Logic (Unchanged)
            pil_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            bmi_data = calculate_bmi_metrics(height_cm, weight_kg, age, gender)
            body_measurements = BodyMeasurements(height_cm, weight_kg, age, gender, bmi_data["bmi"], bmi_data["category"])

            with st.spinner("⚡ SCANNING BIOMETRICS..."):
                annotated_img, analysis = analyze_body_composition(cv_image, body_measurements)

            if annotated_img is None:
                st.error(f"Scan Failed: {analysis.get('error')}")
            else:
                # --- RESULTS DASHBOARD ---
                st.markdown("<h2 class='section-title'>📊 Core Body Metrics</h2>", unsafe_allow_html=True)
                
                # Top Level Metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("BMI SCORE", f"{bmi_data['bmi']}", bmi_data["category"])
                with m2: st.metric("FAT INDEX", f"{analysis['overall_score']:.2f}", "Visual Score")
                with m3: st.metric("IDEAL WEIGHT", f"{bmi_data['healthy_weight_range'][1]} kg")
                with m4: st.metric("SCAN QUALITY", f"{int(analysis['pose_visibility']*100)}%")

                # Visuals
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.markdown("<h2 class='section-title'>📸 SCAN INPUT</h2>", unsafe_allow_html=True)

                    st.image(pil_image, use_container_width=True)
                with col_v2:
                    st.markdown("<h2 class='section-title'>🧬 SEGMENTATION MAP</h2>", unsafe_allow_html=True)

                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)

                
                # Regional Breakdown - EXACTLY like 77.png
                st.markdown("<h2 class='section-title'>📍 REGIONAL HEATMAP</h2>", unsafe_allow_html=True)

                regions = analysis.get("regions", {})

                desired_order = ['face', 'chest', 'stomach', 'left_arm', 'right_arm', 'left_thigh', 'right_thigh']

                row1_items = [('face', regions.get('face')), ('chest', regions.get('chest')), ('stomach', regions.get('stomach'))]
                row2_items = [('left_arm', regions.get('left_arm')), ('right_arm', regions.get('right_arm')), ('left_thigh', regions.get('left_thigh')), ('right_thigh', regions.get('right_thigh'))]

                if any(item[1] for item in row1_items):
                    cols = st.columns(5)
                    for idx, (r_name, r_data) in enumerate(row1_items):
                        if r_data:  # Only display if data exists
                            with cols[idx]:
                                score = r_data['score']
                                level = r_data['level']
                                color = "#9BFF00" if level == "Low" else "#ffaa00" if level == "Moderate" else "#ff4444"
                                
                                # Get ideal range
                                ideal_min, ideal_max = get_ideal_ranges(r_name, gender)
                                
                                st.markdown(
                                    f"""
                                    <div class="metric-card" style="border-top: 3px solid {color}; padding: 10px; text-align: center; min-height: 130px; display: flex; flex-direction: column; justify-content: center;">
                                        <h4 style="margin:0; color:#fff; font-size: 1.1rem;">{r_name.replace('_',' ').upper()}</h4>
                                        <h2 style="margin:8px 0; font-size:1.8rem !important; color: {color};">{score:.2f}</h2>
                                        <div style="margin-bottom: 5px;">
                                            <span style="color:{color}; font-weight: bold;">{level}</span>
                                        </div>
                                        <div>
                                            <small style="color:#CCFF00; font-size:0.75rem;">
                                                Ideal: {ideal_min:.1f} - {ideal_max:.1f}
                                            </small>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                # Second row - 4 columns  
                if any(item[1] for item in row2_items):
                    cols = st.columns(5)
                    for idx, (r_name, r_data) in enumerate(row2_items):
                        if r_data:  # Only display if data exists
                            with cols[idx]:
                                score = r_data['score']
                                level = r_data['level']
                                color = "#9BFF00" if level == "Low" else "#ffaa00" if level == "Moderate" else "#ff4444"
                                
                                # Get ideal range
                                ideal_min, ideal_max = get_ideal_ranges(r_name, gender)
                                
                                st.markdown(
                                    f"""
                                    <div class="metric-card" style="border-top: 3px solid {color}; padding: 10px; text-align: center; min-height: 130px; display: flex; flex-direction: column; justify-content: center;">
                                        <h4 style="margin:0; color:#fff; font-size: 1rem;">{r_name.replace('_',' ').upper()}</h4>
                                        <h2 style="margin:8px 0; font-size:1.6rem !important; color: {color};">{score:.2f}</h2>
                                        <div style="margin-bottom: 5px;">
                                            <span style="color:{color}; font-weight: bold; font-size: 0.9rem;">{level}</span>
                                        </div>
                                        <div>
                                            <small style="color:#CCFF00; font-size:0.7rem;">
                                                Ideal Range: {ideal_min:.1f} - {ideal_max:.1f}
                                            </small>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                # --- 🟢 RESTORED: DETAILED METRICS EXPANDER ---
                # --- 🟢 UPDATED: DETAILED METRICS EXPANDER WITH IDEAL RANGES ---
                # --- 🟢 UPDATED: DETAILED METRICS EXPANDER WITH IDEAL RANGES ---
                with st.expander("📋 DETAILED MORPHOMETRICS (Raw Data)"):
                    st.info("Metrics calculated on segmented shapes. Low lighting or dark clothes may affect density scoring.")
                    for r_name, r_data in regions.items():
                        st.markdown(f"#### {r_name.replace('_', ' ').upper()}")
                        
                        # Get ideal range for this region
                        ideal_min, ideal_max = get_ideal_ranges(r_name, gender)
                        current_score = r_data['score']
                        level = r_data['level']
                        
                        # Display current score vs ideal range - COMPACT 4-column layout
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            status_color = "🟢" if level == "Low" else "🟡" if level == "Moderate" else "🔴"
                            st.metric("STATUS", f"{status_color} {level}", f"Score: {current_score:.2f}")
                        with col2:
                            st.metric("IDEAL RANGE", f"{ideal_min:.1f} - {ideal_max:.1f}")
                        # col3 and col4 remain empty to create smaller boxes
                        
                        # Display the raw metrics
                        metrics = r_data["metrics"]
                        st.markdown("*Raw Metrics:*")
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1: st.metric("AREA RATIO", f"{metrics['area_ratio']:.2f}")
                        with mc2: st.metric("DENSITY", f"{metrics['density']:.2f}")
                        with mc3: st.metric("UNIFORMITY", f"{metrics['uniformity']:.2f}")
                        with mc4: st.metric("COMPACTNESS", f"{metrics['compactness']:.2f}")
                        st.markdown("---")

                # AI Coach
                st.markdown("---")
                st.markdown("<div class='gemini-card'>", unsafe_allow_html=True)

                st.markdown("<h2>🤖 GEMINI INTELLIGENCE</h2>", unsafe_allow_html=True)

                with st.spinner("Generating Neural Transformation Plan..."):
                    recommendations = get_ai_recommendations(analysis, body_measurements)

                
                # --- 🟢 RESTORED: PRIORITY AREAS ALERT ---
                if recommendations.get("priority_areas"):
                    p_str = ", ".join([x.replace("_", " ").upper() for x in recommendations["priority_areas"]])
                    st.markdown(
                        f"""
                        <div style="background: rgba(255, 68, 68, 0.15); border-left: 4px solid #ff4444; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <strong style="color: #ff4444;">⚠ PRIORITY ZONES DETECTED:</strong> <span style="color: #fff;">{p_str}</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.success("✅ No critical high-fat zones detected. Focus on maintenance and performance.")

                t1, t2, t3 = st.tabs(["🥗 NUTRITION PROTOCOL", "🏋 TRAINING BLUEPRINT", "🧬 LIFESTYLE HACKS"])

                with t1:
                    st.markdown("<div class='tab-content-strong'>", unsafe_allow_html=True)
                    for item in recommendations.get("diet", []):
                        st.markdown(f"• {item}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with t2:
                    st.markdown("<div class='tab-content-strong'>", unsafe_allow_html=True)
                    for item in recommendations.get("exercise", []):
                        st.markdown(f"• {item}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with t3:
                    st.markdown("<div class='tab-content-strong'>", unsafe_allow_html=True)
                    for item in recommendations.get("lifestyle", []):
                        st.markdown(f"• {item}")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True) 
                st.markdown("</div>", unsafe_allow_html=True)


    # --- VIEW: INSTRUCTIONS ---
    elif selected == "Instructions":
        st.markdown("<h2>📝 HOW TO SCAN</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='metric-card'>
            <ol>
                <li><b>Lighting:</b> Ensure bright, front-facing light.</li>
                <li><b>Distance:</b> Stand 6-8 feet away. Full body must be visible.</li>
                <li><b>Attire:</b> Wear fitted clothing. Baggy clothes affect segmentation.</li>
                <li><b>Pose:</b> Stand straight, arms slightly away from body (A-Pose).</li>
            </ol>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- VIEW: ABOUT ---
    elif selected == "About":
        st.markdown("<h2>ℹ ABOUT SYSTEM</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='metric-card'>
            This system utilizes <b>Google MediaPipe</b> for pose landmark detection and segmented masking. 
            It calculates morphological metrics (density, compactness) to estimate fat distribution relative to frame size.
            <br><br>
            <i>Disclaimer: Not a medical device. For fitness tracking purposes only.</i>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- FOOTER ---
    st.markdown("""
<style>
.footer {
    color: #ffffff;
}
</style>

<div class="footer">
    <p><strong>Segment Fit Trainer © 2025</strong></p>
    <p>Empowering smarter fitness through AI-driven body analysis</p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()