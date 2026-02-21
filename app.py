import streamlit as st
import cv2
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import os
import tempfile
from datetime import datetime
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Import your custom modules
from config import *
from eye_utils import eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX
from mouth_utils import preprocess_mouth, get_mouth_roi
from alert import AlertSystem

# --- PAGE CONFIG ---
st.set_page_config(page_title="Driver Drowsiness System", page_icon="🚗", layout="wide")

# Google STUN servers help bypass firewalls for the video stream
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-title { font-size: 45px; font-weight: bold; text-align: left; margin-bottom: 20px; }
    .slider-label { font-size: 14px; color: #555; margin-bottom: -10px; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_resources():
    mouth_model = tf.keras.models.load_model(YAWN_MODEL_PATH)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    alert_sys = AlertSystem(ALARM_SOUND_PATH)
    return mouth_model, face_landmarker, alert_sys

mouth_model, face_landmarker, alert = load_resources()

# --- WEBRTC VIDEO PROCESSOR ---
class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.ear_history = deque(maxlen=EAR_HISTORY)
        self.yawn_history = deque(maxlen=MOUTH_HISTORY)
        self.eye_start_time = None
        self.ear_thresh = 0.23
        self.yawn_thresh = 0.50

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Use current timestamp for MediaPipe Video Mode
        timestamp = int(time.time() * 1000)
        result = face_landmarker.detect_for_video(mp_image, timestamp)
        
        eye_status, mouth_status = "AWAKE", "NORMAL"
        
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            
            # EAR calculation
            left = [landmarks[i] for i in LEFT_EYE_IDX]
            right = [landmarks[i] for i in RIGHT_EYE_IDX]
            ear_val = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
            self.ear_history.append(ear_val)
            avg_ear = np.mean(self.ear_history)
            
            # Yawn Detection
            m_roi, _ = get_mouth_roi(img, landmarks, w, h)
            m_input = preprocess_mouth(m_roi)
            if m_input is not None:
                yawn_val = mouth_model.predict(m_input, verbose=0)[0][1]
                self.yawn_history.append(yawn_val)
                avg_yawn = np.mean(self.yawn_history)
                if avg_yawn > self.yawn_thresh: 
                    mouth_status = "YAWNING"

            # Drowsiness logic
            if avg_ear < self.ear_thresh:
                eye_status = "SLEEPY"
                if self.eye_start_time is None: 
                    self.eye_start_time = time.time()
                elif time.time() - self.eye_start_time > CLOSED_EYE_SECONDS:
                    # Alert triggers on server side
                    alert.play() 
            else:
                self.eye_start_time = None

            # Overlay Text
            color = (0, 0, 255) if eye_status == "SLEEPY" or mouth_status == "YAWNING" else (0, 255, 0)
            cv2.putText(img, f"EYE: {eye_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(img, f"MOUTH: {mouth_status}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
        return img

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Select Detection Mode", ["Real-Time Detection", "Video Upload Detection"])
    st.markdown("---")
    if st.button("Reset Session"):
        st.rerun()

# --- MAIN UI HEADER ---
st.markdown('<div class="main-title">🚗 Driver Drowsiness Detection System</div>', unsafe_allow_html=True)

# --- MODE 1: REAL-TIME (WebRTC) ---
if mode == "Real-Time Detection":
    col1, col2 = st.columns(2)
    with col1:
        ear_thresh = st.slider("Eye Aspect Ratio (Sensitivity)", 0.10, 0.40, 0.23, 0.01, key="ear_slider")
    with col2:
        yawn_thresh = st.slider("Yawn Confidence (Threshold)", 0.10, 0.90, 0.50, 0.05, key="yawn_slider")

    # The WebRTC component
    webrtc_ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_transformer_factory=DrowsinessTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Push slider values to the transformer logic
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.ear_thresh = ear_thresh
        webrtc_ctx.video_transformer.yawn_thresh = yawn_thresh

# --- MODE 2: VIDEO UPLOAD ---
elif mode == "Video Upload Detection":
    st.subheader("📁 Upload Video for Analysis")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        if st.button("Process Video"):
            cap = cv2.VideoCapture(tfile.name)
            ear_history = deque(maxlen=7)
            yawn_history = deque(maxlen=7)
            eye_start_time = None
            frame_win = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (1280, 720))
                
                # Use a dummy wrapper or reuse the logic here
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = face_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
                
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    # Simple drawing for upload mode
                    cv2.putText(frame, "Processing...", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                frame_win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()
