# ------------------ INSTALL FIRST ------------------
# pip install streamlit ultralytics opencv-python pyttsx3

import streamlit as st
import cv2
from ultralytics import YOLO
import pyttsx3
from collections import Counter
import time

# ------------------ STREAMLIT SETUP ------------------
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("🎯 YOLOv8 Real-Time Object Detection")
st.caption("Live webcam detection with voice alerts")

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")
start_btn = st.sidebar.button("▶ Start Detection")
stop_btn = st.sidebar.button("⏹ Stop Detection")

# ------------------ SESSION STATE ------------------
if "run" not in st.session_state:
    st.session_state.run = False

if start_btn:
    st.session_state.run = True

if stop_btn:
    st.session_state.run = False

# ------------------ PLACEHOLDERS ------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📷 Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Detected Objects")
    count_placeholder = st.empty()

# ------------------ LOAD MODEL ------------------
model = YOLO("yolov8n.pt")

# ------------------ TTS ENGINE ------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)
engine.startLoop(False)

last_spoken_text = ""

# ------------------ MAIN LOOP (FIXED PART) ------------------
if st.session_state.run:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot access camera")
        st.stop()

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        detected_labels = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                detected_labels.append(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        counts = Counter(detected_labels)

        # draw counts
        y_offset = 30
        for obj, cnt in counts.items():
            cv2.putText(frame, f"{obj}: {cnt}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 30

        # speech
        if counts:
            spoken_text = "Detected " + ", ".join(
                [f"{cnt} {obj}" for obj, cnt in counts.items()]
            )
        else:
            spoken_text = "No objects detected"

        if spoken_text != last_spoken_text:
            engine.say(spoken_text)
            last_spoken_text = spoken_text

        engine.iterate()

        # STREAMLIT DISPLAY (fixed)
        video_placeholder.image(frame, channels="BGR", use_column_width=True)
        count_placeholder.write(counts if counts else "No objects detected")

        time.sleep(0.03)

    # CLEAN STOP (important)
    cap.release()
    engine.endLoop()

else:
    st.info("▶ Click **Start Detection** to begin")
