import os
import streamlit as st
from ultralytics import YOLO
import cv2
import random
import time
from gtts import gTTS
import pygame
import threading
from datetime import datetime, timedelta

# Initialize pygame mixer
pygame.mixer.quit()  # Ensure the mixer is fully stopped
pygame.mixer.init()

# Load YOLOv8 model
yolo = YOLO("yolov8n.pt")

# Streamlit app layout
st.set_page_config(page_title="Assistive Vision App", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        font-family: "Arial", sans-serif;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        justify-content: center;
        align-items: center;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    .stCheckbox {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a container for the buttons
st.markdown('<div class="button-container">', unsafe_allow_html=True)

# # Render buttons inside the container
# start_detection = st.button("Start Detection", key="start")
# stop_detection = st.button("Stop Detection", key="stop")

st.markdown('</div>', unsafe_allow_html=True)

# Display welcome image
welcome_image_path = "bismillah.png"  # Ensure this image exists in the script's directory
if os.path.exists(welcome_image_path):
    st.image(welcome_image_path, use_container_width=True, caption="Bismillah hir Rehman Ar Raheem")
else:
    st.warning("Welcome image not found! Please add 'bismillah.png' in the script directory.")

st.title("Object Detection & Assistive Vision App for Visually Impaired People")
st.write("This application provides real-time object recognition and optional audio alerts.")

# Directory to store temp audio files
audio_temp_dir = "audio_temp_files"
if not os.path.exists(audio_temp_dir):
    os.makedirs(audio_temp_dir)

# Placeholder for video frames
stframe = st.empty()

# User controls
col1, col2 = st.columns(2)
with col1:
    start_detection = st.button("Start Detection")
with col2:
    stop_detection = st.button("Stop Detection")
audio_activation = st.checkbox("Enable Audio Alerts", value=False)

# Categories for audio alerts (hazardous objects or living things)
alert_categories = {"person", "cat", "dog", "knife", "fire", "gun"}

# Dictionary to store the last alert timestamp for each object
last_alert_time = {}
alert_cooldown = timedelta(seconds=10)  # 10-second cooldown for alerts


def play_audio_alert(label, position):
    """Generate and play an audio alert."""
    phrases = [
        f"Be careful, there's a {label} on your {position}.",
        f"Watch out! {label} detected on your {position}.",
        f"Alert! A {label} is on your {position}.",
    ]
    caution_note = random.choice(phrases)

    temp_file_path = os.path.join(audio_temp_dir, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3")

    tts = gTTS(caution_note)
    tts.save(temp_file_path)

    try:
        pygame.mixer.music.load(temp_file_path)
        pygame.mixer.music.play()

        def cleanup_audio_file():
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.stop()
            try:
                os.remove(temp_file_path)
            except OSError as e:
                print(f"Error deleting file {temp_file_path}: {e}")

        threading.Thread(target=cleanup_audio_file, daemon=True).start()

    except Exception as e:
        print(f"Error playing audio alert: {e}")


def process_frame(frame, audio_mode):
    """Process a single video frame for object detection."""
    results = yolo(frame)
    result = results[0]

    detected_objects = {}
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]

        if audio_mode and label not in alert_categories:
            continue

        frame_center_x = frame.shape[1] // 2
        obj_center_x = (x1 + x2) // 2
        position = "left" if obj_center_x < frame_center_x else "right"

        detected_objects[label] = position

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return detected_objects, frame


# Main logic
if start_detection:
    st.success("Object detection started.")
    try:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            st.error("Could not access the webcam. Please check your camera settings.")
        else:
            while not stop_detection:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to capture video. Please check your camera.")
                    break

                detected_objects, processed_frame = process_frame(frame, audio_activation)

                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)

                if audio_activation:
                    current_time = datetime.now()
                    for label, position in detected_objects.items():
                        if (
                            label not in last_alert_time
                            or current_time - last_alert_time[label] > alert_cooldown
                        ):
                            play_audio_alert(label, position)
                            last_alert_time[label] = current_time

                time.sleep(0.1)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if 'video_capture' in locals() and video_capture.isOpened():
            video_capture.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()

elif stop_detection:
    st.warning("Object detection stopped.")