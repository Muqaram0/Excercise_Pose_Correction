import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import deque

# Load YOLO model
model = YOLO('C:/Users/narji/Desktop/best/best.pt').to('cuda')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize buffers for angles and elbow positions
angle_buffer = deque(maxlen=10)
elbow_position_buffer = deque(maxlen=10)

# Global variable for push-up tracking
prev_hip_y = None


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = np.array([b[0] - a[0], b[1] - a[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    dot_product = np.dot(ab, bc)
    mag_ab = np.linalg.norm(ab)
    mag_bc = np.linalg.norm(bc)
    if mag_ab == 0 or mag_bc == 0:
        return 0
    angle = np.degrees(np.arccos(dot_product / (mag_ab * mag_bc)))
    return angle


# Function to check bicep curl form
def check_bicep_curl_form(landmarks, frame):
    feedback = "Good Form"
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_elbow_angle = calculate_angle(
        [left_shoulder[0], left_shoulder[1]],
        [left_elbow[0], left_elbow[1]],
        [left_wrist[0], left_wrist[1]]
    )
    shoulder_elbow_alignment = abs(left_shoulder[1] - left_elbow[1])
    if left_elbow_angle < 10 or left_elbow_angle > 170:
        feedback = "Bad Form: Incorrect Elbow Angle"
    if shoulder_elbow_alignment > 120:
        feedback = "Bad Form: Shoulder-Elbow Alignment Off"
    cv2.putText(frame, f"Elbow Angle: {int(left_elbow_angle)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Shoulder-Elbow Alignment: {shoulder_elbow_alignment:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return feedback


# Function to check push-up form
def check_pushup_form(landmarks, frame):
    feedback = "Good Form"
    global prev_hip_y
    
    # Extract key landmarks
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    head = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Push-up phase detection logic
    phase = "Up Phase"
    if prev_hip_y is not None:
        if hip[1] > prev_hip_y:  # Hip moving down
            phase = "Down Phase"
        else:  # Hip moving up
            phase = "Up Phase"
    prev_hip_y = hip[1]

    # Midpoint of ankle and toe
    midpoint_y = (toe[1] + ankle[1]) / 2

    # Check for back straightness (Shoulder-Hip-Knee angle ~ 180°)
    back_angle = calculate_angle(
        [shoulder[0], shoulder[1]],  # Shoulder
        [hip[0], hip[1]],           # Hip
        [knee[0], knee[1]]          # Knee
    )

    # Check the conditions for bad form
    if phase == "Up Phase":
        if knee[1] >= midpoint_y:  # Knees too low
            feedback = "Bad Form: Lift your knees"
        elif back_angle > 20 or back_angle < 0:  # Relaxed threshold for back straightness
            feedback = "Bad Form: Straighten your back"
    
    elif phase == "Down Phase":
        if knee[1] >= midpoint_y:  # Knees too low
            feedback = "Bad Form: Lift your knees"
        elif back_angle > 20 or back_angle < 0:  # Relaxed threshold for back straightness
            feedback = "Bad Form: Straighten your back"


    # Debugging Display
    cv2.putText(frame, f"Phase: {phase}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Knee Y: {int(knee[1])}, Midpoint Y: {int(midpoint_y)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Hip Y: {int(hip[1])}, Head Y: {int(head[1])}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Back Angle: {int(back_angle)}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Feedback display

    return feedback




# Function to check squat form
def check_squat_form(landmarks, frame):
    feedback = "Good Form"

    # Extract key points (x, y)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    left_toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

    # Unpack (x, y) values
    shoulder_x, shoulder_y = left_shoulder
    hip_x, hip_y = left_hip
    knee_x, knee_y = left_knee
    ankle_x, ankle_y = left_ankle
    toe_x, toe_y = left_toe

    # Calculate horizontal and vertical shoulder-hip distances
    shoulder_hip_horizontal_distance = abs(shoulder_x - hip_x)
    shoulder_hip_vertical_distance = abs(shoulder_y - hip_y)

    # Calculate angles
    back_angle = calculate_angle([shoulder_x, shoulder_y], [hip_x, hip_y], [ankle_x, ankle_y])
    hip_knee_ankle_angle = calculate_angle([hip_x, hip_y], [knee_x, knee_y], [ankle_x, ankle_y])

    # Check knee-toe alignment
    knee_toe_distance = knee_x - toe_x

    # Standing position
    if back_angle < 50 and hip_knee_ankle_angle < 30:
        feedback = "Good Form: Standing Position"

    # Shoulders ahead of knees
    elif shoulder_x > knee_x + 20:  # Correct logic here
        feedback = "Bad Form: Shoulders Ahead of Knees"

    # Squatting position logic
    elif 90 <= hip_knee_ankle_angle <= 160:
        if back_angle < 60:
            feedback = "Bad Form: Keep Back Straight"
        elif knee_toe_distance < -20:
            feedback = "Bad Form: Knees Past Toes"
        else:
            feedback = "Good Form: Squatting"

    # Deep squat logic
    elif hip_knee_ankle_angle > 160:
        if back_angle < 60:
            feedback = "Bad Form: Keep Back Straight"
        elif knee_toe_distance < -20:
            feedback = "Bad Form: Knees Past Toes"
        else:
            feedback = "Good Form: Deep Squat"

    # Debugging Information
    cv2.putText(frame, f"Back Angle: {int(back_angle)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Hip-Knee-Ankle: {int(hip_knee_ankle_angle)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(frame, f"Knee-Toe Dist: {knee_toe_distance:.2f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(frame, f"Shoulder-Hip H. Dist: {shoulder_hip_horizontal_distance:.2f}", (10, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Shoulder-Hip V. Dist: {shoulder_hip_vertical_distance:.2f}", (10, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return feedback




# Streamlit App
st.title("Real-Time Exercise Feedback")
st.sidebar.title("Settings")
video_source = st.sidebar.selectbox("Select Video Source", ("DroidCam USB", "Webcam", "Upload Video"))
if video_source == "DroidCam USB":
    st.sidebar.write("Ensure DroidCam is running and connected via USB.")
    cap = cv2.VideoCapture(1)
elif video_source == "Webcam":
    cap = cv2.VideoCapture(0)
elif video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded_file:
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file)

if st.sidebar.button("Start"):
    if 'cap' in locals() and cap and cap.isOpened():
        st_frame = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            if len(results[0].boxes):
                sorted_boxes = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)
                class_id = int(sorted_boxes[0].cls)
                exercise_type = {0: 'bicep curl', 1: 'push-up', 2: 'squat'}.get(class_id, 'Unknown')
            else:
                exercise_type = 'Unknown'
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            feedback = "Unknown"
            if pose_results.pose_landmarks:
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in pose_results.pose_landmarks.landmark]
                if exercise_type == "bicep curl":
                    feedback = check_bicep_curl_form(landmarks, frame)
                    angles = {}
                    try:
            # Get key landmarks
                        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                        head = landmarks[mp_pose.PoseLandmark.NOSE.value]  # Nose as head reference

                        # Add current elbow position to the buffer
                        elbow_position_buffer.append((elbow[0], elbow[1]))

                        # Check if the back is straight (i.e., shoulder, hip, and head are in alignment)
                        if abs(shoulder[0] - hip[0]) > 20 or abs(shoulder[1] - hip[1]) > 20:
                            feedback = "Bad Form: Keep your back straight"

                        # Check if the elbow is too far apart from the hips (horizontal distance between elbow and hip)
                        if abs(elbow[0] - hip[0]) > 100:  # Tune this threshold based on testing
                            feedback = "Bad Form: Keep your elbows closer to your body"

                        # Check if the elbow is stationary (compare with previous frame)
                        if len(elbow_position_buffer) > 1:
                            prev_elbow = elbow_position_buffer[-2]
                            # Calculate the distance moved between frames
                            distance_moved = np.linalg.norm(np.array([elbow[0] - prev_elbow[0], elbow[1] - prev_elbow[1]]))
                            if distance_moved > 15:  # Tune this threshold for allowed movement
                                feedback = "Bad Form: Elbow is moving too much"
                            else:
                                feedback = "Good Form: Elbow is stationary"

                        # Calculate the angle at the elbow
                        elbow_angle = calculate_angle(shoulder, elbow, wrist)
                        angles['elbow'] = elbow_angle

                        # Display the angle on the frame
                        elbow_coords = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                        cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (int(elbow_coords[0]), int(elbow_coords[1]) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Optional: Debugging coordinates
                        cv2.putText(frame, f"Elbow X: {int(elbow[0])}, Elbow Y: {int(elbow[1])}",
                                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"Shoulder X: {int(shoulder[0])}, Shoulder Y: {int(shoulder[1])}",
                                    (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    except IndexError:
                        "Unknown", {}
                    
                elif exercise_type == "push-up":
                    feedback = check_pushup_form(landmarks, frame)
                elif exercise_type == "squat":
                    feedback = check_squat_form(landmarks, frame)
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Exercise: {exercise_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            color = (0, 255, 0) if "Good Form" in feedback else (0, 0, 255)
            cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            st_frame.image(frame, channels="BGR", use_column_width=True)
        cap.release()
    else:
        st.error("No video source selected or invalid file!")
