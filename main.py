import os
import time
import math

import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'hand_landmarker.task')


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
latest_result = None
current_gesture = None


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def get_distance(point1, point2):
    """Calculate Euclidean distance between two landmarks."""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)


def is_finger_extended(landmarks, finger_tip_idx, finger_pip_idx, finger_mcp_idx):
    """Check if a finger is extended based on landmark positions."""
    tip = landmarks[finger_tip_idx]
    pip = landmarks[finger_pip_idx]
    mcp = landmarks[finger_mcp_idx]
    
    # Finger is extended if tip is farther from wrist than pip
    wrist = landmarks[0]
    tip_to_wrist = get_distance(tip, wrist)
    pip_to_wrist = get_distance(pip, wrist)
    
    return tip_to_wrist > pip_to_wrist


def is_thumb_extended(landmarks):
    """Check if thumb is extended."""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]
    
    # Thumb is extended if tip is farther from wrist than IP joint
    tip_to_wrist = get_distance(thumb_tip, wrist)
    ip_to_wrist = get_distance(thumb_ip, wrist)
    
    return tip_to_wrist > ip_to_wrist


def detect_gesture(hand_landmarks):
    """Detect gesture based on hand landmarks."""
    landmarks = hand_landmarks
    
    # Check each finger's state
    thumb_extended = is_thumb_extended(landmarks)
    index_extended = is_finger_extended(landmarks, 8, 6, 5)
    middle_extended = is_finger_extended(landmarks, 12, 10, 9)
    ring_extended = is_finger_extended(landmarks, 16, 14, 13)
    pinky_extended = is_finger_extended(landmarks, 20, 18, 17)
    
    # Count extended fingers
    fingers_up = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    
    # Detect specific gestures
    # Fist - all fingers closed
    if fingers_up == 0:
        return "FIST"
    
    # Open Palm - all fingers extended
    if fingers_up == 5:
        return "OPEN PALM"
    
    # Thumbs Up - only thumb extended
    if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        # Check if thumb is pointing up
        if landmarks[4].y < landmarks[2].y:
            return "THUMBS UP"
        else:
            return "THUMBS DOWN"
    
    # Pointing - only index finger extended
    if not thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "POINTING"
    
    # Peace/Victory - index and middle fingers extended
    if not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "PEACE"
    
    # Rock/Horn - index and pinky extended, thumb can be extended or not
    if index_extended and not middle_extended and not ring_extended and pinky_extended:
        return "ROCK"
    
    # OK Sign - thumb and index finger touch, others extended
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = get_distance(thumb_tip, index_tip)
    
    if distance < 0.05 and middle_extended and ring_extended and pinky_extended:
        return "OK"
    
    # Three fingers (counting gesture)
    if thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "THREE"
    
    # Four fingers
    if not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
        return "FOUR"
    
    # Generic counting
    return f"{fingers_up} FINGERS"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker:
    # Continuously capture images from the webcam and feed them into the hand landmarker:
    import cv2

    def draw_hand_landmarks(frame, hand_landmarks):
        height, width = frame.shape[:2]
        points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Standard MediaPipe hand connections (21 landmarks).
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 0, 255), 2)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        image.flags.writeable = True
        
        # Draw landmarks and detect gestures
        if latest_result and latest_result.hand_landmarks:
            gestures = []
            for hand_landmarks in latest_result.hand_landmarks:
                draw_hand_landmarks(image, hand_landmarks)
                gesture = detect_gesture(hand_landmarks)
                gestures.append(gesture)
            
            # Display detected gestures on screen
            y_offset = 50
            for i, gesture in enumerate(gestures):
                text = f"Hand {i+1}: {gesture}"
                cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 0), 2, cv2.LINE_AA)
                y_offset += 50
        
        cv2.imshow('Gesture Detector', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()