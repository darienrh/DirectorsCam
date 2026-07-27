import cv2
import mediapipe as mp
import numpy as np

# ==========================================
# --- VARIABLES YOU CAN CHANGE ---
# ==========================================
CAMERA_INDEX = 0  # Change to 1 or tyo 2 if ur using an external USB webcam
MIN_DETECTION_CONFIDENCE = 0.7  # Higher = stricter hand detection (0.0 to 1.0)
MIN_TRACKING_CONFIDENCE = 0.7   # Higher = smoother tracking (0.0 to 1.0)
BOX_PADDING = 15  # Extra pixels of padding around the finger frame
DEFAULT_FILTER = 0  # Starting filter: 0=Cyberpunk Tint, 1=Invert, 2=Pixelate, 3=Thermal, 4=Neon Edges
# ==========================================

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

# Connect to Webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
current_filter = DEFAULT_FILTER
filter_names = ["Cyberpunk Tint", "Color Invert", "Pixelate", "Thermal Glow", "Neon Grid"]

def apply_filter(roi, filter_type):
    """Applies different visual effects to the ROI."""
    if roi.size == 0:
        return roi
    
    h, w, _ = roi.shape
    
    # 0: Cyberpunk Pink/Blue Tint (Like the video's neon purple look)
    if filter_type == 0:
        tinted = cv2.applyColorMap(roi, cv2.COLORMAP_COOL)
        return cv2.addWeighted(roi, 0.3, tinted, 0.7, 0)
        
    # 1: Color Inversion (Negative effect)
    elif filter_type == 1:
        return cv2.bitwise_not(roi)
        
    # 2: Pixelate / Mosaic (Like the grid effect in the video)
    elif filter_type == 2:
        pixel_size = max(8, int(w / 12))
        small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
    # 3: Thermal Glow
    elif filter_type == 3:
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
        
    # 4: Neon Edges / Grid
    elif filter_type == 4:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        colored_edges[np.where((colored_edges == [255, 255, 255]).all(axis=2))] = [255, 0, 255] # Pink edges
        return cv2.addWeighted(roi, 0.4, colored_edges, 0.8, 0)

    return roi

print("--- CONTROLS ---")
print("[F] - Cycle through filters")
print("[Q] - Quit program")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Flip the frame horizontally for a natural selfie-mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        frame_points = []
        
        # Extract thumb tips (landmark 4) and index finger tips (landmark 8) from both hands
        for hand_landmarks in results.multi_hand_landmarks:
            for index in [4, 8]: # 4 = Thumb tip, 8 = Index tip
                lm = hand_landmarks.landmark[index]
                px, py = int(lm.x * w), int(lm.y * h)
                frame_points.append((px, py))
        
        if len(frame_points) == 4:
            # Calculate bounding box coordinates around the 4 finger tips
            x_coords = [p[0] for p in frame_points]
            y_coords = [p[1] for p in frame_points]
            
            x_min = max(0, min(x_coords) - BOX_PADDING)
            y_min = max(0, min(y_coords) - BOX_PADDING)
            x_max = min(w, max(x_coords) + BOX_PADDING)
            y_max = min(h, max(y_coords) + BOX_PADDING)
            
            # Ensure the box has valid dimensions before applying filter
            if x_max > x_min + 10 and y_max > y_min + 10:
                # Extract the Region of Interest (ROI) between your hands
                roi = frame[y_min:y_max, x_min:x_max]
                
                # Apply the selected filter to the box
                filtered_roi = apply_filter(roi, current_filter)
                
                # Place the filtered region back into the main frame
                frame[y_min:y_max, x_min:x_max] = filtered_roi
                
                # Draw a sleek digital border around the frame (like a camera viewfinder)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                
                # Draw corner accents for a cyber/camera look
                corner_len = 15
                cv2.line(frame, (x_min, y_min), (x_min + corner_len, y_min), (255, 0, 255), 3)
                cv2.line(frame, (x_min, y_min), (x_min, y_min + corner_len), (255, 0, 255), 3)
                cv2.line(frame, (x_max, y_max), (x_max - corner_len, y_max), (255, 0, 255), 3)
                cv2.line(frame, (x_max, y_max), (x_max, y_max - corner_len), (255, 0, 255), 3)

    # Display current filter name on screen
    cv2.putText(frame, f"Filter: {filter_names[current_filter]} [Press 'F' to change]", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the video output
    cv2.imshow("Hand Frame Filter", frame)

    # Keyboard listening
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('f') or key == ord('F'):
        current_filter = (current_filter + 1) % len(filter_names)

# release resources
cap.release()
cv2.destroyAllWindows()