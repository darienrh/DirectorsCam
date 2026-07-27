import cv2
import mediapipe as mp
import numpy as np
import time

# ==========================================
# --- VARIABLES YOU CAN CHANGE ---
# ==========================================
CAMERA_INDEX = 0  # Change to 1 or 2 if using an external USB webcam
MIN_DETECTION_CONFIDENCE = 0.7  # Higher = stricter hand detection (0.0 to 1.0)
MIN_TRACKING_CONFIDENCE = 0.7   # Higher = smoother tracking (0.0 to 1.0)
BOX_PADDING = 15  # Extra pixels of padding around the finger frame
DEFAULT_FILTER = 0  # Starting filter (0 to 7)
SWITCH_COOLDOWN = 1.5  # Seconds to wait before allowing another hand-gesture switch
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

# Initialize Webcam & State Variables
cap = cv2.VideoCapture(CAMERA_INDEX)
current_filter = DEFAULT_FILTER
last_switch_time = 0
switch_flash_endtime = 0

filter_names = [
    "Cyberpunk Tint", 
    "Color Invert", 
    "Pixelate", 
    "Thermal Glow", 
    "Neon Grid",
    "VHS Glitch (RGB Split)",
    "Matrix Night Vision",
    "Golden Hour Sepia"
]

def apply_filter(roi, filter_type):
    """Applies different visual effects to the region of interest (ROI)."""
    if roi.size == 0:
        return roi
    
    h, w, _ = roi.shape
    
    # 0: Cyberpunk Pink/Blue Tint
    if filter_type == 0:
        tinted = cv2.applyColorMap(roi, cv2.COLORMAP_COOL)
        return cv2.addWeighted(roi, 0.3, tinted, 0.7, 0)
        
    # 1: Color Invert (Negative effect)
    elif filter_type == 1:
        return cv2.bitwise_not(roi)
        
    # 2: Pixelate / Mosaic
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

    # 5: VHS Glitch / RGB Split
    elif filter_type == 5:
        shift = max(2, int(w / 30))
        b, g, r = cv2.split(roi)
        r_shifted = np.roll(r, shift, axis=1)   # Shift Red channel right
        b_shifted = np.roll(b, -shift, axis=1)  # Shift Blue channel left
        return cv2.merge([b_shifted, g, r_shifted])

    # 6: Matrix Night Vision
    elif filter_type == 6:
        b, g, r = cv2.split(roi)
        g = cv2.add(g, 50)  # Boost green
        b = cv2.multiply(b, 0.5).astype(np.uint8) # Dim blue
        r = cv2.multiply(r, 0.5).astype(np.uint8) # Dim red
        matrix = cv2.merge([b, g, r])
        matrix[::4, :] = (0, 50, 0)  # Add dark horizontal scanlines every 4 pixels
        return matrix

    # 7: Golden Hour Sepia
    elif filter_type == 7:
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(roi, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)

    return roi

print("--- CONTROLS ---")
print("GESTURE 1: Flip your hands upside down to change filter!")
print("GESTURE 2: Pop up your pinky finger while framing to change filter!")
print("[F] - Cycle through filters with keyboard")
print("[Q] - Quit program")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Flip the frame horizontally for a natural selfie-mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()

    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        frame_points = []
        gesture_detected = False
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract thumb tips (4) and index finger tips (8)
            for index in [4, 8]:
                lm = hand_landmarks.landmark[index]
                px, py = int(lm.x * w), int(lm.y * h)
                frame_points.append((px, py))
            
            # --- GESTURE DETECTION LOGIC ---
            # 1. Inverted Hand Check: Is Index tip (8) lower on screen than Wrist (0)?
            if hand_landmarks.landmark[8].y > hand_landmarks.landmark[0].y:
                gesture_detected = True
                
            # 2. Pinky Pop Check: Is Pinky tip (20) extended higher than Pinky knuckle (18)?
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
                gesture_detected = True
        
        # If a switch gesture is detected AND the cooldown timer has finished
        if gesture_detected and (current_time - last_switch_time > SWITCH_COOLDOWN):
            current_filter = (current_filter + 1) % len(filter_names)
            last_switch_time = current_time
            switch_flash_endtime = current_time + 0.4  # Flash green border for 0.4 seconds

        # Build and render the frame box
        if len(frame_points) == 4:
            x_coords = [p[0] for p in frame_points]
            y_coords = [p[1] for p in frame_points]
            
            x_min = max(0, min(x_coords) - BOX_PADDING)
            y_min = max(0, min(y_coords) - BOX_PADDING)
            x_max = min(w, max(x_coords) + BOX_PADDING)
            y_max = min(h, max(y_coords) + BOX_PADDING)
            
            if x_max > x_min + 10 and y_max > y_min + 10:
                roi = frame[y_min:y_max, x_min:x_max]
                filtered_roi = apply_filter(roi, current_filter)
                frame[y_min:y_max, x_min:x_max] = filtered_roi
                
                # Determine border color: Flash Green if switched recently, else White/Pink
                if current_time < switch_flash_endtime:
                    border_color = (0, 255, 0)  # Bright Green Flash
                    corner_color = (0, 255, 0)
                else:
                    border_color = (255, 255, 255) # Standard White
                    corner_color = (255, 0, 255)   # Cyber Pink

                # Draw viewfinder borders and corners
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), border_color, 2)
                corner_len = 15
                cv2.line(frame, (x_min, y_min), (x_min + corner_len, y_min), corner_color, 3)
                cv2.line(frame, (x_min, y_min), (x_min, y_min + corner_len), corner_color, 3)
                cv2.line(frame, (x_max, y_max), (x_max - corner_len, y_max), corner_color, 3)
                cv2.line(frame, (x_max, y_max), (x_max, y_max - corner_len), corner_color, 3)

    # Display current filter and instructions on screen
    status_text = f"Filter: {filter_names[current_filter]}"
    if current_time < switch_flash_endtime:
        status_text += " [GESTURE SWITCHED!]"
        
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Invert hands or pop pinky finger to change filter", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Show video output
    cv2.imshow("Hand Frame Filter", frame)

    # Keyboard fallback
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('f') or key == ord('F'):
        current_filter = (current_filter + 1) % len(filter_names)

# Cleanup
cap.release()
cv2.destroyAllWindows()