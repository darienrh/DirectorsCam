import cv2
import mediapipe as mp
import numpy as np
import time

# ==========================================
# --- VARIABLES YOU CAN CHANGE ---
# ==========================================
CAMERA_INDEX = 0  # Change to 1 or 2 if using an external USB webcam
MIN_DETECTION_CONFIDENCE = 0.7  
MIN_TRACKING_CONFIDENCE = 0.7   
BOX_PADDING = 15  
DEFAULT_FILTER = 0  
BORDER_THICKNESS = 1  # Thinner border line (default was 2)
CORNER_THICKNESS = 2  # Thinner corner accents (default was 3)
# ==========================================

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

# Initialize Variables
cap = cv2.VideoCapture(CAMERA_INDEX)
current_filter = DEFAULT_FILTER
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
    
    if filter_type == 0:
        tinted = cv2.applyColorMap(roi, cv2.COLORMAP_COOL)
        return cv2.addWeighted(roi, 0.3, tinted, 0.7, 0)
    elif filter_type == 1:
        return cv2.bitwise_not(roi)
    elif filter_type == 2:
        pixel_size = max(8, int(w / 12))
        small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif filter_type == 3:
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
    elif filter_type == 4:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        colored_edges[np.where((colored_edges == [255, 255, 255]).all(axis=2))] = [255, 0, 255]
        return cv2.addWeighted(roi, 0.4, colored_edges, 0.8, 0)
    elif filter_type == 5:
        shift = max(2, int(w / 30))
        b, g, r = cv2.split(roi)
        r_shifted = np.roll(r, shift, axis=1)
        b_shifted = np.roll(b, -shift, axis=1)
        return cv2.merge([b_shifted, g, r_shifted])
    elif filter_type == 6:
        b, g, r = cv2.split(roi)
        g = cv2.add(g, 50)
        b = cv2.multiply(b, 0.5).astype(np.uint8)
        r = cv2.multiply(r, 0.5).astype(np.uint8)
        matrix = cv2.merge([b, g, r])
        matrix[::4, :] = (0, 50, 0)
        return matrix
    elif filter_type == 7:
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(roi, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)

    return roi

# Mouse Callback Function to change filters on click
def mouse_click_callback(event, x, y, flags, param):
    global current_filter, switch_flash_endtime
    if event == cv2.EVENT_LBUTTONDOWN:
        current_filter = (current_filter + 1) % len(filter_names)
        switch_flash_endtime = time.time() + 0.3  # Flash green for 0.3 seconds

# Setup full screen window display
window_name = "Hand Frame Filter"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, mouse_click_callback)

print("--- CONTROLS ---")
print("-> LEFT CLICK anywhere on the screen to change filters!")
print("-> Press [Q] or [ESC] to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        frame_points = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            for index in [4, 8]:
                lm = hand_landmarks.landmark[index]
                px, py = int(lm.x * w), int(lm.y * h)
                frame_points.append((px, py))
        
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
                
                # Visual logic for standard view vs click flash
                if current_time < switch_flash_endtime:
                    border_color = (0, 255, 0)  # Flash Green
                    corner_color = (0, 255, 0)  # Flash Green
                else:
                    border_color = (255, 255, 255) # Sleek White Frame
                    corner_color = (0, 0, 0)       # Solid Black Corners

                # Draw Viewfinder Box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), border_color, BORDER_THICKNESS)
                
                # Draw Viewfinder Corners
                corner_len = 15
                cv2.line(frame, (x_min, y_min), (x_min + corner_len, y_min), corner_color, CORNER_THICKNESS)
                cv2.line(frame, (x_min, y_min), (x_min, y_min + corner_len), corner_color, CORNER_THICKNESS)
                cv2.line(frame, (x_max, y_max), (x_max - corner_len, y_max), corner_color, CORNER_THICKNESS)
                cv2.line(frame, (x_max, y_max), (x_max, y_max - corner_len), corner_color, CORNER_THICKNESS)

    # Display clean overlay UI
    cv2.putText(frame, f"Filter: {filter_names[current_filter]}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, frame)

    # Quit keys fallback
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27: # 27 is the Escape key
        break

cap.release()
cv2.destroyAllWindows()