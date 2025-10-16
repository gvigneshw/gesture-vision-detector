import cv2
import mediapipe as mp
import numpy as np
import time
import os
from pathlib import Path

class GestureVisionDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh (for mouth landmarks) and Hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize detectors
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # FPS calculation
        self.prev_time = 0
        
        # Current gesture state (three states requested)
        self.current_gesture = "no_meme"
        self.face_detected = False
        
        # Load images for different gestures
        self.gesture_images = self.load_gesture_images()
        
    def load_gesture_images(self):
        """Load images for different gesture states"""
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)
        
        # Expect these images from the user; create placeholders if missing
        gesture_types = [
            "pointing", "pointing_mouth", "no_meme"
        ]
        
        images = {}
        for gesture in gesture_types:
            img_path = images_dir / f"{gesture}.jpg"
            if img_path.exists():
                images[gesture] = cv2.imread(str(img_path))
            else:
                # Create placeholder image if file doesn't exist
                images[gesture] = self.create_placeholder_image(gesture)
        
        return images
    
    def create_placeholder_image(self, gesture_name):
        """Create a placeholder image with text"""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 50
        
        # Add gradient effect
        for i in range(400):
            color_val = int(50 + (i / 400) * 100)
            img[i, :] = [color_val, color_val // 2, color_val // 3]
        
        # Add text
        text = gesture_name.replace("_", " ").upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text size and position
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (600 - text_size[0]) // 2
        text_y = (400 + text_size[1]) // 2
        
        # Add shadow
        cv2.putText(img, text, (text_x + 3, text_y + 3), font, 1.5, (0, 0, 0), 3)
        # Add main text
        cv2.putText(img, text, (text_x, text_y), font, 1.5, (255, 255, 255), 3)
        
        return img
    
    def count_fingers(self, hand_landmarks):
        """Count the number of extended fingers"""
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        thumb_tip = 4
        
        fingers_up = 0
        
        # Check thumb (different logic for left/right hand)
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            fingers_up += 1
        
        # Check other fingers
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers_up += 1
        
        return fingers_up
    
    def is_pointing(self, hand_landmarks):
        """Return True when only the index finger is clearly extended."""
        fingers = self.count_fingers(hand_landmarks)
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        return fingers == 1 and index_tip.y < middle_tip.y

    def _landmark_to_pixel(self, lm, frame_width, frame_height):
        return int(lm.x * frame_width), int(lm.y * frame_height)

    def _mouth_center_from_face_mesh(self, face_landmarks, frame_width, frame_height):
        """Estimate mouth center using upper/lower inner lip landmarks 13 and 14."""
        # Safety: ensure indices exist
        upper_idx, lower_idx = 13, 14
        upper = face_landmarks.landmark[upper_idx]
        lower = face_landmarks.landmark[lower_idx]
        cx = (upper.x + lower.x) * 0.5
        cy = (upper.y + lower.y) * 0.5
        return int(cx * frame_width), int(cy * frame_height)

    def _expanded_face_bbox(self, face_landmarks, frame_shape, expand_ratio=0.25):
        """Compute an expanded face bounding box from face mesh points."""
        h, w = frame_shape[:2]
        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
        ys = [int(lm.y * h) for lm in face_landmarks.landmark]
        if not xs or not ys:
            return None
        x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
        y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))
        # Expand bbox by ratio
        dx = int((x_max - x_min) * expand_ratio)
        dy = int((y_max - y_min) * expand_ratio)
        x_min = max(0, x_min - dx)
        x_max = min(w - 1, x_max + dx)
        y_min = max(0, y_min - dy)
        y_max = min(h - 1, y_max + dy)
        return (x_min, y_min, x_max, y_max)

    def is_index_touching_mouth(self, hand_landmarks, face_landmarks, frame_shape):
        """Return True if index fingertip is within a pixel threshold of mouth center."""
        h, w = frame_shape[:2]
        index_tip = hand_landmarks.landmark[8]
        ix, iy = self._landmark_to_pixel(index_tip, w, h)
        mx, my = self._mouth_center_from_face_mesh(face_landmarks, w, h)
        # Threshold proportional to frame size for robustness
        threshold = max(15, int(0.02 * max(w, h)))
        dx = ix - mx
        dy = iy - my
        return (dx * dx + dy * dy) ** 0.5 <= threshold
    
    def process_frame(self, frame):
        """Process a single frame for face and hand detection"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face (mesh) and hands
        face_results = self.face_mesh.process(rgb_frame)
        self.face_detected = face_results.multi_face_landmarks is not None
        
        # Detect hands
        hand_results = self.hands.process(rgb_frame)
        
        # Optionally draw expanded face bbox for debugging
        expanded_bbox = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            expanded_bbox = self._expanded_face_bbox(face_landmarks, frame.shape)
            if expanded_bbox:
                x1, y1, x2, y2 = expanded_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        # Determine three-state gesture based on pointing and mouth proximity
        new_state = "no_meme"
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # If any hand landmark enters the expanded face bbox, trigger mouth state
                if expanded_bbox is not None:
                    x1, y1, x2, y2 = expanded_bbox
                    for lm in hand_landmarks.landmark:
                        px, py = self._landmark_to_pixel(lm, frame.shape[1], frame.shape[0])
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            new_state = "pointing_mouth"
                            break
                    if new_state == "pointing_mouth":
                        break
                # Otherwise, if any hand is in pointing configuration, mark pointing
                if self.is_pointing(hand_landmarks):
                    new_state = "pointing"
        self.current_gesture = new_state
        
        return frame
    
    def get_display_image(self):
        """Get the image to display based on current state"""
        return self.gesture_images.get(self.current_gesture, 
                                       self.gesture_images["no_meme"])
    
    def calculate_fps(self):
        """Calculate and return current FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return fps
    
    def add_overlay_info(self, frame, fps):
        """Add FPS and status information to frame"""
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add text information
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        face_status = "DETECTED" if self.face_detected else "NOT DETECTED"
        face_color = (0, 255, 0) if self.face_detected else (0, 0, 255)
        cv2.putText(frame, f"Face: {face_status}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
        
        cv2.putText(frame, f"State: {self.current_gesture.upper()}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop for running the detector"""
        print("Starting Gesture Vision Detector...")
        print("Press 'q' to quit")
        print("\nStates:")
        print("- pointing")
        print("- pointing_mouth")
        print("- no_meme")
        
        cv2.namedWindow('Gesture Vision Detector', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Gesture Display', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Add overlay information
                display_frame = self.add_overlay_info(processed_frame, fps)
                
                # Get gesture image to display
                gesture_img = self.get_display_image()
                
                # Display frames
                cv2.imshow('Gesture Vision Detector', display_frame)
                cv2.imshow('Gesture Display', gesture_img)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nShutting down...")
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_detection.close()
        self.hands.close()
        print("Cleanup complete")

def main():
    try:
        detector = GestureVisionDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your webcam is connected and not being used by another application")
        print("2. Install required packages: pip install -r requirements.txt")
        print("3. Check if you have proper permissions to access the camera")

if __name__ == "__main__":
    main()