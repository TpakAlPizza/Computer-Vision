"""
HAND-ONLY Detection with Wrist Cutting
Specifically isolates the hand by finding fingers and cutting at the wrist
"""

import cv2
import numpy as np
import time
import math
from collections import deque, Counter

class HandIsolator:
    """Isolates only the hand by finding fingers and cutting at wrist"""
    
    def __init__(self):
        # Skin detection parameters (tight for hand-only)
        self.lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        
        # VERY restrictive zone - bottom 35% only
        self.hand_zone_top = 0.65  # Start at 65% from top
        self.hand_zone_bottom = 0.95  # End at 95%
        
        # For wrist detection
        self.wrist_search_ratio = 0.3  # Search wrist in bottom 30% of contour
        self.min_finger_length = 0.15  # Minimum finger length relative to hand height
        
    def create_hand_zone_mask(self, frame_shape):
        """Create mask where hand is allowed to be"""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        y_start = int(h * self.hand_zone_top)
        y_end = int(h * self.hand_zone_bottom)
        
        mask[y_start:y_end, :] = 255
        return mask, y_start, y_end
    
    def find_fingertips(self, contour):
        """Find fingertips in contour"""
        fingertips = []
        
        try:
            # Get convex hull defects (valleys between fingers)
            hull = cv2.convexHull(contour, returnPoints=False)
            
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                
                if defects is not None:
                    # Find defect points (valleys between fingers)
                    valleys = []
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])  # Finger start
                        end = tuple(contour[e][0])    # Finger end
                        far = tuple(contour[f][0])    # Valley point
                        
                        # Calculate angle at valley
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        
                        angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57
                        
                        # Valid finger valley
                        if d > 10000 and angle < 80:  # Strict threshold
                            valleys.append((start, end, far, angle))
                    
                    # Extract fingertips from valleys
                    if len(valleys) >= 2:  # Need at least 2 valleys for fingers
                        for start, end, far, angle in valleys:
                            # The "start" and "end" points are fingertips
                            fingertips.append(start)
                            fingertips.append(end)
                    
                    # Remove duplicates (points might be shared between valleys)
                    if fingertips:
                        # Keep only unique fingertips
                        unique_fingertips = []
                        for point in fingertips:
                            if not any(self.distance(point, up) < 10 for up in unique_fingertips):
                                unique_fingertips.append(point)
                        fingertips = unique_fingertips
                        
        except Exception as e:
            print(f"Fingertip detection error: {e}")
        
        return fingertips
    
    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def find_wrist_point(self, contour, fingertips):
        """Find wrist point by analyzing contour shape"""
        if len(fingertips) < 2:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Find lowest point of contour (likely wrist)
        lowest_point = None
        lowest_y = 0
        
        for point in contour:
            py = point[0][1]
            if py > lowest_y:
                lowest_y = py
                lowest_point = tuple(point[0])
        
        if lowest_point:
            # Find points on either side of the wrist
            wrist_candidates = []
            
            # Look for points near the bottom of the contour
            for point in contour:
                p = tuple(point[0])
                # Points near the bottom (within 10% of height from bottom)
                if p[1] > y + h * 0.9:
                    # Check if this point is "inside" the hand (not a fingertip)
                    is_fingertip = any(self.distance(p, ft) < 20 for ft in fingertips)
                    if not is_fingertip:
                        wrist_candidates.append(p)
            
            if wrist_candidates:
                # Take the average of wrist candidates
                avg_x = int(np.mean([p[0] for p in wrist_candidates]))
                avg_y = int(np.mean([p[1] for p in wrist_candidates]))
                return (avg_x, avg_y)
        
        return lowest_point
    
    def cut_at_wrist(self, contour, wrist_point):
        """Cut the contour at the wrist point"""
        if wrist_point is None:
            return contour
        
        # Find the point in contour closest to wrist point
        min_dist = float('inf')
        wrist_idx = 0
        
        for i, point in enumerate(contour):
            p = tuple(point[0])
            dist = self.distance(p, wrist_point)
            if dist < min_dist:
                min_dist = dist
                wrist_idx = i
        
        # Create a line at the wrist (horizontal cut)
        wx, wy = wrist_point
        
        # Create new contour with points above the wrist
        hand_contour = []
        for point in contour:
            px, py = point[0]
            if py < wy:  # Only keep points above wrist
                hand_contour.append(point)
        
        if hand_contour:
            # Add wrist points to close the contour
            # Add points along a horizontal line at wrist level
            wrist_y = wy
            min_x = min(p[0][0] for p in hand_contour)
            max_x = max(p[0][0] for p in hand_contour)
            
            # Add bottom line to close the contour
            for x in range(min_x, max_x, 5):
                hand_contour.append(np.array([[x, wrist_y]], dtype=np.int32))
            
            return np.array(hand_contour, dtype=np.int32)
        
        return contour
    
    def isolate_hand_only(self, frame):
        """Main function: isolate only the hand (cut at wrist)"""
        h, w = frame.shape[:2]
        
        # Create hand zone mask
        zone_mask, y_start, y_end = self.create_hand_zone_mask(frame.shape)
        
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin detection
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply zone mask
        skin_mask = cv2.bitwise_and(skin_mask, zone_mask)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        # Find largest contour in hand zone
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Skip if too small
        if area < 1000:
            return None, None, None
        
        # Find fingertips
        fingertips = self.find_fingertips(largest_contour)
        
        # Find wrist point
        wrist_point = self.find_wrist_point(largest_contour, fingertips)
        
        # Cut contour at wrist
        hand_only_contour = self.cut_at_wrist(largest_contour, wrist_point)
        
        if hand_only_contour is not None and len(hand_only_contour) > 0:
            # Create mask for hand only
            hand_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(hand_mask, [hand_only_contour], -1, 255, -1)
            
            # Get bounding box
            x, y, w_rect, h_rect = cv2.boundingRect(hand_only_contour)
            
            # Add padding
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_rect = min(w - x, w_rect + 2 * padding)
            h_rect = min(h - y, h_rect + 2 * padding)
            
            hand_roi = (x, y, w_rect, h_rect)
            
            return hand_mask, hand_roi, hand_only_contour, fingertips, wrist_point
        
        return None, None, None, None, None

class HandShapeAnalyzer:
    """Analyzes hand shape for gesture recognition"""
    
    def __init__(self):
        # Gesture definitions based on finger configuration
        self.gestures = {
            'open_palm': {'min_fingers': 4, 'max_fingers': 5, 'aspect_range': (0.7, 1.5)},
            'fist': {'min_fingers': 0, 'max_fingers': 1, 'aspect_range': (0.8, 1.3)},
            'peace': {'min_fingers': 2, 'max_fingers': 2, 'aspect_range': (0.9, 1.8)},
            'thumbs_up': {'min_fingers': 1, 'max_fingers': 1, 'aspect_range': (1.2, 2.5)},
            'pointing': {'min_fingers': 1, 'max_fingers': 1, 'aspect_range': (1.5, 3.0)},
            'ok': {'min_fingers': 1, 'max_fingers': 1, 'aspect_range': (0.8, 1.3)},
        }
        
        self.gesture_history = deque(maxlen=10)
    
    def analyze_hand_shape(self, hand_contour, fingertips):
        """Analyze hand shape and recognize gesture"""
        if hand_contour is None:
            return "no_hand", 0.0
        
        # Count fingers (use provided fingertips or estimate)
        if fingertips:
            finger_count = len(fingertips)
        else:
            finger_count = self.estimate_fingers(hand_contour)
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(hand_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate area ratio (hand compactness)
        area = cv2.contourArea(hand_contour)
        bounding_area = w * h
        compactness = area / bounding_area if bounding_area > 0 else 0
        
        # Match to gesture
        best_gesture = "unknown"
        best_score = 0
        
        for gesture_name, template in self.gestures.items():
            score = 0
            
            # Check finger count
            if template['min_fingers'] <= finger_count <= template['max_fingers']:
                score += 2
            elif abs(finger_count - template['min_fingers']) <= 1:
                score += 1
            
            # Check aspect ratio
            low, high = template['aspect_range']
            if low <= aspect_ratio <= high:
                score += 1
            
            # Check compactness (different gestures have different compactness)
            if gesture_name == 'fist' and compactness > 0.8:
                score += 1
            elif gesture_name == 'open_palm' and 0.5 < compactness < 0.8:
                score += 1
            elif gesture_name in ['peace', 'thumbs_up', 'pointing'] and 0.4 < compactness < 0.7:
                score += 1
            
            # Normalize score
            normalized_score = score / 4.0  # Max possible score is 4
            
            if normalized_score > best_score:
                best_score = normalized_score
                best_gesture = gesture_name
        
        # Add to history
        self.gesture_history.append((best_gesture, best_score))
        
        # Apply temporal smoothing
        if len(self.gesture_history) >= 5:
            recent = list(self.gesture_history)[-5:]
            gestures = [g for g, s in recent if s > 0.4]
            
            if gestures:
                gesture_counts = Counter(gestures)
                most_common = gesture_counts.most_common(1)
                if most_common:
                    best_gesture = most_common[0][0]
                    # Get average score for this gesture
                    scores = [s for g, s in recent if g == best_gesture]
                    best_score = np.mean(scores) if scores else best_score
        
        return best_gesture, min(best_score, 1.0)
    
    def estimate_fingers(self, contour):
        """Estimate finger count from contour"""
        try:
            # Get convex hull defects
            hull = cv2.convexHull(contour, returnPoints=False)
            
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                
                if defects is not None:
                    finger_count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 15000:  # High threshold to avoid noise
                            finger_count += 1
                    
                    return min(finger_count, 5)
        except:
            pass
        
        return 0

class HandGestureSystem:
    """Complete hand gesture system with wrist cutting"""
    
    def __init__(self):
        self.hand_isolator = HandIsolator()
        self.shape_analyzer = HandShapeAnalyzer()
        
        # Display options
        self.show_fingertips = True
        self.show_wrist = True
        self.show_cut_line = True
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_frame(self, frame):
        """Process a frame and return results"""
        self.frame_count += 1
        
        # Calculate FPS
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        
        # Isolate hand (cut at wrist)
        hand_mask, hand_roi, hand_contour, fingertips, wrist_point = self.hand_isolator.isolate_hand_only(frame)
        
        # Recognize gesture
        gesture = "no_hand"
        confidence = 0.0
        
        if hand_contour is not None:
            gesture, confidence = self.shape_analyzer.analyze_hand_shape(hand_contour, fingertips)
        
        return {
            'frame': frame,
            'hand_mask': hand_mask,
            'hand_roi': hand_roi,
            'hand_contour': hand_contour,
            'fingertips': fingertips,
            'wrist_point': wrist_point,
            'gesture': gesture,
            'confidence': confidence,
            'fps': self.fps
        }

class EnhancedVisualizer:
    """Enhanced visualization with clear feedback"""
    
    def __init__(self):
        self.show_debug = False
    
    def draw_detection(self, result):
        """Draw detection results on frame"""
        frame = result['frame'].copy()
        h, w = frame.shape[:2]
        
        # Draw hand zone
        isolator = HandIsolator()
        zone_mask, y_start, y_end = isolator.create_hand_zone_mask(frame.shape)
        
        cv2.rectangle(frame, (0, y_start), (w, y_end), (0, 255, 255), 2)
        cv2.putText(frame, "PUT HAND HERE", (10, y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw hand detection if found
        if result['hand_contour'] is not None:
            # Draw hand contour (in green)
            cv2.drawContours(frame, [result['hand_contour']], -1, (0, 255, 0), 2)
            
            # Draw bounding box
            if result['hand_roi']:
                x, y, w_rect, h_rect = result['hand_roi']
                cv2.rectangle(frame, (x, y), (x+w_rect, y+h_rect), (255, 0, 0), 2)
            
            # Draw fingertips
            if result['fingertips'] and self.show_debug:
                for fingertip in result['fingertips']:
                    fx, fy = fingertip
                    cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "F", (fx-3, fy+3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw wrist point
            if result['wrist_point'] and self.show_debug:
                wx, wy = result['wrist_point']
                cv2.circle(frame, (wx, wy), 10, (255, 0, 255), -1)
                cv2.putText(frame, "W", (wx-3, wy+3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw wrist cut line
                if self.show_cut_line:
                    cv2.line(frame, (0, wy), (w, wy), (255, 0, 255), 2)
        
        # Draw gesture information
        if result['gesture'] != "no_hand":
            # Color code by confidence
            if result['confidence'] > 0.7:
                color = (0, 255, 0)  # Green
            elif result['confidence'] > 0.4:
                color = (0, 200, 200)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            text = f"{result['gesture']} ({result['confidence']:.0%})"
            
            # Text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(frame, (10, 10), (20 + text_size[0], 40 + text_size[1]), (0, 0, 0), -1)
            
            cv2.putText(frame, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {result['fps']:.1f}", (w - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw mask preview
        if result['hand_mask'] is not None:
            mask_resized = cv2.resize(result['hand_mask'], (100, 100))
            mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            frame[10:110, 10:110] = mask_colored
            cv2.rectangle(frame, (10, 10), (110, 110), (255, 255, 255), 1)
            cv2.putText(frame, "Hand Only", (15, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw instructions
        instructions = [
            "INSTRUCTIONS:",
            "1. Place ONLY your hand in yellow zone",
            "2. Show palm clearly to camera",
            "3. Keep wrist near bottom of zone",
            "4. Make clear gestures:",
            "   - Open palm: All fingers extended",
            "   - Fist: Make tight fist",
            "   - Peace: Index + middle up",
            "   - Pointing: Just index finger",
            "",
            "CONTROLS:",
            "D - Toggle debug view",
            "Q - Quit"
        ]
        
        for i, line in enumerate(instructions):
            y_pos = h - 200 + i * 15
            font_size = 0.35 if i > 3 else 0.4
            color = (200, 200, 255) if i > 3 else (255, 255, 200)
            
            cv2.putText(frame, line, (w - 250, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
        
        return frame

def main():
    """Main application"""
    print("\n" + "="*60)
    print("HAND-ONLY DETECTION WITH WRIST CUTTING")
    print("="*60)
    print("This system CUTS OFF the arm at the wrist")
    print("It should detect ONLY the hand, not the arm")
    print("="*60)
    
    # Initialize system
    system = HandGestureSystem()
    visualizer = EnhancedVisualizer()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nCRITICAL INSTRUCTIONS:")
    print("1. Place your hand in the YELLOW zone")
    print("2. Position your WRIST at the BOTTOM of the yellow zone")
    print("3. The system will CUT OFF everything below the wrist")
    print("4. Show PALM clearly to camera (not side view)")
    print("5. Make exaggerated gestures")
    print("\nPress D to see debug info (fingertips, wrist)")
    print("Press Q to quit")
    print("="*60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Process frame
        result = system.process_frame(frame)
        
        # Draw visualization
        display = visualizer.draw_detection(result)
        
        # Show result
        cv2.imshow('Hand-Only Detection', display)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            visualizer.show_debug = not visualizer.show_debug
            print(f"Debug mode: {'ON' if visualizer.show_debug else 'OFF'}")
        
        # Print detection info occasionally
        if system.frame_count % 30 == 0 and result['gesture'] != "no_hand":
            if result['fingertips']:
                print(f"Detected {len(result['fingertips'])} fingers: {result['gesture']} ({result['confidence']:.0%})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Session ended")
    print("="*60)

if __name__ == "__main__":
    main()