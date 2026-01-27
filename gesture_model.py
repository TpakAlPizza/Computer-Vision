import cv2
import numpy as np
import time
import math
from collections import deque, Counter

# استيراد المكتبات اللازمة لمعالجة الصور وتحليل الإيماءات

class HandIsolator:
    def __init__(self):
        self.lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        self.hand_zone_top = 0.65
        self.hand_zone_bottom = 0.95
        self.wrist_search_ratio = 0.3
        self.min_finger_length = 0.15

    def create_hand_zone_mask(self, frame_shape):
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        y_start = int(h * self.hand_zone_top)
        y_end = int(h * self.hand_zone_bottom)
        mask[y_start:y_end, :] = 255
        return mask, y_start, y_end

    def find_fingertips(self, contour):
        fingertips = []
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57
                        if d > 12000 and angle < 75:
                            fingertips.append(start)
                            fingertips.append(end)
                    if fingertips:
                        unique_fingertips = []
                        seen_points = set()
                        for point in fingertips:
                            point_key = (point[0] // 10, point[1] // 10)
                            if point_key not in seen_points:
                                unique_fingertips.append(point)
                                seen_points.add(point_key)
                        fingertips = unique_fingertips
        except Exception as e:
            pass
        return fingertips

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def find_wrist_point(self, contour, fingertips):
        if len(contour) < 100:
            return None
        
        contour_points = [tuple(point[0]) for point in contour]
        sorted_by_y = sorted(contour_points, key=lambda p: p[1], reverse=True)
        
        bottom_points = sorted_by_y[:max(10, len(sorted_by_y)//20)]
        
        if not bottom_points:
            return None
        
        avg_x = int(np.mean([p[0] for p in bottom_points]))
        avg_y = int(np.mean([p[1] for p in bottom_points]))
        
        wrist_y = avg_y
        wrist_x = avg_x
        
        return (wrist_x, wrist_y)

    def cut_at_wrist(self, contour, wrist_point):
        if wrist_point is None or len(contour) < 3:
            return contour
        
        wx, wy = wrist_point
        hand_points = []
        
        for point in contour:
            px, py = point[0]
            if py < wy - 10:
                hand_points.append(point)
        
        if len(hand_points) < 3:
            return contour
        
        hand_points_np = np.array(hand_points, dtype=np.int32)
        
        if len(hand_points_np) > 0:
            x_coords = [p[0][0] for p in hand_points]
            y_coords = [p[0][1] for p in hand_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            wrist_y = wy
            
            points_to_add = []
            for x in range(min_x, max_x + 1, 5):
                points_to_add.append(np.array([[x, wrist_y]], dtype=np.int32))
            
            if points_to_add:
                hand_points_np = np.vstack([hand_points_np] + points_to_add)
            
            return hand_points_np
        
        return contour

    def isolate_hand_only(self, frame):
        h, w = frame.shape[:2]
        zone_mask, y_start, y_end = self.create_hand_zone_mask(frame.shape)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        skin_mask = cv2.bitwise_and(skin_mask, zone_mask)
        
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None, None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 800:
            return None, None, None, None, None
        
        fingertips = self.find_fingertips(largest_contour)
        wrist_point = self.find_wrist_point(largest_contour, fingertips)
        
        if wrist_point is not None:
            wrist_point = (wrist_point[0], min(wrist_point[1], y_end - 10))
        
        hand_only_contour = self.cut_at_wrist(largest_contour, wrist_point)
        
        if hand_only_contour is not None and len(hand_only_contour) > 0:
            hand_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(hand_mask, [hand_only_contour], -1, 255, -1)
            
            x, y, w_rect, h_rect = cv2.boundingRect(hand_only_contour)
            
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_rect = min(w - x, w_rect + 2 * padding)
            h_rect = min(h - y, h_rect + 2 * padding)
            
            hand_roi = (x, y, w_rect, h_rect)
            
            return hand_mask, hand_roi, hand_only_contour, fingertips, wrist_point
        
        return None, None, None, None, None


# تحليل شكل اليد والتعرف على الإيماءات
class HandShapeAnalyzer:
    def __init__(self):
        self.gestures = {
            'open_palm': {'min_fingers': 4, 'max_fingers': 5, 'aspect_range': (0.7, 1.5)},
            'fist': {'min_fingers': 0, 'max_fingers': 1, 'aspect_range': (0.8, 1.3)},
            'peace': {'min_fingers': 2, 'max_fingers': 2, 'aspect_range': (0.9, 1.8)},
            'thumbs_up': {'min_fingers': 1, 'max_fingers': 1, 'aspect_range': (1.2, 2.5)},
            'pointing': {'min_fingers': 1, 'max_fingers': 1, 'aspect_range': (1.5, 3.0)},
            'ok': {'min_fingers': 2, 'max_fingers': 2, 'aspect_range': (0.8, 1.3)},
        }
        self.gesture_history = deque(maxlen=10)

    def analyze_hand_shape(self, hand_contour, fingertips):
        if hand_contour is None:
            return "no_hand", 0.0
        
        if fingertips:
            finger_count = len(fingertips)
        else:
            finger_count = self.estimate_fingers(hand_contour)
        
        finger_count = min(finger_count, 5)
        
        x, y, w, h = cv2.boundingRect(hand_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        area = cv2.contourArea(hand_contour)
        bounding_area = w * h
        compactness = area / bounding_area if bounding_area > 0 else 0
        
        best_gesture = "unknown"
        best_score = 0
        
        for gesture_name, template in self.gestures.items():
            score = 0
            
            if template['min_fingers'] <= finger_count <= template['max_fingers']:
                score += 2
            elif abs(finger_count - template['min_fingers']) <= 1:
                score += 1
            
            low, high = template['aspect_range']
            if low <= aspect_ratio <= high:
                score += 1
            
            if gesture_name == 'fist' and compactness > 0.75:
                score += 1
            elif gesture_name == 'open_palm' and 0.4 < compactness < 0.75:
                score += 1
            elif gesture_name in ['peace', 'thumbs_up', 'pointing', 'ok'] and 0.3 < compactness < 0.65:
                score += 1
            
            normalized_score = score / 4.0
            
            if normalized_score > best_score:
                best_score = normalized_score
                best_gesture = gesture_name
        
        self.gesture_history.append((best_gesture, best_score))
        
        if len(self.gesture_history) >= 5:
            recent = list(self.gesture_history)[-5:]
            gestures = [g for g, s in recent if s > 0.5]
            
            if gestures:
                gesture_counts = Counter(gestures)
                most_common = gesture_counts.most_common(1)
                if most_common:
                    best_gesture = most_common[0][0]
                    scores = [s for g, s in recent if g == best_gesture]
                    best_score = np.mean(scores) if scores else best_score
        
        return best_gesture, min(best_score, 1.0)

    def estimate_fingers(self, contour):
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    finger_count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 15000:
                            finger_count += 1
                    return min(finger_count + 1, 5)
        except:
            pass
        return 0


# نظام كامل للتعرف على إيماءات اليد
class HandGestureSystem:
    def __init__(self):
        self.hand_isolator = HandIsolator()
        self.shape_analyzer = HandShapeAnalyzer()
        self.show_fingertips = True
        self.show_wrist = True
        self.show_cut_line = True
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def process_frame(self, frame):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        
        hand_mask, hand_roi, hand_contour, fingertips, wrist_point = self.hand_isolator.isolate_hand_only(frame)
        
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


# تصوير وتحسين العرض المرئي
class EnhancedVisualizer:
    def __init__(self):
        self.show_debug = False

    def draw_detection(self, result):
        frame = result['frame'].copy()
        h, w = frame.shape[:2]
        
        isolator = HandIsolator()
        zone_mask, y_start, y_end = isolator.create_hand_zone_mask(frame.shape)
        
        cv2.rectangle(frame, (0, y_start), (w, y_end), (0, 255, 255), 2)
        cv2.putText(frame, "PUT HAND HERE", (10, y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if result['hand_contour'] is not None:
            cv2.drawContours(frame, [result['hand_contour']], -1, (0, 255, 0), 2)
            
            if result['hand_roi']:
                x, y, w_rect, h_rect = result['hand_roi']
                cv2.rectangle(frame, (x, y), (x+w_rect, y+h_rect), (255, 0, 0), 2)
            
            if result['fingertips'] and self.show_debug:
                for fingertip in result['fingertips']:
                    fx, fy = fingertip
                    cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "F", (fx-3, fy+3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if result['wrist_point'] and self.show_debug:
                wx, wy = result['wrist_point']
                cv2.circle(frame, (wx, wy), 10, (255, 0, 255), -1)
                cv2.putText(frame, "W", (wx-3, wy+3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.line(frame, (0, wy), (w, wy), (255, 0, 255), 2)
        
        if result['gesture'] != "no_hand":
            if result['confidence'] > 0.7:
                color = (0, 255, 0)
            elif result['confidence'] > 0.4:
                color = (0, 200, 200)
            else:
                color = (0, 0, 255)
            
            text = f"{result['gesture']} ({result['confidence']:.0%})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(frame, (10, 10), (20 + text_size[0], 40 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(frame, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        cv2.putText(frame, f"FPS: {result['fps']:.1f}", (w - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if result['hand_mask'] is not None:
            mask_resized = cv2.resize(result['hand_mask'], (100, 100))
            mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            frame[10:110, 10:110] = mask_colored
            cv2.rectangle(frame, (10, 10), (110, 110), (255, 255, 255), 1)
            cv2.putText(frame, "Hand Only", (15, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
    print("\n" + "="*60)
    print("HAND-ONLY DETECTION WITH WRIST CUTTING")
    print("="*60)
    print("This system CUTS OFF the arm at the wrist")
    print("It should detect ONLY the hand, not the arm")
    print("="*60)
    
    system = HandGestureSystem()
    visualizer = EnhancedVisualizer()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
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
        
        frame = cv2.flip(frame, 1)
        result = system.process_frame(frame)
        display = visualizer.draw_detection(result)
        cv2.imshow('Hand-Only Detection', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            visualizer.show_debug = not visualizer.show_debug
            print(f"Debug mode: {'ON' if visualizer.show_debug else 'OFF'}")
        
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
