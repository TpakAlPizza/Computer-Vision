# استيراد المكتبات المطلوبة لتتبع الكائنات والتعلم التفاعلي
import cv2
import numpy as np
import pickle
import os
import time
import sys
from collections import defaultdict, deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# فئة التتبع التفاعلي متعدد الكائنات
class InteractiveTracker:
    def __init__(self, model_path="tracking_model.pkl"):
        self.model_path = model_path
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.track_colors = {}
        self.next_object_id = 0
        self.objects_db = {}
        self.detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        self.min_contour_area = 500
        self.classes = self.load_coco_classes()
        self.corrections = []
        self.user_feedback = []
        self.learning_mode = False
        self.current_correction = None
        self.load_model()
        self.selected_box = None
        self.drawing_box = False
        self.start_point = None
        self.correction_label = ""
        self.show_help = True
        self.gui_available = self.check_gui()
        
    def load_coco_classes(self):
        classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
        return classes
    
    def check_gui(self):
        try:
            test_window = cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("test")
            return True
        except:
            return False
    
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.objects_db = saved_data.get('objects_db', {})
                    self.next_object_id = saved_data.get('next_id', 0)
                    self.corrections = saved_data.get('corrections', [])
                print(f"✓ Loaded model with {len(self.objects_db)} learned objects")
                return True
            except Exception as e:
                print(f"✗ Could not load model: {e}")
                return False
        return False
    
    def save_model(self):
        try:
            data = {
                'objects_db': self.objects_db,
                'next_id': self.next_object_id,
                'corrections': self.corrections,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Model saved with {len(self.objects_db)} objects")
            return True
        except Exception as e:
            print(f"✗ Could not save model: {e}")
            return False
    
    def detect_objects_simple(self, frame, confidence_threshold=0.3):
        height, width = frame.shape[:2]
        fg_mask = self.detector.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 30 and h > 30 and w < width * 0.8 and h < height * 0.8:
                    if w * h > 50000:
                        label = "large object"
                    elif w * h > 20000:
                        label = "medium object"
                    else:
                        label = "small object"
                    roi = frame[y:y+h, x:x+w]
                    feature = self.extract_features(roi)
                    boxes.append({
                        'box': (x, y, w, h),
                        'label': label,
                        'confidence': 0.7,
                        'feature': feature,
                        'class_id': -1
                    })
        return boxes
    
    def extract_features(self, image):
        if image is None or image.size == 0:
            return np.zeros(10)
        try:
            image = cv2.resize(image, (32, 32))
            avg_color = np.mean(image, axis=(0, 1))
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            feature = np.concatenate([avg_color, [edge_density]])
            return feature / 255.0
        except:
            return np.zeros(10)
    
    def match_with_existing(self, detection, max_distance=1.5):
        if detection['feature'] is None:
            return None, float('inf')
        best_match = None
        best_distance = float('inf')
        for obj_id, obj_data in self.objects_db.items():
            if obj_data.get('feature') is not None:
                try:
                    distance = np.linalg.norm(obj_data['feature'] - detection['feature'])
                    if distance < best_distance and distance < max_distance:
                        best_distance = distance
                        best_match = obj_id
                except:
                    continue
        return best_match, best_distance
    
    def update_tracking(self, detections):
        current_objects = {}
        for detection in detections:
            obj_id, distance = self.match_with_existing(detection)
            if obj_id is not None:
                current_objects[obj_id] = detection['box']
                self.track_history[obj_id].append(detection['box'])
                if detection['feature'] is not None and 'feature' in self.objects_db[obj_id]:
                    try:
                        alpha = 0.2
                        old_feature = self.objects_db[obj_id]['feature']
                        if old_feature is not None:
                            self.objects_db[obj_id]['feature'] = (
                                alpha * detection['feature'] + 
                                (1 - alpha) * old_feature
                            )
                    except:
                        self.objects_db[obj_id]['feature'] = detection['feature']
                if 'count' in self.objects_db[obj_id]:
                    self.objects_db[obj_id]['count'] += 1
                else:
                    self.objects_db[obj_id]['count'] = 1
            else:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.track_colors[obj_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255)
                )
                self.objects_db[obj_id] = {
                    'label': detection['label'],
                    'feature': detection['feature'],
                    'count': 1,
                    'first_seen': datetime.now()
                }
                current_objects[obj_id] = detection['box']
                self.track_history[obj_id].append(detection['box'])
        return current_objects
    
    def draw_tracking(self, frame, current_objects):
        for obj_id, box in current_objects.items():
            x, y, w, h = box
            obj_info = self.objects_db.get(obj_id, {})
            label = obj_info.get('label', 'Unknown')
            count = obj_info.get('count', 0)
            color = self.track_colors.get(obj_id, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_text = f"ID:{obj_id} {label}"
            if count > 0:
                label_text += f" ({count})"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - 25), (x + text_size[0], y), color, -1)
            cv2.putText(frame, label_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if self.selected_box:
            x, y, w, h = self.selected_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(frame, "Selected for correction", (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if self.correction_label:
            cv2.putText(frame, f"Label: {self.correction_label}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame
    
    def draw_ui_overlay(self, frame, fps):
        overlay = frame.copy()
        height, width = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (min(400, width), 200), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        instructions = [
            "INTERACTIVE MULTI-OBJECT TRACKER",
            f"FPS: {fps:.1f} | Objects: {len(self.objects_db)} | Corrections: {len(self.corrections)}",
            "",
            "CONTROLS:",
            "• Click & drag to select object",
            "• Type label, press ENTER to correct",
            "• 'h': Toggle help",
            "• 's': Save model",
            "• 'r': Reset selection",
            "• 'q': Quit",
            "",
            f"Mode: {'LEARNING' if self.learning_mode else 'TRACKING'}"
        ]
        for i, text in enumerate(instructions):
            y_pos = 30 + i * 20
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def process_without_gui(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"✗ Cannot open video source {video_source}")
            return
        
        print("\n" + "="*50)
        print("RUNNING IN HEADLESS MODE")
        print("="*50)
        print("\nControls via keyboard input:")
        print("1. Type 'd' to create a dummy detection")
        print("2. Type 'c' to see current objects")
        print("3. Type 's' to save model")
        print("4. Type 'q' to quit")
        print("\nStarting processing...")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            frame_count += 1
            detections = self.detect_objects_simple(frame)
            current_objects = self.update_tracking(detections)
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"\rFPS: {fps:.1f} | Objects: {len(current_objects)}", end="")
            
            try:
                import select
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    
                    if key == 'q':
                        print("\nQuitting...")
                        break
                    elif key == 's':
                        self.save_model()
                    elif key == 'c':
                        print(f"\nCurrent objects: {len(self.objects_db)}")
                        for obj_id, data in self.objects_db.items():
                            print(f"  ID:{obj_id} - {data.get('label', 'Unknown')}")
                    elif key == 'd':
                        height, width = frame.shape[:2]
                        dummy_box = (width//3, height//3, 100, 100)
                        dummy_feature = np.random.rand(10)
                        obj_id = self.next_object_id
                        self.next_object_id += 1
                        self.objects_db[obj_id] = {
                            'label': 'dummy_object',
                            'feature': dummy_feature,
                            'count': 1,
                            'first_seen': datetime.now()
                        }
                        print(f"\nAdded dummy object ID:{obj_id}")
            except:
                pass
        
        cap.release()
        self.save_model()
        print("\n✓ Processing complete. Model saved.")
    
    def process_with_gui(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"✗ Cannot open video source {video_source}")
            return
        
        cv2.namedWindow("Interactive Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Interactive Tracker", 800, 600)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.start_point = (x, y)
                self.drawing_box = True
                
            elif event == cv2.EVENT_LBUTTONUP:
                if self.drawing_box:
                    end_point = (x, y)
                    x1, y1 = self.start_point
                    x2, y2 = end_point
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    
                    if w > 20 and h > 20:
                        self.selected_box = (x, y, w, h)
                        print(f"\n✓ Box selected at: ({x}, {y}, {w}, {h})")
                        print("Type label for this object and press ENTER (ESC to cancel):")
                        self.learning_mode = True
                    
                    self.drawing_box = False
                    self.start_point = None
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.selected_box = None
                self.learning_mode = False
                self.correction_label = ""
                print("\n✗ Selection cancelled")
        
        cv2.setMouseCallback("Interactive Tracker", mouse_callback)
        
        print("\n" + "="*50)
        print("INTERACTIVE MULTI-OBJECT TRACKING")
        print("="*50)
        print("\nDrag mouse to select objects, type labels to teach the AI!")
        print("Starting tracking...")
        
        fps_counter = 0
        fps_time = time.time()
        last_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            self.last_frame = frame.copy()
            fps_counter += 1
            
            current_time = time.time()
            if current_time - fps_time >= 1.0:
                last_fps = fps_counter
                fps_counter = 0
                fps_time = current_time
            
            detections = self.detect_objects_simple(frame)
            current_objects = self.update_tracking(detections)
            frame = self.draw_tracking(frame, current_objects)
            
            if self.show_help:
                frame = self.draw_ui_overlay(frame, last_fps)
            
            cv2.imshow("Interactive Tracker", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if self.learning_mode:
                if 32 <= key <= 126:
                    self.correction_label += chr(key)
                elif key == 13:
                    if self.correction_label and self.selected_box:
                        self.apply_correction()
                elif key == 27:
                    self.learning_mode = False
                    self.selected_box = None
                    self.correction_label = ""
                    print("\n✗ Correction cancelled")
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_model()
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('r'):
                self.selected_box = None
                self.learning_mode = False
                self.correction_label = ""
                print("\n✓ Selection reset")
            elif key == ord('c'):
                print(f"\nCurrent database ({len(self.objects_db)} objects):")
                for obj_id, data in self.objects_db.items():
                    label = data.get('label', 'Unknown')
                    count = data.get('count', 0)
                    print(f"  ID:{obj_id:3d} - {label:20s} (seen {count:3d} times)")
        
        cap.release()
        cv2.destroyAllWindows()
        self.save_model()
        print("\n✓ Tracker stopped. Model saved.")
    
    def apply_correction(self):
        if not self.selected_box or not self.correction_label:
            return
        
        try:
            x, y, w, h = self.selected_box
            if hasattr(self, 'last_frame'):
                roi = self.last_frame[y:y+h, x:x+w]
                feature = self.extract_features(roi)
            else:
                feature = np.random.rand(10)
            
            obj_id = None
            min_distance = float('inf')
            
            for existing_id, obj_data in self.objects_db.items():
                if obj_data.get('feature') is not None:
                    distance = np.linalg.norm(obj_data['feature'] - feature)
                    if distance < min_distance and distance < 1.0:
                        min_distance = distance
                        obj_id = existing_id
            
            if obj_id is None:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.track_colors[obj_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255)
                )
            
            self.objects_db[obj_id] = {
                'label': self.correction_label,
                'feature': feature,
                'count': self.objects_db.get(obj_id, {}).get('count', 0) + 1,
                'first_seen': datetime.now()
            }
            
            self.corrections.append({
                'timestamp': datetime.now(),
                'box': self.selected_box,
                'label': self.correction_label,
                'object_id': obj_id
            })
            
            print(f"\n✓ Correction applied: Object {obj_id} = '{self.correction_label}'")
            
            self.selected_box = None
            self.correction_label = ""
            self.learning_mode = False
            
        except Exception as e:
            print(f"\n✗ Error applying correction: {e}")
            self.selected_box = None
            self.correction_label = ""
            self.learning_mode = False
    
    def process_video(self, video_source=0):
        if self.gui_available:
            self.process_with_gui(video_source)
        else:
            self.process_without_gui(video_source)
    
    def process_video_file(self, video_path):
        if not os.path.exists(video_path):
            print(f"✗ Video file not found: {video_path}")
            return
        
        if video_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            frame = cv2.imread(video_path)
            if frame is not None:
                print(f"Processing image: {video_path}")
                detections = self.detect_objects_simple(frame)
                current_objects = self.update_tracking(detections)
                frame = self.draw_tracking(frame, current_objects)
                output_path = f"result_{os.path.basename(video_path)}"
                cv2.imwrite(output_path, frame)
                print(f"✓ Result saved to: {output_path}")
                if self.gui_available:
                    cv2.imshow("Result", frame)
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
            else:
                print(f"✗ Could not read image: {video_path}")
        else:
            self.process_video(video_path)

# دالة رئيسية لتشغيل البرنامج
def main():
    print("="*60)
    print("INTERACTIVE MULTI-OBJECT TRACKER")
    print("="*60)
    print("\nThis system learns from your corrections!")
    print("When the tracker makes a mistake, you can teach it the right label.")
    
    tracker = InteractiveTracker()
    
    print("\nSelect input source:")
    print("1. Webcam (default)")
    print("2. Video file")
    print("3. Test with sample image")
    
    try:
        choice = input("Enter choice [1-3] (default: 1): ").strip()
    except:
        choice = "1"
    
    if choice == "2":
        video_path = input("Enter video file path: ").strip()
        if os.path.exists(video_path):
            tracker.process_video_file(video_path)
        else:
            print(f"✗ File not found: {video_path}")
            print("Using webcam instead...")
            tracker.process_video(0)
    elif choice == "3":
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.rectangle(test_image, (300, 150), (400, 250), (255, 0, 0), -1)
        cv2.imwrite("test_image.jpg", test_image)
        print("Created test_image.jpg")
        tracker.process_video_file("test_image.jpg")
    else:
        tracker.process_video(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("Thank you for using the Interactive Tracker!")
        print("="*60)
