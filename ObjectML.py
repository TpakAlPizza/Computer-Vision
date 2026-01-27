# استيراد المكتبات
import cv2
import numpy as np
import pickle
import os
import time
import sys

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# فئة كاشف الأشياء البسيط
class SimpleObjectDetector:
    def __init__(self, use_matplotlib=True):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("Error: No camera found!")
                sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.cascade_files = {
            'face': 'haarcascade_frontalface_default.xml',
            'eye': 'haarcascade_eye.xml',
            'smile': 'haarcascade_smile.xml'
        }
        
        self.cascades = {}
        self.load_cascades()
        
        self.learned_objects = {}
        self.model_file = "objects_memory.pkl"
        self.load_objects()
        
        self.correction_mode = False
        self.selected_box = None
        self.correction_text = ""
        self.key_pressed = None
        self.running = True
        
        self.colors = {
            'face': (0, 255, 0),
            'eye': (255, 0, 0),
            'smile': (0, 255, 255),
            'learned': (255, 0, 255),
            'color': (0, 165, 255),
        }
        
        self.use_matplotlib = use_matplotlib and MATPLOTLIB_AVAILABLE
        
        print("=" * 60)
        print("OBJECT DETECTOR WITH LEARNING")
        print("=" * 60)
        print(f"\nLoaded {len(self.learned_objects)} learned objects")
        print("\nCONTROLS:")
        print("  c - Correct/Teach object")
        print("  Type - Enter object name when correcting")
        print("  ENTER - Save correction")
        print("  s - Show learned objects")
        print("  r - Reset memory")
        print("  q - Quit")
        print("\nPoint camera at objects and press 'c' to teach me!")
    
    def load_cascades(self):
        cascade_path = cv2.data.haarcascades
        
        for name, filename in self.cascade_files.items():
            full_path = os.path.join(cascade_path, filename)
            if os.path.exists(full_path):
                self.cascades[name] = cv2.CascadeClassifier(full_path)
                print(f"✓ Loaded {name} cascade")
            else:
                print(f"✗ Could not find {filename}")
                self.cascades[name] = None
    
    def load_objects(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.learned_objects = pickle.load(f)
                print(f"✓ Loaded {len(self.learned_objects)} learned objects")
            except:
                print("Starting with empty memory")
                self.learned_objects = {}
        else:
            print("Starting with empty memory")
            self.learned_objects = {}
    
    def save_objects(self):
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.learned_objects, f)
        except Exception as e:
            print(f"Error saving objects: {e}")
    
    def detect_faces(self, gray):
        if self.cascades.get('face'):
            faces = self.cascades['face'].detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        return []
    
    def detect_eyes(self, gray, face_region):
        if self.cascades.get('eye'):
            x, y, w, h = face_region
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.cascades['eye'].detectMultiScale(roi_gray)
            
            eyes_full = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
            return eyes_full
        return []
    
    def detect_smiles(self, gray, face_region):
        if self.cascades.get('smile'):
            x, y, w, h = face_region
            roi_gray = gray[y:y+h, x:x+w]
            smiles = self.cascades['smile'].detectMultiScale(
                roi_gray,
                scaleFactor=1.8,
                minNeighbors=20,
                minSize=(25, 25)
            )
            
            smiles_full = [(x + sx, y + sy, sw, sh) for (sx, sy, sw, sh) in smiles]
            return smiles_full
        return []
    
    def detect_color_blobs(self, frame):
        detections = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        colors = {
            'red': ([0, 120, 70], [10, 255, 255]),
            'blue': ([100, 150, 0], [140, 255, 255]),
            'green': ([36, 100, 100], [86, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
        }
        
        for color_name, (lower, upper) in colors.items():
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    is_duplicate = False
                    for det in detections:
                        dx, dy, dw, dh = det['box']
                        center_dist = np.sqrt(((x + w/2) - (dx + dw/2))**2 + ((y + h/2) - (dy + dh/2))**2)
                        if center_dist < 50:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        object_name = None
                        confidence = 0.0
                        
                        for learned_name, learned_data in self.learned_objects.items():
                            lx, ly, lw, lh = learned_data['position']
                            if (abs(x - lx) < 100 and abs(y - ly) < 100 and 
                                abs(w - lw) < 50 and abs(h - lh) < 50):
                                object_name = learned_name
                                confidence = 0.8
                                break
                        
                        detections.append({
                            'box': (x, y, w, h),
                            'type': 'color',
                            'name': object_name or f'{color_name} object',
                            'confidence': confidence or min(area / 10000, 0.9),
                            'color': color_name
                        })
        
        return detections
    
    def draw_detections(self, frame, detections):
        for detection in detections:
            x, y, w, h = detection['box']
            obj_type = detection['type']
            name = detection['name']
            confidence = detection.get('confidence', 0.0)
            
            if obj_type in self.colors:
                color = self.colors[obj_type]
            elif name in self.learned_objects:
                color = self.colors['learned']
            else:
                color = (200, 200, 200)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            if confidence > 0:
                label = f"{name} ({confidence:.1f})"
            else:
                label = name
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(frame, 
                         (x, y - text_height - 10),
                         (x + text_width + 10, y),
                         color, -1)
            
            cv2.putText(frame, label,
                       (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if self.correction_mode and self.selected_box == (x, y, w, h):
                cv2.rectangle(frame, (x-3, y-3), (x + w + 3, y + h + 3), (0, 0, 255), 3)
                cv2.putText(frame, "SELECTED", 
                           (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def draw_ui(self, frame):
        cv2.putText(frame, "Press 'c' to teach, ENTER to save, 'q' to quit", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        stats = f"Objects: {len(self.learned_objects)} learned"
        cv2.putText(frame, stats, 
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.correction_mode:
            cv2.rectangle(frame, (0, 30), (400, 100), (0, 0, 0), -1)
            
            lines = [
                "TEACHING MODE:",
                f"Click object, type name: '{self.correction_text}'",
                "Press ENTER to save"
            ]
            
            for i, line in enumerate(lines):
                color = (0, 0, 255) if i == 0 else (0, 255, 255)
                cv2.putText(frame, line, 
                           (10, 50 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        return frame
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()
        
        gray_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        all_detections = []
        
        faces = self.detect_faces(gray_cv)
        for (x, y, w, h) in faces:
            all_detections.append({
                'box': (x, y, w, h),
                'type': 'face',
                'name': 'Face',
                'confidence': 1.0
            })
            
            eyes = self.detect_eyes(gray_cv, (x, y, w, h))
            for (ex, ey, ew, eh) in eyes:
                all_detections.append({
                    'box': (ex, ey, ew, eh),
                    'type': 'eye',
                    'name': 'Eye',
                    'confidence': 1.0
                })
            
            smiles = self.detect_smiles(gray_cv, (x, y, w, h))
            for (sx, sy, sw, sh) in smiles:
                all_detections.append({
                    'box': (sx, sy, sw, sh),
                    'type': 'smile',
                    'name': 'Smile',
                    'confidence': 0.9
                })
        
        color_detections = self.detect_color_blobs(frame)
        all_detections.extend(color_detections)
        
        self.detections = all_detections
        
        display_frame = frame.copy()
        display_frame = self.draw_detections(display_frame, all_detections)
        display_frame = self.draw_ui(display_frame)
        
        return display_frame
    
    def handle_mouse_click(self, event):
        if not self.correction_mode or event.xdata is None or event.ydata is None:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        for detection in self.detections:
            dx, dy, dw, dh = detection['box']
            if dx <= x <= dx + dw and dy <= y <= dy + dh:
                self.selected_box = detection['box']
                print(f"✓ Selected object at ({dx}, {dy}) - {detection['name']}")
                return
    
    def learn_object(self, box, name):
        if not name.strip():
            return False
        
        x, y, w, h = box
        
        self.learned_objects[name.strip()] = {
            'position': (x, y, w, h),
            'learned_time': time.time(),
            'color': 'unknown'
        }
        
        self.save_objects()
        print(f"✓ Learned: '{name}'")
        return True
    
    def run_matplotlib(self):
        print("\nStarting with matplotlib display...")
        print("Close the plot window or press 'q' in the window to exit")
        
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.img_display = self.ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.ax.axis('off')
        self.fig.canvas.mpl_connect('button_press_event', self.handle_mouse_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        plt.ion()
        plt.show()
        
        try:
            while self.running:
                frame = self.process_frame()
                if frame is None:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self.img_display.set_data(frame_rgb)
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                
                if not plt.fignum_exists(self.fig.number):
                    self.running = False
                    break
                
                plt.pause(0.03)
                
        except Exception as e:
            print(f"Error in matplotlib display: {e}")
        finally:
            plt.close('all')
            self.cleanup()
    
    def run_no_gui(self):
        print("\nStarting detection without GUI...")
        print("Press Ctrl+C to stop")
        
        frame_count = 0
        try:
            while self.running:
                frame = self.process_frame()
                if frame is None:
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"\nFrame {frame_count}:")
                    print(f"  Detections: {len(self.detections)}")
                    print(f"  Learned objects: {len(self.learned_objects)}")
                    if self.correction_mode:
                        print(f"  Teaching: '{self.correction_text}'")
                
                self.check_terminal_input()
                
                time.sleep(0.03)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def on_key_press(self, event):
        if event.key is None:
            return
        
        key = event.key.lower()
        
        if key == 'q':
            self.running = False
        elif key == 'c':
            self.correction_mode = True
            self.correction_text = ""
            if self.detections:
                self.selected_box = self.detections[0]['box']
            print("\nTeaching mode activated!")
        elif key == 'enter':
            if self.correction_mode and self.selected_box and self.correction_text:
                self.learn_object(self.selected_box, self.correction_text)
                self.correction_mode = False
                self.correction_text = ""
        elif key == 's':
            print("\n" + "=" * 40)
            print("LEARNED OBJECTS:")
            print("=" * 40)
            if self.learned_objects:
                for name, data in self.learned_objects.items():
                    print(f"  • {name}")
            else:
                print("  None yet!")
            print("=" * 40)
        elif key == 'r':
            self.learned_objects = {}
            self.save_objects()
            print("✓ Memory cleared!")
        elif self.correction_mode and len(key) == 1:
            if key.isalnum() or key in [' ', '-', '_']:
                self.correction_text += key
            elif key == 'backspace':
                self.correction_text = self.correction_text[:-1]
    
    def check_terminal_input(self):
        try:
            import select
            
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                self.process_key(key)
                
        except:
            pass
    
    def process_key(self, key):
        if key == 'q':
            self.running = False
        elif key == 'c':
            self.correction_mode = True
            self.correction_text = ""
            print("\n=== TEACHING MODE ===")
            print("Selected first object. Type name and press ENTER.")
        elif key == '\n' or key == '\r':
            if self.correction_mode and self.selected_box and self.correction_text:
                self.learn_object(self.selected_box, self.correction_text)
                self.correction_mode = False
        elif key == 's':
            print("\n" + "=" * 40)
            print("LEARNED OBJECTS:")
            print("=" * 40)
            for name in self.learned_objects:
                print(f"  • {name}")
            print("=" * 40)
        elif key == 'r':
            self.learned_objects = {}
            self.save_objects()
            print("✓ Memory cleared!")
        elif self.correction_mode:
            if key.isalnum() or key in [' ', '-', '_']:
                self.correction_text += key
                print(f"Name: {self.correction_text}")
            elif ord(key) == 127:
                self.correction_text = self.correction_text[:-1]
                print(f"Name: {self.correction_text}")
    
    def cleanup(self):
        self.cap.release()
        if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
            plt.close('all')
        print("\n✓ Cleaned up. Goodbye!")
    
    def run(self):
        if self.use_matplotlib:
            self.run_matplotlib()
        else:
            self.run_no_gui()

# الدالة الرئيسية
def main():
    use_matplotlib = True
    if not MATPLOTLIB_AVAILABLE:
        print("Note: matplotlib not found, running in console mode")
        use_matplotlib = False
    
    detector = SimpleObjectDetector(use_matplotlib=use_matplotlib)
    detector.run()

if __name__ == "__main__":
    main()
