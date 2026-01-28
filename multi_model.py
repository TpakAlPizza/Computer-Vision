# استيراد المكتبات المطلوبة
import cv2
import numpy as np
import pickle
import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from collections import defaultdict, deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# تعريف نموذج الشبكة العصبية لاستخراج الميزات
class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super(FeatureExtractor, self).__init__()
        # استخدام ResNet مسبق التدريب
        self.backbone = models.resnet18(pretrained=True)
        # إزالة الطبقة الأخيرة (fully connected)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # طبقة إضافية لتقليل الأبعاد
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        # تطبيع الميزات
        features = nn.functional.normalize(features, p=2, dim=1)
        return features

# فئة كاملة للتعرف على الكائنات باستخدام YOLO
class ObjectDetector:
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        
        # تحميل YOLO (يفضل استخدام YOLOv8 أو YOLOv5)
        try:
            # محاولة تحميل YOLO من OpenCV DNN
            self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
            self.classes = self.load_coco_classes()
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            self.yolo_available = True
        except:
            print("⚠ YOLO weights not found, using simple detector")
            self.yolo_available = False
            
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
    
    def detect_with_yolo(self, frame):
        height, width = frame.shape[:2]
        
        # تحضير الصورة للنموذج
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # إحداثيات الصندوق
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # الزاوية العلوية اليسرى
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # تطبيق Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                
                detections.append({
                    'box': (x, y, w, h),
                    'label': label,
                    'confidence': confidences[i],
                    'class_id': class_ids[i]
                })
        
        return detections
    
    def detect_simple(self, frame):
        """كشف بسيط إذا لم يتوفر YOLO"""
        height, width = frame.shape[:2]
        
        # تطبيق مرشح الخلفية
        fg_mask = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50).apply(frame)
        
        # تحسين القناع
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # زيادة الحد الأدنى للمساحة
                x, y, w, h = cv2.boundingRect(contour)
                if w > 40 and h > 40:  # زيادة الحد الأدنى للأبعاد
                    boxes.append({
                        'box': (x, y, w, h),
                        'label': 'object',
                        'confidence': 0.5,
                        'class_id': -1
                    })
        
        return boxes

# فئة التتبع التفاعلي متعدد الكائنات مع NN
class InteractiveTrackerNN:
    def __init__(self, model_path="tracking_model.pkl"):
        self.model_path = model_path
        
        # تهيئة مكونات NN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # نموذج استخراج الميزات
        self.feature_extractor = FeatureExtractor(feature_dim=256).to(self.device)
        self.feature_extractor.eval()
        
        # تحويلات الصور
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # المعالج
        self.detector = ObjectDetector()
        
        # هياكل البيانات للتتبع
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.track_colors = {}
        self.next_object_id = 0
        self.objects_db = {}
        
        # قاعدة بيانات الميزات
        self.feature_db = {}  # obj_id -> قائمة الميزات
        
        # وضع التعلم والتصحيحات
        self.corrections = []
        self.learning_mode = False
        self.selected_box = None
        self.correction_label = ""
        
        # واجهة المستخدم
        self.show_help = True
        self.gui_available = self.check_gui()
        
        # محرك التحسين
        self.optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.001)
        self.criterion = nn.TripletMarginLoss(margin=1.0)
        
        # تحميل النموذج إذا موجود
        self.load_model()
    
    def check_gui(self):
        try:
            test_window = cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("test")
            return True
        except:
            return False
    
    def load_model(self):
        """تحميل النموذج المدرب"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.objects_db = saved_data.get('objects_db', {})
                    self.next_object_id = saved_data.get('next_id', 0)
                    self.corrections = saved_data.get('corrections', [])
                    
                    # تحميل أوزان NN إذا كانت موجودة
                    nn_weights = saved_data.get('nn_weights')
                    if nn_weights:
                        self.feature_extractor.load_state_dict(nn_weights)
                    
                print(f"✓ Loaded model with {len(self.objects_db)} learned objects")
                return True
            except Exception as e:
                print(f"✗ Could not load model: {e}")
                return False
        return False
    
    def save_model(self):
        """حفظ النموذج المدرب"""
        try:
            data = {
                'objects_db': self.objects_db,
                'next_id': self.next_object_id,
                'corrections': self.corrections,
                'nn_weights': self.feature_extractor.state_dict(),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Model saved with {len(self.objects_db)} objects")
            return True
        except Exception as e:
            print(f"✗ Could not save model: {e}")
            return False
    
    def extract_features_nn(self, image):
        """استخراج الميزات باستخدام الشبكة العصبية"""
        if image is None or image.size == 0:
            return None
        
        try:
            # تحويل الصورة للتنسيق المناسب
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # تطبيق التحويلات
            tensor_img = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # استخراج الميزات
            with torch.no_grad():
                features = self.feature_extractor(tensor_img)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def detect_objects(self, frame):
        """اكتشاف الكائنات في الإطار"""
        if self.detector.yolo_available:
            detections = self.detector.detect_with_yolo(frame)
        else:
            detections = self.detector.detect_simple(frame)
        
        # استخراج الميزات لكل كشف
        for detection in detections:
            x, y, w, h = detection['box']
            roi = frame[y:y+h, x:x+w]
            features = self.extract_features_nn(roi)
            detection['feature'] = features
        
        return detections
    
    def match_with_existing(self, detection, max_distance=1.2):
        """مطابقة الكشف مع الكائنات الموجودة باستخدام المسافة الكوزينية"""
        if detection['feature'] is None:
            return None, float('inf')
        
        best_match = None
        best_distance = float('inf')
        
        for obj_id, obj_data in self.objects_db.items():
            if 'features' in obj_data and len(obj_data['features']) > 0:
                # حساب متوسط المسافة مع جميع الميزات المخزنة للكائن
                distances = []
                for stored_feature in obj_data['features']:
                    if stored_feature is not None:
                        # استخدام المسافة الكوزينية
                        distance = 1 - np.dot(detection['feature'], stored_feature) / (
                            np.linalg.norm(detection['feature']) * np.linalg.norm(stored_feature) + 1e-10
                        )
                        distances.append(distance)
                
                if distances:
                    avg_distance = np.mean(distances)
                    if avg_distance < best_distance and avg_distance < max_distance:
                        best_distance = avg_distance
                        best_match = obj_id
        
        return best_match, best_distance
    
    def update_tracking(self, detections):
        """تحديث التتبع"""
        current_objects = {}
        
        for detection in detections:
            obj_id, distance = self.match_with_existing(detection)
            
            if obj_id is not None:
                # تحديث كائن موجود
                current_objects[obj_id] = detection['box']
                self.track_history[obj_id].append(detection['box'])
                
                # تحديث الميزات (متوسط متحرك)
                if detection['feature'] is not None:
                    if 'features' not in self.objects_db[obj_id]:
                        self.objects_db[obj_id]['features'] = []
                    
                    self.objects_db[obj_id]['features'].append(detection['feature'])
                    
                    # الحفاظ على آخر 10 ميزات فقط
                    if len(self.objects_db[obj_id]['features']) > 10:
                        self.objects_db[obj_id]['features'].pop(0)
                
                # تحديث العداد
                self.objects_db[obj_id]['count'] = self.objects_db[obj_id].get('count', 0) + 1
                
            else:
                # كائن جديد
                obj_id = self.next_object_id
                self.next_object_id += 1
                
                # لون عشوائي للتتبع
                self.track_colors[obj_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255)
                )
                
                # تخزين بيانات الكائن
                self.objects_db[obj_id] = {
                    'label': detection.get('label', 'unknown'),
                    'features': [detection['feature']] if detection['feature'] is not None else [],
                    'count': 1,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now()
                }
                
                current_objects[obj_id] = detection['box']
                self.track_history[obj_id].append(detection['box'])
        
        return current_objects
    
    def draw_tracking(self, frame, current_objects):
        """رسم الصناديق والمعلومات على الإطار"""
        for obj_id, box in current_objects.items():
            x, y, w, h = box
            
            # الحصول على معلومات الكائن
            obj_info = self.objects_db.get(obj_id, {})
            label = obj_info.get('label', 'Unknown')
            count = obj_info.get('count', 0)
            color = self.track_colors.get(obj_id, (0, 255, 0))
            
            # رسم المستطيل
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # إعداد النص
            label_text = f"ID:{obj_id} {label}"
            if count > 0:
                label_text += f" ({count})"
            
            # خلفية للنص
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                frame,
                (x, max(0, y - text_height - 10)),
                (x + text_width, max(0, y)),
                color,
                -1
            )
            
            # كتابة النص
            cv2.putText(
                frame,
                label_text,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
            # رسم مسار التتبع
            history = list(self.track_history[obj_id])
            for i in range(1, len(history)):
                x1, y1, w1, h1 = history[i-1]
                x2, y2, w2, h2 = history[i]
                center1 = (x1 + w1//2, y1 + h1//2)
                center2 = (x2 + w2//2, y2 + h2//2)
                cv2.line(frame, center1, center2, color, 2)
        
        # رسم الصندوق المحدد للتصحيح
        if self.selected_box:
            x, y, w, h = self.selected_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            
            # نص التحديد
            select_text = f"Selected: {self.correction_label}" if self.correction_label else "Selected for correction"
            cv2.putText(
                frame,
                select_text,
                (x, max(0, y - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
        
        return frame
    
    def draw_ui_overlay(self, frame, fps):
        """رسم واجهة المستخدم"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # خلفية شبه شفافة
        cv2.rectangle(overlay, (0, 0), (min(450, width), 220), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # تعليمات
        instructions = [
            "NEURAL NETWORK OBJECT TRACKER",
            f"FPS: {fps:.1f} | Objects: {len(self.objects_db)} | Corrections: {len(self.corrections)}",
            "",
            "CONTROLS:",
            "• Drag: Select object",
            "• Type label + ENTER: Correct",
            "• h: Toggle help",
            "• s: Save model",
            "• r: Reset selection",
            "• c: Show objects",
            "• q: Quit",
            "",
            f"Device: {self.device}",
            f"Mode: {'LEARNING' if self.learning_mode else 'TRACKING'}"
        ]
        
        for i, text in enumerate(instructions):
            y_pos = 30 + i * 20
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(
                frame,
                text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return frame
    
    def apply_correction(self):
        """تطبيق تصحيح المستخدم"""
        if not self.selected_box or not self.correction_label:
            return
        
        try:
            x, y, w, h = self.selected_box
            
            # استخراج الميزات من المنطقة المحددة
            if hasattr(self, 'last_frame'):
                roi = self.last_frame[y:y+h, x:x+w]
                features = self.extract_features_nn(roi)
            else:
                features = None
            
            # البحث عن أقرب كائن موجود
            obj_id = None
            min_distance = float('inf')
            
            if features is not None:
                for existing_id, obj_data in self.objects_db.items():
                    if 'features' in obj_data and len(obj_data['features']) > 0:
                        distances = []
                        for stored_feature in obj_data['features']:
                            if stored_feature is not None:
                                distance = 1 - np.dot(features, stored_feature) / (
                                    np.linalg.norm(features) * np.linalg.norm(stored_feature) + 1e-10
                                )
                                distances.append(distance)
                        
                        if distances:
                            avg_distance = np.mean(distances)
                            if avg_distance < min_distance and avg_distance < 1.0:
                                min_distance = avg_distance
                                obj_id = existing_id
            
            # إذا لم يتم العثور على تطابق، إنشاء كائن جديد
            if obj_id is None:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.track_colors[obj_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255)
                )
            
            # تحديث أو إنشاء الكائن
            self.objects_db[obj_id] = {
                'label': self.correction_label,
                'features': [features] if features is not None else [],
                'count': self.objects_db.get(obj_id, {}).get('count', 0) + 1,
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'user_corrected': True
            }
            
            # حفظ التصحيح
            self.corrections.append({
                'timestamp': datetime.now(),
                'box': self.selected_box,
                'label': self.correction_label,
                'object_id': obj_id
            })
            
            print(f"\n✓ Correction applied: Object {obj_id} = '{self.correction_label}'")
            
            # إعادة تعيين
            self.selected_box = None
            self.correction_label = ""
            self.learning_mode = False
            
        except Exception as e:
            print(f"\n✗ Error applying correction: {e}")
            import traceback
            traceback.print_exc()
            self.selected_box = None
            self.correction_label = ""
            self.learning_mode = False
    
    def train_with_triplet_loss(self, anchor, positive, negative):
        """تدريب النموذج باستخدام Triplet Loss"""
        try:
            self.feature_extractor.train()
            
            # تحويل البيانات
            anchor = self.transform(anchor).unsqueeze(0).to(self.device)
            positive = self.transform(positive).unsqueeze(0).to(self.device)
            negative = self.transform(negative).unsqueeze(0).to(self.device)
            
            # تمرير عبر الشبكة
            anchor_features = self.feature_extractor(anchor)
            positive_features = self.feature_extractor(positive)
            negative_features = self.feature_extractor(negative)
            
            # حساب الخسارة
            loss = self.criterion(anchor_features, positive_features, negative_features)
            
            # تحديث الأوزان
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.feature_extractor.eval()
            
            return loss.item()
            
        except Exception as e:
            print(f"Training error: {e}")
            return None
    
    def process_with_gui(self, video_source=0):
        """المعالجة مع واجهة المستخدم الرسومية"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"✗ Cannot open video source {video_source}")
            return
        
        cv2.namedWindow("Neural Network Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Neural Network Tracker", 1024, 768)
        
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
        
        cv2.setMouseCallback("Neural Network Tracker", mouse_callback)
        
        print("\n" + "="*60)
        print("NEURAL NETWORK OBJECT TRACKER")
        print("="*60)
        print("\nDrag to select objects, type labels to teach the AI!")
        print("Starting tracking...")
        
        # إحصائيات الأداء
        fps_counter = 0
        fps_time = time.time()
        last_fps = 0
        processing_times = deque(maxlen=100)
        
        while True:
            start_process = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # حفظ الإطار الأخير للتصحيحات
            self.last_frame = frame.copy()
            
            # زيادة عداد FPS
            fps_counter += 1
            
            # حساب FPS
            current_time = time.time()
            if current_time - fps_time >= 1.0:
                last_fps = fps_counter
                fps_counter = 0
                fps_time = current_time
            
            # اكتشاف الكائنات
            detection_start = time.time()
            detections = self.detect_objects(frame)
            detection_time = time.time() - detection_start
            
            # تحديث التتبع
            tracking_start = time.time()
            current_objects = self.update_tracking(detections)
            tracking_time = time.time() - tracking_start
            
            # رسم التتبع
            drawing_start = time.time()
            frame = self.draw_tracking(frame, current_objects)
            drawing_time = time.time() - drawing_start
            
            # رسم واجهة المستخدم
            if self.show_help:
                frame = self.draw_ui_overlay(frame, last_fps)
            
            # عرض أوقات المعالجة
            total_time = time.time() - start_process
            processing_times.append(total_time)
            
            avg_process_time = np.mean(processing_times) if processing_times else 0
            cv2.putText(
                frame,
                f"Detect: {detection_time*1000:.1f}ms | Track: {tracking_time*1000:.1f}ms | Avg: {avg_process_time*1000:.1f}ms",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
            
            # عرض الإطار
            cv2.imshow("Neural Network Tracker", frame)
            
            # معالجة مدخلات لوحة المفاتيح
            key = cv2.waitKey(1) & 0xFF
            
            # وضع التعلم (إدخال التسمية)
            if self.learning_mode:
                if 32 <= key <= 126:  # حرف عادي
                    self.correction_label += chr(key)
                elif key == 13:  # Enter
                    if self.correction_label and self.selected_box:
                        self.apply_correction()
                elif key == 27:  # ESC
                    self.learning_mode = False
                    self.selected_box = None
                    self.correction_label = ""
                    print("\n✗ Correction cancelled")
            
            # أوامر التحكم
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
                    features_count = len(data.get('features', []))
                    print(f"  ID:{obj_id:3d} - {label:20s} (count: {count:3d}, features: {features_count})")
        
        # التنظيف
        cap.release()
        cv2.destroyAllWindows()
        self.save_model()
        print("\n✓ Tracker stopped. Model saved.")
    
    def process_video(self, video_source=0):
        """بدء المعالجة"""
        if self.gui_available:
            self.process_with_gui(video_source)
        else:
            print("GUI not available. Running in headless mode is not fully implemented.")
            # يمكن تنفيذ وضع بدون واجهة هنا
    
    def process_video_file(self, video_path):
        """معالجة ملف فيديو أو صورة"""
        if not os.path.exists(video_path):
            print(f"✗ Video file not found: {video_path}")
            return
        
        # إذا كان ملف صورة
        if video_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            frame = cv2.imread(video_path)
            if frame is not None:
                print(f"Processing image: {video_path}")
                
                # اكتشاف الكائنات
                detections = self.detect_objects(frame)
                current_objects = self.update_tracking(detections)
                frame = self.draw_tracking(frame, current_objects)
                
                # حفظ النتيجة
                output_path = f"result_{os.path.basename(video_path)}"
                cv2.imwrite(output_path, frame)
                print(f"✓ Result saved to: {output_path}")
                
                # عرض إذا كانت الواجهة متوفرة
                if self.gui_available:
                    cv2.imshow("Result", frame)
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
            else:
                print(f"✗ Could not read image: {video_path}")
        else:
            # معالجة ملف فيديو
            self.process_video(video_path)

# الدالة الرئيسية
def main():
    print("="*60)
    print("NEURAL NETWORK OBJECT TRACKER")
    print("="*60)
    print("\nThis system uses deep learning for object tracking!")
    print("You can teach it by correcting mislabeled objects.")
    
    # إنشاء التتبع
    tracker = InteractiveTrackerNN()
    
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
        # إنشاء صورة اختبارية
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.rectangle(test_image, (300, 150), (400, 250), (255, 0, 0), -1)
        cv2.circle(test_image, (500, 300), 50, (0, 0, 255), -1)
        
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
        print("Thank you for using the Neural Network Tracker!")
        print("="*60)