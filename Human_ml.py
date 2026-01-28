# استيراد المكتبات الأساسية
import cv2
import numpy as np
import argparse
import time
import os
from collections import deque

# محاولة استيراد YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

class AccuratePoseTracker:
    def __init__(self, model_size='n'):
        """
        تهيئة متتبع الوضعية باستخدام YOLOv8
        
        Args:
            model_size: حجم النموذج:
                'n' = nano (الأصغر والأسرع - 6MB)
                's' = small (صغير - 22MB)
                'm' = medium (متوسط - 50MB)
                'l' = large (كبير - 76MB)
        """
        # تحويل حجم النموذج إلى التسمية الصحيحة
        model_sizes = {
            'n': 'nano',
            's': 'small', 
            'm': 'medium',
            'l': 'large'
        }
        
        model_name = f'yolov8{model_size}-pose.pt'
        print(f"Loading {model_sizes.get(model_size, model_size)} model: {model_name}")
        print("This may download the model on first run (6-76 MB depending on size)...")
        
        # تحميل النموذج (سيقوم بتنزيله تلقائياً إذا لم يكن موجوداً)
        self.model = YOLO(model_name)
        
        # مفاصل الجسم في YOLOv8 (17 نقطة)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # روابط العظام
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # الرأس
            (5, 6), (5, 7), (7, 9),  # الذراع الأيسر
            (6, 8), (8, 10),  # الذراع الأيمن
            (5, 11), (6, 12),  # الجسم العلوي
            (11, 13), (13, 15),  # الرجل اليسرى
            (12, 14), (14, 16),  # الرجل اليمنى
            (11, 12)  # الحوض
        ]
        
        # ألوان للعظام
        self.skeleton_colors = [
            (255, 100, 100), (255, 150, 100), (255, 200, 100), (255, 255, 100),
            (100, 255, 100), (100, 255, 150), (100, 255, 200),
            (100, 200, 255), (100, 150, 255),
            (100, 100, 255), (150, 100, 255),
            (200, 100, 255), (255, 100, 255),
            (255, 100, 200), (255, 100, 150),
            (200, 200, 200)
        ]
        
        # ألوان للمفاصل
        self.joint_colors = [
            (255, 0, 0),     # أحمر للأنف
            (0, 255, 0),     # أخضر للعيون
            (0, 255, 0),
            (0, 0, 255),     # أزرق للأذنين
            (0, 0, 255),
            (255, 255, 0),   # سماوي للأكتاف
            (255, 255, 0),
            (255, 0, 255),   # بنفسجي للمرفقين
            (255, 0, 255),
            (0, 255, 255),   # أصفر للمعصمين
            (0, 255, 255),
            (128, 0, 128),   # بنفسجي غامق للوركين
            (128, 0, 128),
            (255, 165, 0),   # برتقالي للركبتين
            (255, 165, 0),
            (0, 128, 128),   # تركواز للكاحلين
            (0, 128, 128)
        ]
        
        # إعدادات
        self.conf_threshold = 0.5
        self.joint_size = 8
        self.bone_thickness = 3
        
        # لتنعيم الحركة
        self.pose_history = deque(maxlen=3)
        
        # إحصائيات
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.inference_time = 0
        
        print(f"Model loaded successfully!")
    
    def get_keypoints_from_results(self, results):
        """استخراج النقاط من نتائج YOLO"""
        if results[0].keypoints is None:
            return []
        
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        if len(keypoints_data) == 0:
            return []
        
        # اختيار الشخص مع معظم النقاط المرئية
        best_idx = 0
        max_visible = 0
        
        for i, person_kpts in enumerate(keypoints_data):
            visible = np.sum(person_kpts[:, 2] > 0.3)
            if visible > max_visible:
                max_visible = visible
                best_idx = i
        
        return keypoints_data[best_idx]
    
    def smooth_keypoints(self, keypoints):
        """تنعيم حركة النقاط"""
        if len(self.pose_history) == 0:
            self.pose_history.append(keypoints)
            return keypoints
        
        smoothed = keypoints.copy()
        
        # إذا كان لدينا تاريخ، ننعم النقاط
        if len(self.pose_history) > 0:
            for i in range(len(keypoints)):
                if keypoints[i, 2] > 0.3:  # إذا كانت النقطة مرئية
                    # جمع القيم التاريخية
                    points = [keypoints[i, :2]]
                    for hist in self.pose_history:
                        if hist[i, 2] > 0.3:
                            points.append(hist[i, :2])
                    
                    if len(points) > 1:
                        # حساب المتوسط
                        avg_x = np.mean([p[0] for p in points])
                        avg_y = np.mean([p[1] for p in points])
                        smoothed[i, 0] = avg_x
                        smoothed[i, 1] = avg_y
        
        self.pose_history.append(smoothed)
        return smoothed
    
    def calculate_angles(self, keypoints):
        """حساب زوايا المفاصل"""
        angles = {}
        
        def get_point(idx):
            if keypoints[idx, 2] > 0.3:  # ثقة كافية
                return np.array([keypoints[idx, 0], keypoints[idx, 1]])
            return None
        
        def calculate_angle(a, b, c):
            if a is None or b is None or c is None:
                return None
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            return angle
        
        # زوايا الذراع الأيسر
        left_shoulder = get_point(5)
        left_elbow = get_point(7)
        left_wrist = get_point(9)
        
        if all(p is not None for p in [left_shoulder, left_elbow, left_wrist]):
            angles['left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # زوايا الذراع الأيمن
        right_shoulder = get_point(6)
        right_elbow = get_point(8)
        right_wrist = get_point(10)
        
        if all(p is not None for p in [right_shoulder, right_elbow, right_wrist]):
            angles['right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # زوايا الرجل اليسرى
        left_hip = get_point(11)
        left_knee = get_point(13)
        left_ankle = get_point(15)
        
        if all(p is not None for p in [left_hip, left_knee, left_ankle]):
            angles['left_knee'] = calculate_angle(left_hip, left_knee, left_ankle)
        
        # زوايا الرجل اليمنى
        right_hip = get_point(12)
        right_knee = get_point(14)
        right_ankle = get_point(16)
        
        if all(p is not None for p in [right_hip, right_knee, right_ankle]):
            angles['right_knee'] = calculate_angle(right_hip, right_knee, right_ankle)
        
        return angles
    
    def draw_skeleton(self, frame, keypoints):
        """رسم الهيكل العظمي على الإطار"""
        h, w = frame.shape[:2]
        
        # رسم العظام
        for i, (start_idx, end_idx) in enumerate(self.skeleton):
            if (keypoints[start_idx, 2] > 0.3 and keypoints[end_idx, 2] > 0.3):
                start_pt = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                end_pt = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                
                # التأكد من أن النقاط داخل الإطار
                if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
                    0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
                    
                    color = self.skeleton_colors[i % len(self.skeleton_colors)]
                    cv2.line(frame, start_pt, end_pt, color, self.bone_thickness)
        
        # رسم المفاصل
        for i in range(len(keypoints)):
            if keypoints[i, 2] > 0.3:  # إذا كانت النقطة مرئية
                x = int(keypoints[i, 0])
                y = int(keypoints[i, 1])
                conf = keypoints[i, 2]
                
                # التأكد من أن النقطة داخل الإطار
                if 0 <= x < w and 0 <= y < h:
                    size = int(self.joint_size * (0.5 + conf * 0.5))
                    color = self.joint_colors[i % len(self.joint_colors)]
                    
                    # رسم دائرة سوداء خلفية
                    cv2.circle(frame, (x, y), size + 2, (0, 0, 0), -1)
                    # رسم المفصل
                    cv2.circle(frame, (x, y), size, color, -1)
                    # نقطة مركزية بيضاء
                    cv2.circle(frame, (x, y), max(2, size // 3), (255, 255, 255), -1)
    
    def draw_info(self, frame, angles, inference_time, num_persons):
        """رسم المعلومات على الشاشة"""
        # حساب FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # رسم الإحصائيات
        stats_y = 30
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Persons: {num_persons}",
            f"Inference: {inference_time:.0f}ms",
            f"Confidence: {self.conf_threshold:.2f}",
            "Q: Quit  +/-: Adjust confidence",
            "S: Save screenshot"
        ]
        
        for i, text in enumerate(stats):
            color = (255, 255, 255)
            if i == 0:
                if self.fps < 20:
                    color = (0, 165, 255)  # برتقالي إذا كان FPS منخفضاً
                else:
                    color = (0, 255, 0)  # أخضر إذا كان FPS جيداً
            
            cv2.putText(frame, text, (10, stats_y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # رسم الزوايا
        if angles:
            angles_y = 180
            cv2.putText(frame, "Joint Angles:", (10, angles_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            angles_y += 25
            
            for angle_name, angle_value in angles.items():
                if angle_value is not None:
                    text = f"{angle_name}: {int(angle_value)}°"
                    cv2.putText(frame, text, (10, angles_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    angles_y += 25
    
    def process_frame(self, frame):
        """معالجة إطار واحد"""
        start_time = time.time()
        
        # تشغيل النموذج
        results = self.model(frame, 
                           conf=self.conf_threshold,
                           iou=0.45,
                           verbose=False,
                           max_det=1)  # اكتشاف شخص واحد فقط لتحسين الأداء
        
        self.inference_time = (time.time() - start_time) * 1000
        
        output_frame = frame.copy()
        num_persons = 0
        
        if results[0].keypoints is not None:
            keypoints = self.get_keypoints_from_results(results)
            
            if len(keypoints) > 0:
                num_persons = 1
                
                # تنعيم حركة النقاط
                smoothed_kpts = self.smooth_keypoints(keypoints)
                
                # حساب الزوايا
                angles = self.calculate_angles(smoothed_kpts)
                
                # رسم الهيكل العظمي
                self.draw_skeleton(output_frame, smoothed_kpts)
                
                # رسم مربع حول الشخص
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    box = results[0].boxes.data[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return output_frame, num_persons
    
    def run(self, camera_id=0):
        """تشغيل المتتبع"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
        
        # إعدادات الكاميرا لتحسين الأداء
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*50)
        print("Accurate Pose Tracker Started!")
        print("="*50)
        print("Controls:")
        print("  Q - Quit")
        print("  + - Increase confidence threshold")
        print("  - - Decrease confidence threshold")
        print("  S - Save screenshot")
        print("="*50)
        
        angles = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            # قلب الإطار
            frame = cv2.flip(frame, 1)
            
            # معالجة الإطار
            processed_frame, num_persons = self.process_frame(frame)
            
            # حساب الزوايا
            if num_persons > 0:
                results = self.model(frame, conf=self.conf_threshold, verbose=False, max_det=1)
                keypoints = self.get_keypoints_from_results(results)
                if len(keypoints) > 0:
                    angles = self.calculate_angles(keypoints)
            
            # رسم المعلومات
            self.draw_info(processed_frame, angles, self.inference_time, num_persons)
            
            # عرض الإطار
            cv2.imshow('Accurate Pose Tracker', processed_frame)
            
            # معالجة الأوامر
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('+'):
                self.conf_threshold = min(0.9, self.conf_threshold + 0.05)
                print(f"Confidence threshold: {self.conf_threshold:.2f}")
            elif key == ord('-'):
                self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                print(f"Confidence threshold: {self.conf_threshold:.2f}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pose_tracker_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nTracker stopped.")

def main():
    parser = argparse.ArgumentParser(description='Accurate Pose Tracker')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l'],
                       help='Model size: n (nano - fastest), s (small), m (medium), l (large - most accurate)')
    
    args = parser.parse_args()
    
    tracker = AccuratePoseTracker(model_size=args.model)
    tracker.run(camera_id=args.camera)

if __name__ == "__main__":
    main()