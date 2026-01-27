# مكتبات استيراد
import cv2
import numpy as np
import argparse
import time
import mediapipe as mp

# فئة تقدير الوضعية الخفيفة
class LitePoseEstimator:
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.keypoint_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
            'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
            'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
            'right_index', 'left_thumb', 'right_thumb', 'left_hip',
            'right_hip', 'left_knee', 'right_knee', 'left_ankle',
            'right_ankle', 'left_heel', 'right_heel', 'left_foot_index',
            'right_foot_index'
        ]
        
        self.skeleton_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (23, 25), (25, 27), (27, 29), (29, 31),
            (12, 24), (24, 26), (26, 28), (28, 30), (30, 32),
            (23, 24), (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8)
        ]
        
        self.colors = {
            'skeleton': (255, 0, 0),
            'landmark': (0, 255, 0),
            'text': (255, 255, 255),
            'highlight': (0, 255, 255)
        }
        
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        self.cap = None
        
    def draw_custom_skeleton(self, frame, landmarks):
        height, width, _ = frame.shape
        
        for connection in self.skeleton_connections:
            start_idx, end_idx = connection
            
            if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
                start_point = (int(landmarks[start_idx].x * width), 
                              int(landmarks[start_idx].y * height))
                end_point = (int(landmarks[end_idx].x * width), 
                            int(landmarks[end_idx].y * height))
                
                cv2.line(frame, start_point, end_point, 
                        self.colors['skeleton'], 2)
    
    def draw_custom_landmarks(self, frame, landmarks):
        height, width, _ = frame.shape
        
        for idx, landmark in enumerate(landmarks):
            if landmark.visibility > 0.5:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                cv2.circle(frame, (x, y), 4, self.colors['landmark'], -1)
                
                if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    cv2.circle(frame, (x, y), 6, self.colors['highlight'], 2)
    
    def calculate_angles(self, landmarks, frame_shape):
        angles = {}
        height, width = frame_shape[:2]
        
        def get_point(idx):
            if landmarks[idx].visibility > 0.5:
                return np.array([landmarks[idx].x * width, landmarks[idx].y * height])
            return None
        
        def calculate_angle(p1, p2, p3):
            if p1 is None or p2 is None or p3 is None:
                return None
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm_product == 0:
                return None
            
            cosine = np.clip(dot_product / norm_product, -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine))
            
            return angle
        
        left_shoulder = get_point(11)
        left_elbow = get_point(13)
        left_wrist = get_point(15)
        
        right_shoulder = get_point(12)
        right_elbow = get_point(14)
        right_wrist = get_point(16)
        
        left_hip = get_point(23)
        left_knee = get_point(25)
        left_ankle = get_point(27)
        
        right_hip = get_point(24)
        right_knee = get_point(26)
        right_ankle = get_point(28)
        
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        if left_elbow_angle is not None:
            angles['left_elbow'] = left_elbow_angle
        if right_elbow_angle is not None:
            angles['right_elbow'] = right_elbow_angle
        if left_knee_angle is not None:
            angles['left_knee'] = left_knee_angle
        if right_knee_angle is not None:
            angles['right_knee'] = right_knee_angle
            
        return angles
    
    def draw_angles(self, frame, landmarks, angles):
        height, width, _ = frame.shape
        
        angle_positions = {
            'left_elbow': 13,
            'right_elbow': 14,
            'left_knee': 25,
            'right_knee': 26
        }
        
        for joint, angle in angles.items():
            if joint in angle_positions:
                idx = angle_positions[joint]
                if landmarks[idx].visibility > 0.5:
                    x = int(landmarks[idx].x * width)
                    y = int(landmarks[idx].y * height)
                    
                    angle_text = f"{int(angle)}°"
                    cv2.putText(frame, angle_text, (x + 10, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def draw_posture_feedback(self, frame, angles):
        feedback = []
        y_offset = 60
        
        if 'left_elbow' in angles:
            if angles['left_elbow'] < 60:
                feedback.append("Bend left arm more")
            elif angles['left_elbow'] > 160:
                feedback.append("Straighten left arm")
                
        if 'right_elbow' in angles:
            if angles['right_elbow'] < 60:
                feedback.append("Bend right arm more")
            elif angles['right_elbow'] > 160:
                feedback.append("Straighten right arm")
                
        for i, text in enumerate(feedback):
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.pose.process(rgb_frame)
        
        rgb_frame.flags.writeable = True
        
        person_count = 0
        
        if results.pose_landmarks:
            person_count = 1
            
            self.draw_custom_skeleton(frame, results.pose_landmarks.landmark)
            self.draw_custom_landmarks(frame, results.pose_landmarks.landmark)
            
            angles = self.calculate_angles(results.pose_landmarks.landmark, frame.shape)
            self.draw_angles(frame, results.pose_landmarks.landmark, angles)
            self.draw_posture_feedback(frame, angles)
            
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        return frame, person_count
    
    def draw_stats(self, frame):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Confidence: {self.min_detection_confidence:.1f}",
            "Press 'q' to quit",
            "Press '+'/- to adjust confidence",
            "Press '1/2/3' to change model"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return frame
    
    def run(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"Cannot open camera {camera_id}")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nStarting Lite Pose Estimation")
        print("Controls:")
        print("  q - Quit")
        print("  + - Increase confidence")
        print("  - - Decrease confidence")
        print("  1 - Light model (fastest)")
        print("  2 - Balanced model")
        print("  3 - Heavy model (most accurate)")
        print("  s - Save screenshot")
        
        model_complexity = 1
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            processed_frame, person_count = self.process_frame(frame)
            
            cv2.putText(processed_frame, f"Persons: {person_count}", 
                       (processed_frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            self.draw_stats(processed_frame)
            
            cv2.imshow('Lite Pose Estimation', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('+'):
                self.min_detection_confidence = min(0.9, self.min_detection_confidence + 0.1)
                self.min_tracking_confidence = min(0.9, self.min_tracking_confidence + 0.1)
                self.update_model(model_complexity)
                print(f"Confidence: {self.min_detection_confidence:.1f}")
            elif key == ord('-'):
                self.min_detection_confidence = max(0.1, self.min_detection_confidence - 0.1)
                self.min_tracking_confidence = max(0.1, self.min_tracking_confidence - 0.1)
                self.update_model(model_complexity)
                print(f"Confidence: {self.min_detection_confidence:.1f}")
            elif key == ord('1'):
                model_complexity = 0
                self.update_model(0)
                print("Using light model")
            elif key == ord('2'):
                model_complexity = 1
                self.update_model(1)
                print("Using balanced model")
            elif key == ord('3'):
                model_complexity = 2
                self.update_model(2)
                print("Using heavy model")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pose_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Saved: {filename}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
    
    def update_model(self, complexity):
        self.pose.close()
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

# فئة الوظيفة الرئيسية
def main():
    parser = argparse.ArgumentParser(description='Lite Human Pose Estimation')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence (default: 0.5)')
    
    args = parser.parse_args()
    
    estimator = LitePoseEstimator(
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence
    )
    
    estimator.run(camera_id=args.camera)

# فئة نقطة بدء البرنامج
if __name__ == "__main__":
    main()
