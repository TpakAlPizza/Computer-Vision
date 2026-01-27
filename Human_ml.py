#!/usr/bin/env python3
"""
Human Pose Estimation - Real-time Camera Application
Advanced AI Computer Vision Project

A complete, single-file application for real-time human pose estimation
using webcam feed with a pre-trained model for optimal performance.
"""

import cv2
import torch
import torchvision
import numpy as np
import argparse
import time
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class RealTimePoseEstimator:
    """
    Real-time Human Pose Estimator using webcam feed
    """
    
    def __init__(self, confidence_threshold=0.8, keypoint_threshold=0.5):
        """
        Initialize the pose estimator
        
        Args:
            confidence_threshold: Minimum confidence for person detection
            keypoint_threshold: Minimum confidence for keypoint detection
        """
        self.confidence_threshold = confidence_threshold
        self.keypoint_threshold = keypoint_threshold
        
        # Keypoint names (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections for visualization
        self.skeleton_connections = [
            # Torso
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            # Left arm
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            # Right arm
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            # Left leg
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            # Right leg
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            # Face
            ('left_eye', 'right_eye'),
            ('nose', 'left_eye'),
            ('nose', 'right_eye'),
            ('left_eye', 'left_ear'),
            ('right_eye', 'right_ear')
        ]
        
        # Colors for visualization
        self.colors = {
            'bbox': (0, 255, 0),  # Green
            'keypoint': (0, 0, 255),  # Red
            'skeleton': (255, 0, 0),  # Blue
            'text': (255, 255, 255)  # White
        }
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Load pre-trained model
        self._load_model()
        
        # Initialize webcam
        self.cap = None
        
    def _load_model(self):
        """Load the pre-trained Keypoint RCNN model"""
        print("Loading pre-trained Keypoint RCNN model...")
        self.model = keypointrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        print("✓ Model loaded successfully!")
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"✓ Model running on: {self.device}")
        
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for the model
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        tensor = F.to_tensor(rgb_frame)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def draw_skeleton(self, frame, keypoints):
        """
        Draw skeleton connections on the frame
        
        Args:
            frame: OpenCV frame
            keypoints: Array of keypoints with shape (17, 3) [x, y, confidence]
        """
        # Draw connections
        for connection in self.skeleton_connections:
            start_name, end_name = connection
            start_idx = self.keypoint_names.index(start_name)
            end_idx = self.keypoint_names.index(end_name)
            
            start_conf = keypoints[start_idx, 2]
            end_conf = keypoints[end_idx, 2]
            
            # Only draw if both keypoints are confident
            if start_conf > self.keypoint_threshold and end_conf > self.keypoint_threshold:
                start_x, start_y = int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1])
                end_x, end_y = int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1])
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                        self.colors['skeleton'], 2)
    
    def draw_keypoints(self, frame, keypoints):
        """
        Draw keypoints on the frame
        
        Args:
            frame: OpenCV frame
            keypoints: Array of keypoints with shape (17, 3) [x, y, confidence]
        """
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.keypoint_threshold:
                x_int, y_int = int(x), int(y)
                
                # Draw keypoint
                cv2.circle(frame, (x_int, y_int), 5, self.colors['keypoint'], -1)
                
                # Optional: Draw keypoint index
                # cv2.putText(frame, str(i), (x_int + 5, y_int - 5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def draw_bbox(self, frame, box, label="Person", confidence=1.0):
        """
        Draw bounding box on the frame
        
        Args:
            frame: OpenCV frame
            box: Bounding box coordinates [x1, y1, x2, y2]
            label: Object label
            confidence: Detection confidence
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['bbox'], 2)
        
        # Draw label background
        label_text = f"{label}: {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), self.colors['bbox'], -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
    
    def calculate_angles(self, keypoints):
        """
        Calculate joint angles for pose analysis
        
        Args:
            keypoints: Array of keypoints with shape (17, 3)
            
        Returns:
            Dictionary of joint angles
        """
        angles = {}
        
        # Helper function to calculate angle between three points
        def angle_between(p1, p2, p3):
            if p1[2] < self.keypoint_threshold or p2[2] < self.keypoint_threshold or p3[2] < self.keypoint_threshold:
                return None
            
            # Convert to vectors
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Calculate angle
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm_product == 0:
                return None
            
            cosine = dot_product / norm_product
            cosine = np.clip(cosine, -1, 1)  # Ensure valid range
            angle = np.degrees(np.arccos(cosine))
            
            return angle
        
        # Calculate elbow angles
        left_elbow_angle = angle_between(
            keypoints[self.keypoint_names.index('left_shoulder')],
            keypoints[self.keypoint_names.index('left_elbow')],
            keypoints[self.keypoint_names.index('left_wrist')]
        )
        
        right_elbow_angle = angle_between(
            keypoints[self.keypoint_names.index('right_shoulder')],
            keypoints[self.keypoint_names.index('right_elbow')],
            keypoints[self.keypoint_names.index('right_wrist')]
        )
        
        # Calculate knee angles
        left_knee_angle = angle_between(
            keypoints[self.keypoint_names.index('left_hip')],
            keypoints[self.keypoint_names.index('left_knee')],
            keypoints[self.keypoint_names.index('left_ankle')]
        )
        
        right_knee_angle = angle_between(
            keypoints[self.keypoint_names.index('right_hip')],
            keypoints[self.keypoint_names.index('right_knee')],
            keypoints[self.keypoint_names.index('right_ankle')]
        )
        
        if left_elbow_angle is not None:
            angles['left_elbow'] = left_elbow_angle
        if right_elbow_angle is not None:
            angles['right_elbow'] = right_elbow_angle
        if left_knee_angle is not None:
            angles['left_knee'] = left_knee_angle
        if right_knee_angle is not None:
            angles['right_knee'] = right_knee_angle
            
        return angles
    
    def draw_angles(self, frame, keypoints, angles):
        """
        Draw joint angles on the frame
        
        Args:
            frame: OpenCV frame
            keypoints: Array of keypoints
            angles: Dictionary of joint angles
        """
        # Map angles to keypoint positions for drawing
        angle_positions = {
            'left_elbow': self.keypoint_names.index('left_elbow'),
            'right_elbow': self.keypoint_names.index('right_elbow'),
            'left_knee': self.keypoint_names.index('left_knee'),
            'right_knee': self.keypoint_names.index('right_knee')
        }
        
        for joint, angle in angles.items():
            if joint in angle_positions:
                idx = angle_positions[joint]
                if keypoints[idx, 2] > self.keypoint_threshold:
                    x, y = int(keypoints[idx, 0]), int(keypoints[idx, 1])
                    
                    # Draw angle text
                    angle_text = f"{int(angle)}°"
                    cv2.putText(frame, angle_text, (x + 10, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def process_frame(self, frame):
        """
        Process a single frame for pose estimation
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            Processed frame with pose estimation visualization
        """
        # Preprocess frame
        input_tensor = self.preprocess_frame(frame)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Move predictions to CPU if needed
        predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]
        
        # Process each detected person
        boxes = predictions[0]['boxes'].numpy()
        keypoints = predictions[0]['keypoints'].numpy()
        scores = predictions[0]['scores'].numpy()
        
        person_count = 0
        
        for i, (box, keypoint_set, score) in enumerate(zip(boxes, keypoints, scores)):
            if score < self.confidence_threshold:
                continue
                
            person_count += 1
            
            # Draw bounding box
            self.draw_bbox(frame, box, confidence=score)
            
            # Draw skeleton
            self.draw_skeleton(frame, keypoint_set)
            
            # Draw keypoints
            self.draw_keypoints(frame, keypoint_set)
            
            # Calculate and draw angles
            angles = self.calculate_angles(keypoint_set)
            self.draw_angles(frame, keypoint_set, angles)
        
        return frame, person_count
    
    def draw_performance_stats(self, frame):
        """
        Draw performance statistics on the frame
        
        Args:
            frame: OpenCV frame
            
        Returns:
            Frame with statistics overlay
        """
        # Calculate FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1:  # Update FPS every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # Draw stats overlay
        stats_y = 30
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Confidence Threshold: {self.confidence_threshold}",
            f"Device: {self.device}",
            "Press 'q' to quit",
            "Press '+' to increase confidence",
            "Press '-' to decrease confidence"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (10, stats_y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return frame
    
    def run(self, camera_id=0, window_name="Human Pose Estimation"):
        """
        Run real-time pose estimation from webcam
        
        Args:
            camera_id: Camera device ID
            window_name: Name of the display window
        """
        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print(f"\nStarting real-time pose estimation...")
        print("Press 'q' to quit")
        print("Press '+' to increase confidence threshold")
        print("Press '-' to decrease confidence threshold")
        print("Press 'r' to reset to default threshold")
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Process frame
                processed_frame, person_count = self.process_frame(frame)
                
                # Add person count
                cv2.putText(processed_frame, f"Persons: {person_count}", 
                           (frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Add performance stats
                processed_frame = self.draw_performance_stats(processed_frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('+'):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"Confidence threshold increased to {self.confidence_threshold:.2f}")
                elif key == ord('-'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"Confidence threshold decreased to {self.confidence_threshold:.2f}")
                elif key == ord('r'):
                    self.confidence_threshold = 0.8
                    print(f"Confidence threshold reset to {self.confidence_threshold:.2f}")
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"pose_estimation_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved as {filename}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\nApplication closed")

def main():
    """Main function to run the pose estimator"""
    parser = argparse.ArgumentParser(description='Real-time Human Pose Estimation')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.8,
                       help='Confidence threshold for detection (default: 0.8)')
    parser.add_argument('--keypoint-threshold', type=float, default=0.5,
                       help='Confidence threshold for keypoints (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create and run pose estimator
    estimator = RealTimePoseEstimator(
        confidence_threshold=args.confidence,
        keypoint_threshold=args.keypoint_threshold
    )
    
    estimator.run(camera_id=args.camera)

if __name__ == "__main__":
    main()