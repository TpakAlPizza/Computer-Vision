# مكتبات النظام والتعلم الآلي لتحليل أمراض النبات
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import requests
import zipfile
from pathlib import Path
import argparse
from datetime import datetime

# كاشف أمراض النبات
class PlantDiseaseDetector:
    def __init__(self, model_type='yolo'):
        self.model_type = model_type
        self.model = None
        self.class_names = []
        self.input_size = (416, 416)
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        
    def setup_environment(self):
        print("Setting up environment...")
        
        Path("models").mkdir(exist_ok=True)
        Path("test_images").mkdir(exist_ok=True)
        Path("output").mkdir(exist_ok=True)
        
        self.download_pretrained_model()
    
    def download_pretrained_model(self):
        print("Loading pre-trained model...")
        
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry___Powdery_mildew',
            'Cherry___healthy',
            'Corn___Cercospora_leaf_spot',
            'Corn___Common_rust',
            'Corn___Northern_Leaf_Blight',
            'Corn___healthy',
            'Grape___Black_rot',
            'Grape___Esca',
            'Grape___Leaf_blight',
            'Grape___healthy',
            'Orange___Haunglongbing',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper_bell___Bacterial_spot',
            'Pepper_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        
        if self.model_type == 'yolo':
            self.create_yolo_model()
        else:
            self.create_cnn_model()
    
    def create_yolo_model(self):
        print("Creating YOLO model architecture...")
        
        inputs = keras.Input(shape=(416, 416, 3))
        
        x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(4 + len(self.class_names), activation='sigmoid')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        model_path = "models/plant_disease_yolo.h5"
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                print("Loaded pre-trained weights")
            except:
                print("Could not load weights, using random initialization")
                self.initialize_model_weights()
        else:
            print("No pre-trained weights found. Using random initialization.")
            self.initialize_model_weights()
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['accuracy']
        )
    
    def initialize_model_weights(self):
        initializer = tf.keras.initializers.HeNormal()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = initializer
        self.model.build(self.model.input_shape)
    
    def create_cnn_model(self):
        print("Creating CNN model...")
        
        self.model = keras.Sequential([
            layers.Input(shape=(256, 256, 3)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model_path = "models/plant_disease_cnn.h5"
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                print("Loaded pre-trained weights")
            except:
                print("Could not load weights, using random initialization")
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_image(self, image):
        if self.model_type == 'yolo':
            img = cv2.resize(image, self.input_size)
        else:
            img = cv2.resize(image, (256, 256))
        
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def detect_diseases(self, image_path, output_path=None):
        print(f"\nProcessing image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Error: File does not exist: {image_path}")
            return None, []
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None, []
        
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(f"Error: Invalid image format for {image_path}")
            return None, []
        
        original_h, original_w = image_rgb.shape[:2]
        
        if self.model is None:
            print("Error: Model not initialized. Call setup_environment() first.")
            return None, []
        
        processed_img = self.preprocess_image(image_rgb)
        
        if self.model_type == 'yolo':
            predictions = self.model.predict(processed_img, verbose=0)[0]
            detections = self.parse_yolo_predictions(predictions, original_w, original_h)
        else:
            predictions = self.model.predict(processed_img, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = predictions[class_idx]
            
            detections = [{
                'bbox': [50, 50, original_w-100, original_h-100],
                'confidence': float(confidence),
                'class_id': int(class_idx),
                'class_name': self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown'
            }]
        
        result_image = self.draw_detections(image_rgb, detections)
        
        if output_path:
            try:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    Path(output_dir).mkdir(exist_ok=True)
                cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                print(f"Saved result to: {output_path}")
            except Exception as e:
                print(f"Error saving output: {e}")
        
        return result_image, detections
    
    def parse_yolo_predictions(self, predictions, img_w, img_h):
        detections = []
        
        if len(predictions) < 4 + len(self.class_names):
            return detections
        
        num_detections = 1
        
        for i in range(num_detections):
            start_idx = i * (4 + len(self.class_names))
            
            if start_idx + 4 + len(self.class_names) > len(predictions):
                break
            
            bbox_norm = predictions[start_idx:start_idx+4]
            class_probs = predictions[start_idx+4:start_idx+4+len(self.class_names)]
            class_idx = np.argmax(class_probs)
            confidence = class_probs[class_idx]
            
            if confidence > self.confidence_threshold and class_idx < len(self.class_names):
                x_center = np.clip(bbox_norm[0], 0, 1) * img_w
                y_center = np.clip(bbox_norm[1], 0, 1) * img_h
                width = np.clip(bbox_norm[2], 0, 1) * img_w * 0.5
                height = np.clip(bbox_norm[3], 0, 1) * img_h * 0.5
                
                x_min = max(0, int(x_center - width/2))
                y_min = max(0, int(y_center - height/2))
                width = min(img_w - x_min, int(width))
                height = min(img_h - y_min, int(height))
                
                if width > 0 and height > 0:
                    detections.append({
                        'bbox': [x_min, y_min, width, height],
                        'confidence': float(confidence),
                        'class_id': int(class_idx),
                        'class_name': self.class_names[class_idx]
                    })
        
        return detections
    
    def draw_detections(self, image, detections):
        result = image.copy()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        colors = (colors[:, :3] * 255).astype(int)
        
        for det in detections:
            bbox = det['bbox']
            class_name = det.get('class_name', 'unknown')
            confidence = det.get('confidence', 0.0)
            
            if len(bbox) != 4:
                continue
            
            x, y, w, h = bbox
            
            if x >= result.shape[1] or y >= result.shape[0] or w <= 0 or h <= 0:
                continue
            
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            
            color_idx = det.get('class_id', 0) % len(colors)
            color = tuple(map(int, colors[color_idx]))
            
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            try:
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(result, (x, max(0, y - label_size[1] - 5)), 
                            (x + label_size[0], y), color, -1)
                cv2.putText(result, label, (x, max(5, y - 5)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except:
                pass
            
            if confidence > 0.7 and 'healthy' not in class_name.lower():
                warning = "DISEASE DETECTED!"
                try:
                    cv2.putText(result, warning, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except:
                    pass
        
        return result
    
    def process_video(self, video_path, output_path=None):
        print(f"\nProcessing video: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"Error: File does not exist: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0:
            fps = 30
        
        out = None
        if output_path:
            try:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    Path(output_dir).mkdir(exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    print(f"Error: Could not create video writer for {output_path}")
                    out = None
            except Exception as e:
                print(f"Error setting up video writer: {e}")
                out = None
        
        frame_count = 0
        print(f"Video Info: {width}x{height}, {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 5 == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_img = self.preprocess_image(frame_rgb)
                    
                    if self.model_type == 'yolo':
                        predictions = self.model.predict(processed_img, verbose=0)[0]
                        detections = self.parse_yolo_predictions(predictions, width, height)
                    else:
                        predictions = self.model.predict(processed_img, verbose=0)[0]
                        class_idx = np.argmax(predictions)
                        confidence = predictions[class_idx]
                        
                        detections = [{
                            'bbox': [100, 100, width-200, height-200],
                            'confidence': float(confidence),
                            'class_id': int(class_idx),
                            'class_name': self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown'
                        }]
                    
                    result_frame = self.draw_detections(frame_rgb, detections)
                    result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    result_frame_bgr = frame
            else:
                result_frame_bgr = frame
            
            if out is not None:
                out.write(result_frame_bgr)
            
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if out is not None:
            out.release()
            print(f"Video saved to: {output_path}")
        
        return output_path if out is not None else None
    
    def visualize_results(self, image, detections):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Detection Results")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        if detections:
            classes = [d.get('class_name', 'unknown') for d in detections]
            confidences = [d.get('confidence', 0.0) for d in detections]
            
            y_pos = np.arange(len(classes))
            colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
            
            plt.barh(y_pos, confidences, color=colors)
            plt.yticks(y_pos, classes)
            plt.xlabel('Confidence')
            plt.title('Detected Diseases')
            plt.xlim(0, 1)
            
            for i, conf in enumerate(confidences):
                plt.text(conf + 0.02, i, f'{conf:.2f}', va='center')
        else:
            plt.text(0.5, 0.5, 'No diseases detected', 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plant Disease Detection System')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input image or video')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output (optional)')
    parser.add_argument('--model', type=str, default='yolo',
                       choices=['yolo', 'cnn'],
                       help='Model type to use')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization of results')
    
    args = parser.parse_args()
    
    if not args.input:
        print("Error: --input argument is required")
        return
    
    print("=== Plant Disease Detection System ===")
    detector = PlantDiseaseDetector(model_type=args.model)
    detector.setup_environment()
    
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        return
    
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        result_image, detections = detector.detect_diseases(
            input_path, 
            args.output
        )
        
        if result_image is not None:
            print(f"\nDetected {len(detections)} disease regions:")
            for det in detections:
                print(f"  - {det.get('class_name', 'unknown')}: {det.get('confidence', 0):.2%} confidence")
            
            if args.visualize:
                detector.visualize_results(result_image, detections)
    
    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        output_video = args.output or 'output/detected_video.mp4'
        result = detector.process_video(input_path, output_video)
        if result:
            print(f"\nVideo processing complete. Saved to: {result}")
    
    else:
        print(f"Error: Unsupported file format: {file_ext}")
        print("Supported formats: .jpg, .png, .mp4, .avi, .mov, .mkv")
    
    print("\n=== Processing Complete ===")

if __name__ == "__main__":
    main()
