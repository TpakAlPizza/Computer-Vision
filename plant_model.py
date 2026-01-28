# مكتبات النظام والتعلم الآلي لتحليل أمراض النبات
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import sys
import argparse
from pathlib import Path
import warnings
from typing import List, Tuple, Optional, Dict, Any
import json
from datetime import datetime
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# كاشف أمراض النبات
class PlantDiseaseDetector:
    def __init__(self, model_type: str = 'cnn'):
        """
        مُهيأ لكاشف أمراض النبات
        
        Args:
            model_type: نوع النموذج ('cnn' أو 'custom')
        """
        self.model_type = model_type
        self.model = None
        self.class_names = []
        self.input_size = (224, 224)  # حجم قياسي لتصنيف الصور
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        
        # تعريف فئات الأمراض النباتية
        self._initialize_class_names()
        
    def _initialize_class_names(self):
        """تهيئة أسماء فئات الأمراض النباتية"""
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
        
        # فهرسة سريعة للفئات الصحية
        self.healthy_indices = [i for i, name in enumerate(self.class_names) 
                              if 'healthy' in name.lower()]
    
    def setup_environment(self) -> bool:
        """إعداد البيئة وإنشاء المجلدات الضرورية"""
        try:
            logger.info("إعداد البيئة...")
            
            # إنشاء المجلدات الضرورية
            directories = ["models", "test_images", "output", "logs", "temp"]
            for dir_name in directories:
                Path(dir_name).mkdir(exist_ok=True)
                logger.debug(f"تم إنشاء المجلد: {dir_name}")
            
            # تحميل النموذج
            if not self._load_or_create_model():
                logger.error("فشل في تحميل أو إنشاء النموذج")
                return False
            
            logger.info("تم إعداد البيئة بنجاح")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إعداد البيئة: {e}")
            return False
    
    def _load_or_create_model(self) -> bool:
        """تحميل نموذج موجود أو إنشاء نموذج جديد"""
        model_filename = f"models/plant_disease_{self.model_type}.h5"
        
        try:
            if Path(model_filename).exists():
                logger.info(f"تحميل النموذج من {model_filename}")
                if self.model_type == 'cnn':
                    self._create_cnn_model()
                    self.model.load_weights(model_filename)
                else:
                    self._create_custom_model()
                    self.model.load_weights(model_filename)
                logger.info("تم تحميل النموذج بنجاح")
            else:
                logger.info("إنشاء نموذج جديد...")
                if self.model_type == 'cnn':
                    self._create_cnn_model()
                else:
                    self._create_custom_model()
                logger.info("تم إنشاء نموذج جديد")
            
            return True
            
        except Exception as e:
            logger.error(f"خطأ في تحميل/إنشاء النموذج: {e}")
            return False
    
    def _create_cnn_model(self):
        """إنشاء نموذج CNN لتصنيف الأمراض"""
        logger.info("إنشاء نموذج CNN...")
        
        # استخدام بنية أساسية معمارية (يمكن استبدالها بـ ResNet، EfficientNet، إلخ)
        inputs = layers.Input(shape=(*self.input_size, 3))
        
        # نموذج CNN مبسط
        x = layers.Rescaling(1./255)(inputs)
        
        # كتل الالتفاف
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        
        # طبقات مكثفة
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # طبقة الإخراج
        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)
        
        # تجميع النموذج
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # تجميع النموذج
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info("تم إنشاء نموذج CNN")
    
    def _create_custom_model(self):
        """إنشاء نموذج مخصص مع كشف المنطقة (ليس YOLO حقيقي)"""
        logger.info("إنشاء نموذج مخصص مع كشف المنطقة...")
        
        # هذا ليس YOLO حقيقي، لكنه نموذج يدمج التصنيف مع تقدير المنطقة
        inputs = layers.Input(shape=(256, 256, 3))
        
        # معالجة مسبقة
        x = layers.Rescaling(1./255)(inputs)
        
        # شبكة أساسية
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # فرع التصنيف
        classification_branch = layers.GlobalAveragePooling2D()(x)
        classification_branch = layers.Dense(128, activation='relu')(classification_branch)
        classification_branch = layers.Dropout(0.5)(classification_branch)
        classification_output = layers.Dense(len(self.class_names), 
                                           activation='softmax', 
                                           name='classification')(classification_branch)
        
        # فرع الصندوق المحيط (بسيط)
        box_branch = layers.Flatten()(x)
        box_branch = layers.Dense(64, activation='relu')(box_branch)
        box_branch = layers.Dropout(0.5)(box_branch)
        box_output = layers.Dense(4, activation='sigmoid', name='bounding_box')(box_branch)
        
        # نموذج متعدد المخرجات
        self.model = Model(inputs=inputs, outputs=[classification_output, box_output])
        
        # تجميع النموذج
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'classification': 'categorical_crossentropy',
                'bounding_box': 'mse'
            },
            loss_weights={
                'classification': 1.0,
                'bounding_box': 0.5
            },
            metrics={
                'classification': ['accuracy']
            }
        )
        
        logger.info("تم إنشاء النموذج المخصص")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """معالجة مسبقة للصورة"""
        if self.model_type == 'cnn':
            target_size = self.input_size
        else:
            target_size = (256, 256)
        
        # تغيير الحجم
        img = cv2.resize(image, target_size)
        
        # تطبيع
        img = img.astype(np.float32) / 255.0
        
        # إضافة بُعد الدُفعة
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def detect_diseases(self, image_path: str, output_path: Optional[str] = None
                       ) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        كشف الأمراض في صورة
        
        Args:
            image_path: مسار الصورة
            output_path: مسار الحفظ (اختياري)
            
        Returns:
            tuple: (الصورة المحولة، قائمة بالكشوفات)
        """
        logger.info(f"معالجة الصورة: {image_path}")
        
        # التحقق من وجود الملف
        if not Path(image_path).exists():
            logger.error(f"الملف غير موجود: {image_path}")
            return None, []
        
        # قراءة الصورة
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"لا يمكن قراءة الصورة: {image_path}")
                return None, []
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"خطأ في قراءة الصورة: {e}")
            return None, []
        
        # التحقق من وجود النموذج
        if self.model is None:
            logger.error("النموذج غير مهيأ")
            return None, []
        
        # المعالجة المسبقة
        try:
            processed_img = self.preprocess_image(image_rgb)
            
            # التنبؤ
            if self.model_type == 'cnn':
                predictions = self.model.predict(processed_img, verbose=0)[0]
                
                # إيجاد أعلى ثقة
                class_idx = np.argmax(predictions)
                confidence = float(predictions[class_idx])
                
                # تحضير النتائج
                detections = [{
                    'bbox': None,  # لا يوجد صندوق محيط لـ CNN
                    'confidence': confidence,
                    'class_id': int(class_idx),
                    'class_name': self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown',
                    'is_healthy': 'healthy' in self.class_names[class_idx].lower()
                }]
            else:
                # نموذج مخصص متعدد المخرجات
                class_pred, bbox_pred = self.model.predict(processed_img, verbose=0)
                class_idx = np.argmax(class_pred[0])
                confidence = float(class_pred[0][class_idx])
                
                # صندوق محيط (بسيط)
                bbox = bbox_pred[0] * np.array([image.shape[1], image.shape[0], 
                                               image.shape[1], image.shape[0]])
                
                detections = [{
                    'bbox': bbox.tolist(),
                    'confidence': confidence,
                    'class_id': int(class_idx),
                    'class_name': self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown',
                    'is_healthy': 'healthy' in self.class_names[class_idx].lower()
                }]
            
            # رسم النتائج
            result_image = self._draw_detections(image_rgb.copy(), detections)
            
            # حفظ الناتج
            if output_path:
                self._save_output_image(result_image, output_path)
            
            logger.info(f"تم الكشف عن {len(detections)} مرض/أمراض")
            return result_image, detections
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة: {e}")
            return None, []
    
    def detect_from_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        كشف الأمراض من إطار فيديو مباشرة
        
        Args:
            frame: إطار الفيديو (صورة BGR)
            
        Returns:
            list: قائمة بالكشوفات
        """
        try:
            # تحويل من BGR إلى RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # المعالجة المسبقة
            processed_img = self.preprocess_image(frame_rgb)
            
            # التنبؤ
            if self.model_type == 'cnn':
                predictions = self.model.predict(processed_img, verbose=0)[0]
                
                # إيجاد أعلى ثقة
                class_idx = np.argmax(predictions)
                confidence = float(predictions[class_idx])
                
                # تحضير النتائج
                detections = [{
                    'bbox': None,
                    'confidence': confidence,
                    'class_id': int(class_idx),
                    'class_name': self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown',
                    'is_healthy': 'healthy' in self.class_names[class_idx].lower()
                }]
            else:
                # نموذج مخصص متعدد المخرجات
                class_pred, bbox_pred = self.model.predict(processed_img, verbose=0)
                class_idx = np.argmax(class_pred[0])
                confidence = float(class_pred[0][class_idx])
                
                # صندوق محيط (بسيط)
                bbox = bbox_pred[0] * np.array([frame.shape[1], frame.shape[0], 
                                               frame.shape[1], frame.shape[0]])
                
                detections = [{
                    'bbox': bbox.tolist(),
                    'confidence': confidence,
                    'class_id': int(class_idx),
                    'class_name': self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown',
                    'is_healthy': 'healthy' in self.class_names[class_idx].lower()
                }]
            
            return detections
            
        except Exception as e:
            logger.error(f"خطأ في كشف الأمراض من الإطار: {e}")
            return []
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """رسم الصناديق المحيطة والتسميات على الصورة"""
        result = image.copy()
        
        # توليد ألوان
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        colors = (colors[:, :3] * 255).astype(int)
        
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            is_healthy = det['is_healthy']
            
            # اختيار اللون بناءً على الحالة
            if is_healthy:
                color = (0, 255, 0)  # أخضر للصحة
            elif confidence > 0.7:
                color = (0, 0, 255)  # أحمر للأمراض عالية الثقة
            else:
                color = (255, 165, 0)  # برتقالي للأمراض متوسطة الثقة
            
            # رسم الصندوق المحيط (إن وجد)
            if det['bbox'] is not None:
                bbox = det['bbox']
                x, y, w, h = map(int, bbox)
                
                if 0 <= x < result.shape[1] and 0 <= y < result.shape[0] and w > 0 and h > 0:
                    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # إضافة تسمية
            label = f"{class_name}: {confidence:.2f}"
            try:
                font_scale = 0.6
                thickness = 2
                
                # حساب حجم النص
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # تحديد موقع التسمية
                if det['bbox'] is not None:
                    label_x = max(0, x)
                    label_y = max(0, y - 10)
                else:
                    label_x = 10
                    label_y = 30
                
                # خلفية النص
                cv2.rectangle(result, 
                            (label_x, label_y - text_height - 5),
                            (label_x + text_width, label_y + 5),
                            color, -1)
                
                # النص
                cv2.putText(result, label, 
                          (label_x, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                          (255, 255, 255), thickness)
                
            except Exception as e:
                logger.warning(f"خطأ في إضافة التسمية: {e}")
        
        # إضافة تنبيه للأمراض الخطيرة
        high_risk_detections = [d for d in detections 
                              if not d['is_healthy'] and d['confidence'] > 0.7]
        
        if high_risk_detections:
            warning_text = "تحذير: تم الكشف عن أمراض نباتية!"
            cv2.putText(result, warning_text, 
                      (10, result.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
    
    def _save_output_image(self, image: np.ndarray, output_path: str):
        """حفظ الصورة الناتجة"""
        try:
            output_dir = Path(output_path).parent
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            logger.info(f"تم حفظ الناتج في: {output_path}")
        except Exception as e:
            logger.error(f"خطأ في حفظ الصورة: {e}")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """معالجة فيديو للكشف عن الأمراض"""
        logger.info(f"معالجة الفيديو: {video_path}")
        
        if not Path(video_path).exists():
            logger.error(f"الملف غير موجود: {video_path}")
            return None
        
        # فتح الفيديو
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"لا يمكن فتح الفيديو: {video_path}")
            return None
        
        # معلومات الفيديو
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0:
            fps = 30
            logger.warning("تم تعيين معدل الإطار إلى 30 افتراضياً")
        
        # إعداد كاتب الفيديو
        writer = None
        if output_path:
            try:
                output_dir = Path(output_path).parent
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if not writer.isOpened():
                    logger.error(f"لا يمكن إنشاء ملف الفيديو: {output_path}")
                    writer = None
            except Exception as e:
                logger.error(f"خطأ في إعداد كاتب الفيديو: {e}")
                writer = None
        
        frame_count = 0
        processed_count = 0
        logger.info(f"معلومات الفيديو: {width}x{height}, {fps:.1f} إطار/ثانية")
        
        # معالجة الإطارات
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # معالجة إطار من كل 10 إطارات لتحسين الأداء
            if frame_count % 10 == 0:
                try:
                    # كشف الأمراض في الإطار
                    detections = self.detect_from_frame(frame)
                    
                    if detections:
                        # تحويل الإطار إلى RGB للرسم
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # رسم النتائج
                        frame_with_detections = self._draw_detections(frame_rgb, detections)
                        frame_bgr = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"خطأ في معالجة الإطار {frame_count}: {e}")
                    frame_bgr = frame
            else:
                frame_bgr = frame
            
            # الكتابة إلى الملف
            if writer:
                writer.write(frame_bgr)
            
            # تحديث التقدم
            if frame_count % 100 == 0:
                logger.info(f"تم معالجة {frame_count} إطار ({processed_count} مع كشف)")
        
        # التنظيف
        cap.release()
        if writer:
            writer.release()
            logger.info(f"تم حفظ الفيديو في: {output_path}")
            return output_path
        
        return None
    
    def visualize_results(self, image: np.ndarray, detections: List[Dict[str, Any]]):
        """تصور النتائج"""
        plt.figure(figsize=(15, 6))
        
        # الصورة الأصلية مع الكشوفات
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("نتائج الكشف")
        plt.axis('off')
        
        # مخطط الثقة
        plt.subplot(1, 3, 2)
        if detections:
            classes = [d['class_name'] for d in detections]
            confidences = [d['confidence'] for d in detections]
            colors = ['green' if d['is_healthy'] else 'red' for d in detections]
            
            bars = plt.bar(range(len(classes)), confidences, color=colors)
            plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
            plt.ylabel('الثقة')
            plt.title('ثقة الكشف')
            plt.ylim(0, 1)
            
            # إضافة قيم الثقة
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{conf:.2f}', ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, 'لم يتم الكشف عن أمراض',
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # ملخص الإحصائيات
        plt.subplot(1, 3, 3)
        if detections:
            healthy_count = sum(1 for d in detections if d['is_healthy'])
            disease_count = len(detections) - healthy_count
            
            labels = ['صحي', 'مريض']
            sizes = [healthy_count, disease_count]
            colors_pie = ['green', 'red']
            
            plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('توزيع الصحة والمرض')
        else:
            plt.text(0.5, 0.5, 'لا بيانات',
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, detections: List[Dict[str, Any]], output_path: str):
        """توليد تقرير نصي عن النتائج"""
        try:
            report = []
            report.append("=" * 50)
            report.append("تقرير كشف أمراض النبات")
            report.append("=" * 50)
            report.append(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"عدد الكشوفات: {len(detections)}")
            report.append("-" * 50)
            
            # تفاصيل الكشوفات
            for i, det in enumerate(detections, 1):
                report.append(f"\nالكشف {i}:")
                report.append(f"  المرض: {det['class_name']}")
                report.append(f"  الثقة: {det['confidence']:.2%}")
                report.append(f"  الحالة: {'صحي' if det['is_healthy'] else 'مريض'}")
            
            # ملخص
            report.append("\n" + "-" * 50)
            healthy_count = sum(1 for d in detections if d['is_healthy'])
            disease_count = len(detections) - healthy_count
            
            report.append("الملخص:")
            report.append(f"  النباتات الصحية: {healthy_count}")
            report.append(f"  النباتات المريضة: {disease_count}")
            
            if disease_count > 0:
                report.append("  الحالة: تحتاج إلى تدخل")
            else:
                report.append("  الحالة: جيدة")
            
            report.append("=" * 50)
            
            # حفظ التقرير
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            logger.info(f"تم حفظ التقرير في: {output_path}")
            
        except Exception as e:
            logger.error(f"خطأ في توليد التقرير: {e}")


def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(
        description='نظام كشف أمراض النبات باستخدام التعلم الآلي',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='مسار صورة أو فيديو الإدخال')
    parser.add_argument('--output', type=str, default=None,
                       help='مسار حفظ الناتج (اختياري)')
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'custom'],
                       help='نوع النموذج (cnn أو custom)')
    parser.add_argument('--visualize', action='store_true',
                       help='عرض تصور النتائج')
    parser.add_argument('--report', action='store_true',
                       help='توليد تقرير نصي')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='حد الثقة للكشف (0.0 إلى 1.0)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("نظام كشف أمراض النبات باستخدام التعلم الآلي")
    print("=" * 60)
    
    # التحقق من المدخلات
    if not os.path.exists(args.input):
        print(f"خطأ: ملف الإدخال غير موجود: {args.input}")
        sys.exit(1)
    
    # إنشاء الكاشف
    detector = PlantDiseaseDetector(model_type=args.model)
    detector.confidence_threshold = args.threshold
    
    # إعداد البيئة
    if not detector.setup_environment():
        print("فشل في إعداد البيئة. الخروج...")
        sys.exit(1)
    
    # تحديد نوع الملف
    file_ext = os.path.splitext(args.input)[1].lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        print(f"معالجة صورة: {args.input}")
        
        # الكشف عن الأمراض
        result_image, detections = detector.detect_diseases(
            args.input,
            args.output
        )
        
        if result_image is not None:
            print(f"\nنتائج الكشف:")
            print("-" * 40)
            
            for det in detections:
                status = "صحي" if det['is_healthy'] else "مريض"
                print(f"  • {det['class_name']}: {det['confidence']:.2%} ({status})")
            
            print("-" * 40)
            
            # عرض التصور
            if args.visualize:
                detector.visualize_results(result_image, detections)
            
            # توليد تقرير
            if args.report:
                report_path = args.output.replace('.jpg', '_report.txt') if args.output else 'output/report.txt'
                detector.generate_report(detections, report_path)
    
    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        print(f"معالجة فيديو: {args.input}")
        
        output_video = args.output or 'output/detected_video.mp4'
        result = detector.process_video(args.input, output_video)
        
        if result:
            print(f"✓ تم معالجة الفيديو وحفظه في: {result}")
        else:
            print("✗ فشل في معالجة الفيديو")
    
    else:
        print(f"خطأ: تنسيق ملف غير مدعوم: {file_ext}")
        print("التنسيقات المدعومة:")
        print("  الصور: .jpg, .png, .bmp, .tiff")
        print("  الفيديو: .mp4, .avi, .mov, .mkv")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("اكتملت المعالجة بنجاح!")
    print("=" * 60)


if __name__ == "__main__":
    main()