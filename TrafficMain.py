# traffic_sign_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import pickle
import os
import requests
import zipfile
import io
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚦 Traffic Sign Recognition System")
        self.root.geometry("1200x800")
        
        # Traffic sign classes (GTSRB - 43 classes)
        self.classes = [
            'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
            'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
            'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
            'No passing', 'No passing for vehicles over 3.5 tons', 
            'Right-of-way at intersection', 'Priority road', 'Yield', 'Stop', 
            'No vehicles', 'Vehicles over 3.5 tons prohibited', 'No entry',
            'General caution', 'Dangerous curve left', 'Dangerous curve right', 
            'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 
            'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 
            'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
            'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
            'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
            'Keep left', 'Roundabout mandatory', 'End of no passing',
            'End of no passing by vehicles over 3.5 tons'
        ]
        
        # Common signs for quick reference
        self.common_signs = {
            0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
            13: 'Yield', 14: 'Stop', 17: 'No entry', 25: 'Road work',
            33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
            38: 'Keep right', 39: 'Keep left'
        }
        
        # Initialize model and data
        self.model = None
        self.user_data = {'images': [], 'labels': []}
        self.model_path = 'traffic_sign_cnn.h5'
        self.data_path = 'user_training_data.npz'
        self.dataset_path = 'gtsrb_dataset'
        
        # Current image and prediction
        self.current_image = None
        self.current_prediction = None
        
        # Setup GUI
        self.setup_gui()
        
        # Load or create model
        self.load_or_create_model()
        
        # Load user data if exists
        self.load_user_data()
    
    def setup_gui(self):
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.root.configure(bg='#2c3e50')
        
        # Create main frames
        top_frame = tk.Frame(self.root, bg='#2c3e50')
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
        
        middle_frame = tk.Frame(self.root, bg='#34495e')
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        bottom_frame = tk.Frame(self.root, bg='#2c3e50')
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)
        
        # Title
        title_label = tk.Label(top_frame, 
                             text="🚦 Traffic Sign Recognition System", 
                             font=('Arial', 24, 'bold'),
                             fg='#ecf0f1',
                             bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Left panel for image display
        left_panel = tk.Frame(middle_frame, bg='#34495e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image display area
        self.image_label = tk.Label(left_panel, 
                                   text="No Image Loaded\n\nClick 'Load Image' to begin", 
                                   font=('Arial', 14),
                                   bg='#2c3e50',
                                   fg='#bdc3c7',
                                   relief=tk.RAISED,
                                   bd=2)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for controls and results
        right_panel = tk.Frame(middle_frame, bg='#34495e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control buttons frame
        control_frame = tk.LabelFrame(right_panel, 
                                     text="Controls",
                                     font=('Arial', 12, 'bold'),
                                     bg='#34495e',
                                     fg='#ecf0f1',
                                     padx=20,
                                     pady=20)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Buttons
        btn_style = {'font': ('Arial', 11), 'height': 2, 'width': 15}
        
        self.load_btn = tk.Button(control_frame,
                                 text="📁 Load Image",
                                 command=self.load_image,
                                 bg='#3498db',
                                 fg='white',
                                 activebackground='#2980b9',
                                 **btn_style)
        self.load_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.predict_btn = tk.Button(control_frame,
                                    text="🔍 Predict",
                                    command=self.predict_sign,
                                    bg='#2ecc71',
                                    fg='white',
                                    activebackground='#27ae60',
                                    state=tk.DISABLED,
                                    **btn_style)
        self.predict_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.train_btn = tk.Button(control_frame,
                                  text="🎓 Train Model",
                                  command=self.train_model,
                                  bg='#e74c3c',
                                  fg='white',
                                  activebackground='#c0392b',
                                  **btn_style)
        self.train_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.capture_btn = tk.Button(control_frame,
                                    text="📸 Capture Webcam",
                                    command=self.capture_webcam,
                                    bg='#9b59b6',
                                    fg='white',
                                    activebackground='#8e44ad',
                                    **btn_style)
        self.capture_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Results frame
        result_frame = tk.LabelFrame(right_panel,
                                    text="Prediction Results",
                                    font=('Arial', 12, 'bold'),
                                    bg='#34495e',
                                    fg='#ecf0f1',
                                    padx=20,
                                    pady=20)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Prediction display
        self.result_text = tk.Text(result_frame,
                                  height=10,
                                  width=40,
                                  font=('Courier', 12),
                                  bg='#2c3e50',
                                  fg='#ecf0f1',
                                  relief=tk.FLAT,
                                  wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Insert initial text
        self.result_text.insert(tk.END, "Prediction results will appear here...\n\n")
        self.result_text.insert(tk.END, "Load an image and click 'Predict' to start.")
        self.result_text.config(state=tk.DISABLED)
        
        # Correction frame
        correction_frame = tk.LabelFrame(bottom_frame,
                                        text="Correct Prediction",
                                        font=('Arial', 12, 'bold'),
                                        bg='#2c3e50',
                                        fg='#ecf0f1',
                                        padx=20,
                                        pady=20)
        correction_frame.pack(fill=tk.X)
        
        # Correction controls
        tk.Label(correction_frame,
                text="If prediction is wrong, select correct sign:",
                font=('Arial', 10),
                bg='#2c3e50',
                fg='#ecf0f1').pack(side=tk.LEFT, padx=(0, 10))
        
        # Common signs dropdown
        self.correction_var = tk.StringVar()
        common_signs_list = [f"{k}: {v}" for k, v in self.common_signs.items()]
        self.correction_combo = ttk.Combobox(correction_frame,
                                           textvariable=self.correction_var,
                                           values=common_signs_list,
                                           state="readonly",
                                           width=40)
        self.correction_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.correction_combo.set("Select correct sign...")
        
        # Correction button
        self.correct_btn = tk.Button(correction_frame,
                                    text="✓ Correct & Learn",
                                    command=self.correct_prediction,
                                    bg='#f39c12',
                                    fg='white',
                                    activebackground='#e67e22',
                                    state=tk.DISABLED,
                                    font=('Arial', 11),
                                    height=1)
        self.correct_btn.pack(side=tk.LEFT)
        
        # Status bar
        self.status_bar = tk.Label(self.root,
                                  text="Ready",
                                  bd=1,
                                  relief=tk.SUNKEN,
                                  anchor=tk.W,
                                  font=('Arial', 10),
                                  bg='#34495e',
                                  fg='#ecf0f1')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Statistics label
        self.stats_label = tk.Label(bottom_frame,
                                   text="User corrections: 0 | Model accuracy: N/A",
                                   font=('Arial', 9),
                                   bg='#2c3e50',
                                   fg='#bdc3c7')
        self.stats_label.pack(pady=(0, 10))
    
    def load_or_create_model(self):
        """Load existing model or create a new one"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                self.update_status("✓ Loaded trained model")
                self.train_btn.config(text="🎓 Re-train Model")
                return True
            except:
                self.update_status("⚠ Could not load model, will create new one")
        
        # Create a simple CNN model
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(43, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        
        self.update_status("Created new CNN model - needs training")
        return False
    
    def load_user_data(self):
        """Load user training data if exists"""
        if os.path.exists(self.data_path):
            try:
                data = np.load(self.data_path, allow_pickle=True)
                self.user_data['images'] = list(data['images'])
                self.user_data['labels'] = list(data['labels'])
                self.update_status(f"✓ Loaded {len(self.user_data['labels'])} user corrections")
                self.update_stats()
            except:
                self.update_status("Could not load user data")
    
    def save_user_data(self):
        """Save user training data"""
        try:
            np.savez(self.data_path,
                    images=np.array(self.user_data['images'], dtype=object),
                    labels=np.array(self.user_data['labels']))
            self.update_status(f"Saved {len(self.user_data['labels'])} user corrections")
        except:
            self.update_status("Error saving user data")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def update_stats(self):
        """Update statistics label"""
        accuracy = "N/A"
        if hasattr(self.model, 'history') and self.model.history:
            if 'val_accuracy' in self.model.history.history:
                accuracy = f"{self.model.history.history['val_accuracy'][-1]:.1%}"
        
        self.stats_label.config(
            text=f"User corrections: {len(self.user_data['labels'])} | Model accuracy: {accuracy}"
        )
    
    def load_image(self):
        """Load an image from file"""
        filetypes = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.gif'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select a traffic sign image",
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Load and display image
                image = Image.open(filename)
                image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(image)
                
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                # Store original image
                self.current_image = cv2.imread(filename)
                
                # Enable prediction button
                self.predict_btn.config(state=tk.NORMAL)
                self.correct_btn.config(state=tk.DISABLED)
                
                self.update_status(f"Loaded: {os.path.basename(filename)}")
                
                # Clear results
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Image loaded. Click 'Predict' to analyze.")
                self.result_text.config(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def preprocess_image(self, image):
        """Preprocess image for the model"""
        # Resize to 32x32
        image = cv2.resize(image, (32, 32))
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Expand dimensions for batch
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_sign(self):
        """Predict traffic sign from loaded image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        try:
            self.update_status("Analyzing image...")
            
            # Preprocess image
            processed_image = self.preprocess_image(self.current_image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Get top predictions
            top_indices = np.argsort(predictions)[-5:][::-1]
            
            # Store current prediction
            self.current_prediction = {
                'top_class': int(top_indices[0]),
                'confidence': float(predictions[top_indices[0]]),
                'top_5': [(int(idx), self.classes[idx], float(predictions[idx])) 
                         for idx in top_indices]
            }
            
            # Display results
            self.display_results()
            
            # Enable correction button
            self.correct_btn.config(state=tk.NORMAL)
            
            self.update_status("Prediction complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.update_status("Prediction failed")
    
    def display_results(self):
        """Display prediction results in text box"""
        if not self.current_prediction:
            return
        
        pred = self.current_prediction
        top_class_name = self.classes[pred['top_class']]
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        # Add header
        self.result_text.insert(tk.END, "="*50 + "\n")
        self.result_text.insert(tk.END, "PREDICTION RESULTS\n")
        self.result_text.insert(tk.END, "="*50 + "\n\n")
        
        # Add top prediction
        confidence_color = "#2ecc71" if pred['confidence'] > 0.7 else "#e74c3c"
        self.result_text.insert(tk.END, "🚦 TOP PREDICTION:\n")
        self.result_text.insert(tk.END, f"  Sign: {top_class_name}\n")
        self.result_text.insert(tk.END, f"  Class ID: {pred['top_class']}\n")
        self.result_text.insert(tk.END, f"  Confidence: {pred['confidence']:.1%}\n\n")
        
        # Add top 5 predictions
        self.result_text.insert(tk.END, "📊 TOP 5 PREDICTIONS:\n")
        self.result_text.insert(tk.END, "-"*40 + "\n")
        
        for i, (class_id, class_name, confidence) in enumerate(pred['top_5'], 1):
            if i == 1:
                prefix = "🥇"
            elif i == 2:
                prefix = "🥈"
            elif i == 3:
                prefix = "🥉"
            else:
                prefix = f"{i}."
            
            self.result_text.insert(tk.END, 
                                  f"{prefix} {class_name[:30]:30} ({confidence:.1%})\n")
        
        # Add feedback instructions
        self.result_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.result_text.insert(tk.END, "FEEDBACK:\n")
        
        if pred['confidence'] < 0.5:
            self.result_text.insert(tk.END, "⚠ Low confidence prediction\n")
            self.result_text.insert(tk.END, "Please verify and correct if needed.\n")
        else:
            self.result_text.insert(tk.END, "✓ High confidence prediction\n")
            self.result_text.insert(tk.END, "Still wrong? Select correct sign below.\n")
        
        self.result_text.config(state=tk.DISABLED)
    
    def correct_prediction(self):
        """Correct a wrong prediction and learn from it"""
        if not self.current_prediction or self.current_image is None:
            messagebox.showwarning("Warning", "No prediction to correct!")
            return
        
        selected = self.correction_combo.get()
        if selected == "Select correct sign...":
            messagebox.showwarning("Warning", "Please select a correct sign!")
            return
        
        try:
            # Parse selected sign
            class_id = int(selected.split(":")[0])
            class_name = self.common_signs[class_id]
            
            # Ask for confirmation
            confirm = messagebox.askyesno(
                "Confirm Correction",
                f"Are you sure the correct sign is:\n\n"
                f"{class_name} (ID: {class_id})?\n\n"
                f"This will help improve the model."
            )
            
            if confirm:
                # Store correction
                self.user_data['images'].append(self.current_image.copy())
                self.user_data['labels'].append(class_id)
                
                # Save user data
                self.save_user_data()
                
                # Retrain with user data
                self.retrain_with_user_data()
                
                # Update results to show correction
                self.result_text.config(state=tk.NORMAL)
                self.result_text.insert(tk.END, "\n" + "="*50 + "\n")
                self.result_text.insert(tk.END, f"✓ CORRECTION SAVED:\n")
                self.result_text.insert(tk.END, f"Model learned: {class_name}\n")
                self.result_text.insert(tk.END, "Model will improve with more corrections!\n")
                self.result_text.config(state=tk.DISABLED)
                
                # Reset correction dropdown
                self.correction_combo.set("Select correct sign...")
                self.correct_btn.config(state=tk.DISABLED)
                
                self.update_status(f"Learned correction: {class_name}")
                self.update_stats()
                
        except Exception as e:
            messagebox.showerror("Error", f"Correction failed: {str(e)}")
    
    def retrain_with_user_data(self):
        """Retrain model with accumulated user data"""
        if len(self.user_data['labels']) < 3:
            self.update_status("Need at least 3 corrections to retrain")
            return
        
        try:
            self.update_status("Retraining with user corrections...")
            
            # Convert user data to numpy arrays
            X_user = []
            for img in self.user_data['images']:
                # Preprocess each image
                img_resized = cv2.resize(img, (32, 32))
                img_normalized = img_resized.astype('float32') / 255.0
                X_user.append(img_normalized)
            
            X_user = np.array(X_user)
            y_user = np.array(self.user_data['labels'])
            
            # Create data augmentation for user data
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False  # Traffic signs shouldn't be flipped
            )
            
            # Generate augmented data
            augmented_data = []
            augmented_labels = []
            
            for i in range(len(X_user)):
                img = X_user[i]
                label = y_user[i]
                
                # Reshape for augmentation
                img_reshaped = img.reshape((1,) + img.shape)
                
                # Generate 5 augmented versions
                for batch in datagen.flow(img_reshaped, batch_size=1):
                    augmented_data.append(batch[0])
                    augmented_labels.append(label)
                    if len(augmented_data) >= (i + 1) * 5:
                        break
            
            X_augmented = np.array(augmented_data)
            y_augmented = np.array(augmented_labels)
            
            # Combine with original user data
            X_combined = np.vstack([X_user, X_augmented])
            y_combined = np.hstack([y_user, y_augmented])
            
            # Shuffle data
            indices = np.arange(len(X_combined))
            np.random.shuffle(indices)
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
            
            # Retrain model (fine-tune)
            self.model.fit(
                X_combined, y_combined,
                epochs=10,
                batch_size=8,
                validation_split=0.2,
                verbose=0
            )
            
            # Save updated model
            self.model.save(self.model_path)
            
            self.update_status("Model retrained with user corrections!")
            
            # Show improvement message
            messagebox.showinfo(
                "Success",
                f"Model retrained with {len(self.user_data['labels'])} user corrections!\n"
                f"The model should now be better at recognizing similar signs."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Retraining failed: {str(e)}")
            self.update_status("Retraining failed")
    
    def train_model(self):
        """Train or re-train the model with synthetic data"""
        response = messagebox.askyesno(
            "Train Model",
            "This will generate synthetic traffic signs and train the model.\n"
            "It may take a few minutes. Continue?"
        )
        
        if not response:
            return
        
        try:
            self.update_status("Generating synthetic training data...")
            
            # Generate synthetic traffic signs
            X_train, y_train = self.generate_synthetic_data()
            
            if len(y_train) == 0:
                messagebox.showerror("Error", "Failed to generate training data")
                return
            
            # Create validation split
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Create data augmentation
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2]
            )
            
            # Train the model
            self.update_status("Training CNN model...")
            
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=32),
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=[
                    EarlyStopping(patience=5, restore_best_weights=True),
                    ModelCheckpoint(self.model_path, save_best_only=True)
                ],
                verbose=0
            )
            
            # Save final model
            self.model.save(self.model_path)
            
            # Show training results
            final_acc = history.history['val_accuracy'][-1]
            
            messagebox.showinfo(
                "Training Complete",
                f"Model trained successfully!\n\n"
                f"Training samples: {len(X_train)}\n"
                f"Validation accuracy: {final_acc:.1%}\n\n"
                f"The model is now ready to use!"
            )
            
            self.update_status(f"Model trained - Accuracy: {final_acc:.1%}")
            self.update_stats()
            self.train_btn.config(text="🎓 Re-train Model")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.update_status("Training failed")
    
    def generate_synthetic_data(self):
        """Generate synthetic traffic sign images"""
        X_data = []
        y_data = []
        
        # Create signs for a subset of important classes
        important_classes = list(self.common_signs.keys())
        
        for class_id in important_classes:
            # Generate 50 samples per class
            for i in range(50):
                img = self.create_synthetic_sign(class_id)
                if img is not None:
                    # Preprocess
                    img_resized = cv2.resize(img, (32, 32))
                    img_normalized = img_resized.astype('float32') / 255.0
                    
                    X_data.append(img_normalized)
                    y_data.append(class_id)
        
        return np.array(X_data), np.array(y_data)
    
    def create_synthetic_sign(self, class_id):
        """Create synthetic traffic sign image"""
        img_size = 64
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200
        
        center = (img_size // 2, img_size // 2)
        
        # Create different signs based on class_id
        if class_id in [0, 1, 2]:  # Speed limits
            # Red circle with number
            cv2.circle(img, center, 25, (0, 0, 255), -1)
            cv2.circle(img, center, 23, (255, 255, 255), 2)
            
            # Add number
            number = "20" if class_id == 0 else "30" if class_id == 1 else "50"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(number, font, 0.7, 2)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            cv2.putText(img, number, (text_x, text_y), font, 0.7, (255, 255, 255), 2)
        
        elif class_id == 13:  # Yield
            # Yellow triangle pointing down
            pts = np.array([
                [center[0], 15],
                [15, img_size - 15],
                [img_size - 15, img_size - 15]
            ])
            cv2.fillPoly(img, [pts], (0, 255, 255))
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "YIELD", (10, 40), font, 0.5, (0, 0, 0), 2)
        
        elif class_id == 14:  # Stop
            # Red octagon
            pts = []
            for i in range(8):
                angle = 2 * np.pi * i / 8 - np.pi / 8
                x = center[0] + 25 * np.cos(angle)
                y = center[1] + 25 * np.sin(angle)
                pts.append([x, y])
            cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (0, 0, 255))
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize("STOP", font, 0.6, 2)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            cv2.putText(img, "STOP", (text_x, text_y), font, 0.6, (255, 255, 255), 2)
        
        elif class_id == 17:  # No entry
            # Red circle with bar
            cv2.circle(img, center, 25, (0, 0, 255), -1)
            cv2.rectangle(img, (10, center[1] - 3), (img_size - 10, center[1] + 3), 
                         (255, 255, 255), -1)
        
        elif class_id == 38:  # Keep right
            # Blue rectangle with arrow
            cv2.rectangle(img, (10, 10), (img_size - 10, img_size - 10), (255, 150, 0), -1)
            cv2.arrowedLine(img, (20, 32), (44, 32), (255, 255, 255), 3, tipLength=0.3)
        
        elif class_id == 25:  # Road work
            # Orange diamond
            pts = np.array([
                [center[0], 10],
                [10, center[1]],
                [center[0], img_size - 10],
                [img_size - 10, center[1]]
            ])
            cv2.fillPoly(img, [pts], (0, 165, 255))
            
            # Add person symbol
            cv2.circle(img, (center[0], center[1] - 5), 3, (255, 255, 255), -1)
            cv2.line(img, (center[0], center[1] - 2), (center[0], center[1] + 8), 
                    (255, 255, 255), 2)
            cv2.line(img, (center[0] - 5, center[1] + 3), (center[0] + 5, center[1] + 3), 
                    (255, 255, 255), 2)
        
        else:
            # Generic sign - blue circle
            cv2.circle(img, center, 25, (255, 150, 0), -1)
            cv2.putText(img, str(class_id), (center[0] - 8, center[1] + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add random noise and variations
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Random brightness/contrast
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.randint(-30, 30)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Random blur
        if np.random.random() > 0.7:
            kernel_size = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        return img
    
    def capture_webcam(self):
        """Capture image from webcam"""
        messagebox.showinfo(
            "Webcam Capture",
            "This feature requires a webcam.\n\n"
            "Press 'Space' to capture image or 'ESC' to cancel."
        )
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        
        self.update_status("Press SPACE to capture, ESC to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display webcam feed
            display_frame = cv2.resize(frame, (400, 300))
            cv2.imshow("Webcam - Press SPACE to capture", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # Space bar
                # Capture image
                self.current_image = frame.copy()
                
                # Convert and display
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                pil_image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(pil_image)
                
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                # Enable prediction
                self.predict_btn.config(state=tk.NORMAL)
                self.correct_btn.config(state=tk.DISABLED)
                
                self.update_status("Image captured from webcam")
                break
            
            elif key == 27:  # ESC
                self.update_status("Webcam capture cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    # Check required packages
    required = ['tensorflow', 'opencv-python', 'pillow', 'numpy']
    
    print("🚦 Traffic Sign Recognition System")
    print("=" * 50)
    print("Requirements check:")
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} not installed")
            print(f"  Run: pip install {package}")
    
    print("\nStarting GUI application...")
    
    # Create and run application
    root = tk.Tk()
    app = TrafficSignApp(root)
    app.run()

if __name__ == "__main__":
    main()