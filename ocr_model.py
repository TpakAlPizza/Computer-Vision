# استيراد المكتبات المطلوبة
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
import os
import sys
import subprocess
from queue import Queue
import traceback
from typing import Optional, List, Tuple, Any
import gc

# فئة تطبيق استخراج النص
class TextExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Extractor - EasyOCR Version")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        self.camera_active = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_lock = threading.Lock()
        self.extracted_text = ""
        self.processing_mode = "fast"
        self.reader: Any = None  # EasyOCR reader
        self.is_initializing = False
        self.ocr_initialized = False
        self.image_queue = Queue(maxsize=2)
        self.languages = ['en']
        self.use_gpu = self.check_gpu_availability()
        self.should_stop_camera = threading.Event()
        
        # Initialize variables for image references
        self.current_image_ref = None
        self.video_image_ref = None
        
        self.setup_styles()
        self.create_widgets()
        self.bind_shortcuts()
        self.init_ocr_background()
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame - Camera/Image
        left_frame = ttk.LabelFrame(main_frame, text="Camera / Image", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(left_frame, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.camera_btn = ttk.Button(
            control_frame, 
            text="Start Camera", 
            command=self.toggle_camera
        )
        self.camera_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.capture_btn = ttk.Button(
            control_frame,
            text="Capture & Extract",
            command=self.capture_and_extract,
            state=tk.DISABLED
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Load Image",
            command=self.load_image
        ).pack(side=tk.LEFT, padx=5)
        
        # Language selection
        lang_frame = ttk.Frame(control_frame)
        lang_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(lang_frame, text="Language:").pack(side=tk.LEFT, padx=(0, 5))
        self.lang_var = tk.StringVar(value="en")
        self.lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, 
                                      values=['en', 'ar', 'fr', 'es', 'de'], 
                                      state="readonly", width=5)
        self.lang_combo.pack(side=tk.LEFT)
        self.lang_combo.bind('<<ComboboxSelected>>', self.change_language)
        
        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 5))
        self.mode_var = tk.StringVar(value="fast")
        ttk.Radiobutton(mode_frame, text="Fast", variable=self.mode_var, 
                       value="fast", command=self.change_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Accurate", variable=self.mode_var,
                       value="accurate", command=self.change_mode).pack(side=tk.LEFT, padx=(5, 0))
        
        # Right frame - Extracted Text
        right_frame = ttk.LabelFrame(main_frame, text="Extracted Text", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        text_container = ttk.Frame(right_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.text_display = tk.Text(
            text_container,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="white",
            relief=tk.SUNKEN,
            borderwidth=1
        )
        
        scrollbar = ttk.Scrollbar(text_container, command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=scrollbar.set)
        
        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            action_frame,
            text="Copy to Clipboard",
            command=self.copy_to_clipboard
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            action_frame,
            text="Save to File",
            command=self.save_to_file
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            action_frame,
            text="Clear",
            command=self.clear_text
        ).pack(side=tk.LEFT, padx=5)
        
        # Statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        self.char_count_label = ttk.Label(stats_grid, text="Characters: 0")
        self.char_count_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.word_count_label = ttk.Label(stats_grid, text="Words: 0")
        self.word_count_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.line_count_label = ttk.Label(stats_grid, text="Lines: 0")
        self.line_count_label.grid(row=0, column=2, sticky=tk.W)
        
        # Confidence threshold
        threshold_frame = ttk.Frame(stats_frame)
        threshold_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(threshold_frame, text="Confidence:").pack(side=tk.LEFT, padx=(0, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(threshold_frame, from_=0.1, to=0.9, 
                                     variable=self.confidence_var, orient=tk.HORIZONTAL,
                                     length=150)
        confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.confidence_label = ttk.Label(threshold_frame, text="0.5")
        self.confidence_label.pack(side=tk.LEFT)
        
        self.confidence_var.trace("w", self.update_confidence_label)
        
        # Progress bar and status bar
        self.progress_bar = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))
        
        self.status_bar = ttk.Label(
            self.root,
            text="Initializing OCR engine...",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def bind_shortcuts(self):
        self.root.bind('<Control-o>', lambda e: self.load_image())
        self.root.bind('<Control-s>', lambda e: self.save_to_file())
        self.root.bind('<Control-c>', lambda e: self.copy_to_clipboard())
        self.root.bind('<Escape>', lambda e: self.stop_camera())
        self.root.bind('<F5>', lambda e: self.capture_and_extract())
        
    def check_gpu_availability(self):
        """Check if GPU is available for EasyOCR"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            print("PyTorch not installed. Running on CPU.")
            return False
        except Exception as e:
            print(f"Error checking GPU: {e}")
            return False
            
    def init_ocr_background(self):
        """Initialize EasyOCR in background thread"""
        if self.is_initializing:
            return
            
        self.is_initializing = True
        self.progress_bar.start()
        thread = threading.Thread(target=self.init_easyocr, daemon=True)
        thread.start()
        
    def init_easyocr(self):
        """Initialize EasyOCR reader"""
        try:
            # Try to import easyocr
            try:
                import easyocr
            except ImportError:
                self.root.after(0, self.show_install_instructions)
                return
                
            # Update status
            gpu_status = "with GPU" if self.use_gpu else "without GPU"
            self.root.after(0, self.update_status, f"Initializing OCR {gpu_status}...")
            
            # Initialize reader
            self.reader = easyocr.Reader(
                self.languages, 
                gpu=self.use_gpu,
                verbose=False
            )
            
            self.ocr_initialized = True
            self.root.after(0, self.update_status, f"OCR engine ready ({gpu_status})")
            self.root.after(0, self.enable_buttons)
            
        except Exception as e:
            error_msg = f"Failed to initialize OCR: {str(e)}"
            print(traceback.format_exc())
            self.root.after(0, self.show_error, error_msg)
        finally:
            self.is_initializing = False
            self.root.after(0, self.progress_bar.stop)
            
    def update_status(self, message):
        """Update status bar text"""
        self.status_bar.config(text=message)
        
    def enable_buttons(self):
        """Enable capture button after OCR initialization"""
        self.capture_btn.config(state=tk.NORMAL)
        
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """Start camera capture"""
        self.should_stop_camera.clear()
        
        # Try camera indices
        for i in range(3):
            try:
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    break
            except:
                continue
                
        if self.cap is None or not self.cap.isOpened():
            self.show_error("Cannot access any camera")
            self.camera_btn.config(state=tk.DISABLED)
            return
            
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.camera_active = True
        self.camera_btn.config(text="Stop Camera")
        self.capture_btn.config(state=tk.NORMAL)
        self.update_status("Camera started")
        
        # Start camera update loop
        self.update_camera()
        
    def stop_camera(self):
        """Stop camera capture"""
        self.should_stop_camera.set()
        self.camera_active = False
        self.camera_btn.config(text="Start Camera")
        self.capture_btn.config(state=tk.DISABLED)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear video display
        self.video_label.config(image='')
        self.video_label.image = None
        self.video_image_ref = None
            
        self.update_status("Camera stopped")
        
    def update_camera(self):
        """Update camera frame in UI"""
        if not self.camera_active or self.cap is None or not self.cap.isOpened():
            return
            
        if self.should_stop_camera.is_set():
            return
            
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Store frame
                with self.current_frame_lock:
                    self.current_frame = frame.copy()
                
                # Resize for display
                height, width = frame.shape[:2]
                max_size = 600
                
                if width > max_size or height > max_size:
                    scale = min(max_size/width, max_size/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_resized = cv2.resize(frame, (new_width, new_height))
                else:
                    frame_resized = frame
                
                # Convert for display
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update display with new reference
                self.video_label.config(image=imgtk)
                self.video_label.image = imgtk
                self.video_image_ref = imgtk
                
            # Schedule next update
            if self.camera_active and not self.should_stop_camera.is_set():
                self.root.after(33, self.update_camera)
                
        except Exception as e:
            print(f"Error updating camera: {e}")
            if self.camera_active:
                self.root.after(100, self.update_camera)
        
    def capture_and_extract(self):
        """Capture current frame and extract text"""
        with self.current_frame_lock:
            if self.current_frame is None or self.current_frame.size == 0:
                messagebox.showwarning("No Image", "Please capture or load an image first")
                return
                
            frame_copy = self.current_frame.copy()
            
        if self.is_initializing:
            messagebox.showinfo("Please wait", "OCR engine is still initializing...")
            return
            
        if not self.ocr_initialized or self.reader is None:
            messagebox.showerror("OCR not ready", "OCR engine failed to initialize")
            return
            
        self.update_status("Processing image...")
        thread = threading.Thread(target=self.process_image, args=(frame_copy,), daemon=True)
        thread.start()
            
    def process_image(self, image):
        """Process image and extract text - FIXED UNPACKING ISSUE"""
        try:
            if image is None or image.size == 0:
                self.root.after(0, self.show_error, "Invalid image")
                return
                
            # Preprocess image
            processed = self.preprocess_image(image)
            confidence_threshold = self.confidence_var.get()
            
            # Convert to RGB for OCR
            if len(processed.shape) == 2:
                rgb_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = processed
                
            # Perform OCR - use simpler parameters to avoid format issues
            try:
                results = self.reader.readtext(rgb_image, paragraph=False)
            except Exception as ocr_error:
                print(f"OCR error: {ocr_error}")
                # Try with default parameters
                results = self.reader.readtext(rgb_image)
            
            # Extract text with confidence threshold - FIXED HERE
            extracted_text = ""
            for item in results:
                try:
                    # Handle different return formats
                    if len(item) == 3:
                        # Standard format: (bbox, text, confidence)
                        bbox, text, prob = item
                    elif len(item) == 2:
                        # Some versions return (bbox, text) without confidence
                        bbox, text = item
                        prob = 0.8  # Default confidence
                    else:
                        print(f"Unexpected item format: {item}")
                        continue
                    
                    if prob >= confidence_threshold:
                        extracted_text += text + "\n"
                except Exception as e:
                    print(f"Error processing OCR result: {e}")
                    continue
            
            # Annotate image with bounding boxes
            annotated_image = self.draw_bounding_boxes(image.copy(), results, confidence_threshold)
            
            # Update UI with results
            self.root.after(0, self.update_text_display, extracted_text, annotated_image, processed)
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            print(traceback.format_exc())
            self.root.after(0, self.show_error, error_msg)
            
    def draw_bounding_boxes(self, image, results, confidence_threshold):
        """Draw bounding boxes on image - FIXED UNPACKING ISSUE"""
        for item in results:
            try:
                # Handle different return formats
                if len(item) == 3:
                    # Standard format: (bbox, text, confidence)
                    bbox, text, prob = item
                elif len(item) == 2:
                    # Some versions return (bbox, text) without confidence
                    bbox, text = item
                    prob = 0.8  # Default confidence
                else:
                    continue
                
                if prob < confidence_threshold:
                    continue
                    
                # Convert bbox coordinates to integers
                bbox_array = np.array(bbox, dtype=np.int32)
                if len(bbox_array) < 4:
                    continue
                    
                top_left = tuple(bbox_array[0])
                bottom_right = tuple(bbox_array[2])
                
                # Color based on confidence
                if prob > 0.8:
                    color = (0, 255, 0)  # Green
                elif prob > 0.6:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 0, 255)  # Red
                    
                # Draw rectangle
                cv2.rectangle(image, top_left, bottom_right, color, 2)
                
                # Add confidence label if we have confidence value
                if prob != 0.8:  # Only if not default value
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    label = f"{prob:.2f}"
                    
                    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw label background
                    text_bg_top_left = (top_left[0], top_left[1] - text_size[1] - 5)
                    text_bg_bottom_right = (top_left[0] + text_size[0] + 10, top_left[1] - 5)
                    cv2.rectangle(image, text_bg_top_left, text_bg_bottom_right, color, -1)
                    
                    # Draw label text
                    cv2.putText(image, label, 
                               (top_left[0] + 5, top_left[1] - 10),
                               font, font_scale, (255, 255, 255), thickness)
            except Exception as e:
                print(f"Error drawing bounding box: {e}")
                continue
                    
        return image
        
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        if image is None or image.size == 0:
            return image
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if self.processing_mode == "accurate":
            # Advanced preprocessing
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            # Binarization
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        else:
            # Fast preprocessing
            processed = cv2.adaptiveThreshold(gray, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
            
        return processed
        
    def update_text_display(self, text, annotated_image=None, processed_image=None):
        """Update text display with extracted text"""
        self.extracted_text = text
        self.text_display.delete(1.0, tk.END)
        
        if text:
            self.text_display.insert(1.0, text)
        
        # Update statistics
        char_count = len(text)
        word_count = len(text.split())
        line_count = text.count('\n') + 1 if text else 0
        
        self.char_count_label.config(text=f"Characters: {char_count}")
        self.word_count_label.config(text=f"Words: {word_count}")
        self.line_count_label.config(text=f"Lines: {line_count}")
        
        status_text = f"Text extracted: {word_count} words, {char_count} characters"
        if char_count == 0:
            status_text += " (No text detected)"
        self.update_status(status_text)
        
        # Show results if images provided
        if annotated_image is not None:
            self.show_annotated_image(annotated_image, processed_image)
            
    def show_annotated_image(self, annotated_image, processed_image):
        """Show annotated and processed images in new window"""
        if annotated_image is None or annotated_image.size == 0:
            return
            
        # Create results window
        result_window = tk.Toplevel(self.root)
        result_window.title("Detection Results")
        result_window.transient(self.root)
        
        # Calculate display size
        height, width = annotated_image.shape[:2]
        max_height = 500
        
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            new_height = max_height
            annotated_image = cv2.resize(annotated_image, (new_width, new_height))
            if processed_image is not None:
                processed_image = cv2.resize(processed_image, (new_width, new_height))
        
        # Convert annotated image for display
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        annotated_img = Image.fromarray(annotated_rgb)
        annotated_imgtk = ImageTk.PhotoImage(image=annotated_img)
        
        # Create main frame
        main_result_frame = ttk.Frame(result_window)
        main_result_frame.pack(padx=10, pady=10)
        
        # Annotated image frame
        annotated_frame = ttk.LabelFrame(main_result_frame, text="Detected Text", padding=5)
        annotated_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        annotated_label = ttk.Label(annotated_frame, image=annotated_imgtk)
        annotated_label.image = annotated_imgtk
        annotated_label.pack()
        
        # Processed image frame (if available)
        if processed_image is not None and processed_image.size > 0:
            if len(processed_image.shape) == 2:
                processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
            else:
                processed_rgb = processed_image
                
            processed_img = Image.fromarray(processed_rgb)
            processed_imgtk = ImageTk.PhotoImage(image=processed_img)
            
            processed_frame = ttk.LabelFrame(main_result_frame, text="Preprocessed", padding=5)
            processed_frame.pack(side=tk.LEFT)
            
            processed_label = ttk.Label(processed_frame, image=processed_imgtk)
            processed_label.image = processed_imgtk
            processed_label.pack()
            
        # Close button
        close_btn = ttk.Button(result_window, text="Close", 
                              command=result_window.destroy)
        close_btn.pack(pady=(0, 10))
        
        # Center the result window
        result_window.update_idletasks()
        result_width = result_window.winfo_width()
        result_height = result_window.winfo_height()
        screen_width = result_window.winfo_screenwidth()
        screen_height = result_window.winfo_screenheight()
        x = (screen_width - result_width) // 2
        y = (screen_height - result_height) // 2
        result_window.geometry(f"{result_width}x{result_height}+{x}+{y}")
            
    def load_image(self):
        """Load image from file"""
        if self.is_initializing:
            messagebox.showinfo("Please wait", "OCR engine is still initializing...")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.update_status(f"Loading {os.path.basename(file_path)}...")
            thread = threading.Thread(target=self.load_and_process_image, args=(file_path,), daemon=True)
            thread.start()
            
    def load_and_process_image(self, file_path):
        """Load and process image in background thread"""
        try:
            # Read image
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if image is None:
                self.root.after(0, self.show_error, "Could not read image file")
                return
                
            # Store image
            with self.current_frame_lock:
                self.current_frame = image
            
            # Display image
            height, width = image.shape[:2]
            max_size = 600
            
            if width > max_size or height > max_size:
                scale = min(max_size/width, max_size/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_resized = cv2.resize(image, (new_width, new_height))
            else:
                image_resized = image
                
            # Convert for display
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.root.after(0, self.display_loaded_image, imgtk)
            
            # Process image for text extraction
            self.process_image(image)
            
        except Exception as e:
            error_msg = f"Error loading image: {str(e)}"
            print(traceback.format_exc())
            self.root.after(0, self.show_error, error_msg)
            
    def display_loaded_image(self, imgtk):
        """Display loaded image in UI"""
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
        self.current_image_ref = imgtk
        self.update_status("Image loaded. Processing...")
        
    def update_confidence_label(self, *args):
        """Update confidence threshold label"""
        self.confidence_label.config(text=f"{self.confidence_var.get():.2f}")
        
    def change_mode(self):
        """Change processing mode"""
        self.processing_mode = self.mode_var.get()
        self.update_status(f"Mode changed to {self.processing_mode}")
        
    def change_language(self, event=None):
        """Change OCR language"""
        new_lang = self.lang_var.get()
        if new_lang != self.languages[0]:
            self.languages = [new_lang]
            if self.reader:
                # Reinitialize with new language
                self.update_status(f"Reinitializing OCR with {new_lang}...")
                self.ocr_initialized = False
                self.reader = None
                gc.collect()
                self.init_ocr_background()
        
    def copy_to_clipboard(self):
        """Copy extracted text to clipboard"""
        if self.extracted_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.extracted_text)
            self.update_status("Text copied to clipboard")
            
    def save_to_file(self):
        """Save extracted text to file"""
        if not self.extracted_text:
            messagebox.showwarning("No Text", "No text to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), 
                      ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.extracted_text)
                self.update_status(f"Text saved to {os.path.basename(file_path)}")
            except Exception as e:
                self.show_error(f"Error saving file: {str(e)}")
                
    def clear_text(self):
        """Clear extracted text"""
        self.extracted_text = ""
        self.text_display.delete(1.0, tk.END)
        self.char_count_label.config(text="Characters: 0")
        self.word_count_label.config(text="Words: 0")
        self.line_count_label.config(text="Lines: 0")
        self.update_status("Text cleared")
        
    def show_install_instructions(self):
        """Show EasyOCR installation instructions"""
        response = messagebox.askyesno(
            "Installation Required",
            "EasyOCR is not installed. Would you like to install it now?\n\n"
            "This may take a few minutes and requires an internet connection."
        )
        
        if response:
            self.install_easyocr()
        else:
            self.show_error("EasyOCR is required for this application")
            self.root.after(1000, self.root.destroy)
        
    def install_easyocr(self):
        """Install EasyOCR"""
        self.update_status("Installing EasyOCR and dependencies...")
        self.progress_bar.start()
        
        def install_thread():
            try:
                # Install EasyOCR (it will install torch as dependency)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"])
                
                self.root.after(0, self.installation_complete)
                
            except subprocess.CalledProcessError as e:
                self.root.after(0, self.installation_failed, str(e))
            except Exception as e:
                self.root.after(0, self.installation_failed, str(e))
                
        thread = threading.Thread(target=install_thread, daemon=True)
        thread.start()
        
    def installation_complete(self):
        """Handle successful installation"""
        self.progress_bar.stop()
        self.update_status("Installation complete. Initializing OCR...")
        self.init_ocr_background()
        
    def installation_failed(self, error_msg):
        """Handle failed installation"""
        self.progress_bar.stop()
        self.show_error(f"Installation failed: {error_msg}\n\n"
                       "Please install manually:\n"
                       "pip install easyocr")
        
    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.update_status("Error occurred")
        
    def on_closing(self):
        """Clean up resources before closing"""
        # Stop camera
        self.should_stop_camera.set()
        if self.cap:
            self.cap.release()
        
        # Clear OCR resources
        if self.reader:
            try:
                del self.reader
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        
        # Destroy window
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = TextExtractorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()