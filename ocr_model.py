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


# فئة تطبيق استخراج النص
class TextExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Extractor - EasyOCR Version")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.extracted_text = ""
        self.processing_mode = "fast"
        self.reader = None
        self.is_initializing = False
        self.ocr_initialized = False
        
        self.setup_styles()
        self.create_widgets()
        self.init_ocr_background()
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.bg_color = "#f0f0f0"
        self.primary_color = "#4a6fa5"
        self.secondary_color = "#166088"
        self.accent_color = "#4fc3a1"
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.LabelFrame(main_frame, text="Camera / Image", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(left_frame, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.camera_btn = ttk.Button(
            control_frame, 
            text="Start Camera", 
            command=self.toggle_camera,
            style="Accent.TButton"
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
        
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 5))
        self.mode_var = tk.StringVar(value="fast")
        ttk.Radiobutton(mode_frame, text="Fast", variable=self.mode_var, 
                       value="fast", command=self.change_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Accurate", variable=self.mode_var,
                       value="accurate", command=self.change_mode).pack(side=tk.LEFT, padx=(5, 0))
        
        right_frame = ttk.LabelFrame(main_frame, text="Extracted Text", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        text_container = ttk.Frame(right_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.text_display = tk.Text(
            text_container,
            wrap=tk.WORD,
            font=("Courier", 11),
            bg="white",
            relief=tk.SUNKEN,
            borderwidth=1
        )
        
        scrollbar = ttk.Scrollbar(text_container, command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=scrollbar.set)
        
        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
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
        
        threshold_frame = ttk.Frame(stats_frame)
        threshold_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(threshold_frame, text="Confidence:").pack(side=tk.LEFT, padx=(0, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(threshold_frame, from_=0.1, to=0.9, 
                                     variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.confidence_label = ttk.Label(threshold_frame, text="0.5")
        self.confidence_label.pack(side=tk.LEFT)
        
        self.confidence_var.trace("w", self.update_confidence_label)
        
        self.status_bar = ttk.Label(
            self.root,
            text="Initializing OCR engine...",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def init_ocr_background(self):
        if not self.is_initializing:
            self.is_initializing = True
            thread = threading.Thread(target=self.init_easyocr)
            thread.daemon = True
            thread.start()
        
    def init_easyocr(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
            self.ocr_initialized = True
            self.root.after(0, self.update_status, "OCR engine ready")
            self.root.after(0, self.enable_buttons)
        except ImportError:
            self.root.after(0, self.show_install_instructions)
        except Exception as e:
            self.root.after(0, self.show_error, f"Failed to initialize OCR: {str(e)}")
        finally:
            self.is_initializing = False
            
    def update_status(self, message):
        self.status_bar.config(text=message)
        
    def enable_buttons(self):
        self.capture_btn.config(state=tk.NORMAL)
        
    def toggle_camera(self):
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Cannot access camera")
            self.camera_btn.config(state=tk.DISABLED)
            return
            
        self.camera_active = True
        self.camera_btn.config(text="Stop Camera")
        self.capture_btn.config(state=tk.NORMAL)
        self.update_camera()
        
    def stop_camera(self):
        self.camera_active = False
        self.camera_btn.config(text="Start Camera")
        self.capture_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')
        if hasattr(self.video_label, 'image_ref'):
            delattr(self.video_label, 'image_ref')
        
    def update_camera(self):
        if self.camera_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame = frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                height, width = frame.shape[:2]
                max_size = 600
                if width > max_size or height > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(img)
                
                self.video_label.config(image=imgtk)
                self.video_label.image_ref = imgtk
                
            if self.camera_active:
                self.root.after(30, self.update_camera)
            
    def capture_and_extract(self):
        if self.current_frame is not None and self.current_frame.size > 0:
            if self.is_initializing:
                messagebox.showinfo("Please wait", "OCR engine is still initializing...")
                return
                
            if not self.ocr_initialized or self.reader is None:
                messagebox.showerror("OCR not ready", "OCR engine failed to initialize")
                return
                
            self.status_bar.config(text="Processing image...")
            thread = threading.Thread(target=self.process_image, args=(self.current_frame.copy(),))
            thread.daemon = True
            thread.start()
            
    def process_image(self, image):
        try:
            if image is None or image.size == 0:
                self.root.after(0, self.show_error, "Invalid image")
                return
                
            processed = self.preprocess_image(image)
            confidence_threshold = self.confidence_var.get()
            
            if len(processed.shape) == 2:
                rgb_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = processed
                
            results = self.reader.readtext(rgb_image, 
                                          paragraph=True,
                                          contrast_ths=0.3,
                                          adjust_contrast=0.7,
                                          width_ths=0.7,
                                          decoder='beamsearch',
                                          beamWidth=5)
            
            extracted_text = ""
            for (bbox, text, prob) in results:
                if prob >= confidence_threshold:
                    extracted_text += text + "\n"
            
            if not extracted_text.strip():
                rgb_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.reader.readtext(rgb_original, paragraph=True)
                for (bbox, text, prob) in results:
                    if prob >= confidence_threshold:
                        extracted_text += text + "\n"
            
            annotated_image = self.draw_bounding_boxes(image.copy(), results, confidence_threshold)
            
            self.root.after(0, self.update_text_display, extracted_text, annotated_image, processed)
            
        except Exception as e:
            self.root.after(0, self.show_error, f"Error processing image: {str(e)}")
            
    def draw_bounding_boxes(self, image, results, confidence_threshold):
        for (bbox, text, prob) in results:
            if prob >= confidence_threshold:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                
                color = (0, 255, 0) if prob > 0.7 else (0, 165, 255)
                cv2.rectangle(image, top_left, bottom_right, color, 2)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(f"{prob:.2f}", font, font_scale, thickness)[0]
                
                text_bg_top_left = (top_left[0], top_left[1] - text_size[1] - 5)
                text_bg_bottom_right = (top_left[0] + text_size[0] + 10, top_left[1] - 5)
                cv2.rectangle(image, text_bg_top_left, text_bg_bottom_right, color, -1)
                
                cv2.putText(image, f"{prob:.2f}", 
                           (top_left[0] + 5, top_left[1] - 10),
                           font, font_scale, (255, 255, 255), thickness)
                
        return image
        
    def preprocess_image(self, image):
        if image is None or image.size == 0:
            return image
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.processing_mode == "accurate":
            denoised = cv2.fastNlMeansDenoising(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        else:
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
            
        return processed
        
    def update_text_display(self, text, annotated_image=None, processed_image=None):
        self.extracted_text = text
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, text)
        
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.split('\n'))
        
        self.char_count_label.config(text=f"Characters: {char_count}")
        self.word_count_label.config(text=f"Words: {word_count}")
        self.line_count_label.config(text=f"Lines: {line_count}")
        
        status_text = f"Text extracted: {word_count} words, {char_count} characters"
        if char_count == 0:
            status_text += " (No text detected)"
        self.status_bar.config(text=status_text)
        
        if annotated_image is not None:
            self.show_annotated_image(annotated_image, processed_image)
            
    def show_annotated_image(self, annotated_image, processed_image):
        if annotated_image is None or annotated_image.size == 0:
            return
            
        height, width = annotated_image.shape[:2]
        max_size = 300
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            annotated_image = cv2.resize(annotated_image, (new_width, new_height))
            
            if processed_image is not None:
                processed_image = cv2.resize(processed_image, (new_width, new_height))
        
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        annotated_img = Image.fromarray(annotated_rgb)
        annotated_imgtk = ImageTk.PhotoImage(annotated_img)
        
        result_window = tk.Toplevel(self.root)
        result_window.title("Detection Results")
        result_window.geometry(f"{annotated_img.width * 2 + 30}x{annotated_img.height + 100}")
        
        annotated_label = ttk.Label(result_window, image=annotated_imgtk)
        annotated_label.image_ref = annotated_imgtk
        annotated_label.grid(row=0, column=0, padx=10, pady=10)
        ttk.Label(result_window, text="Detected Text (Green=High confidence)", 
                 font=("Arial", 10)).grid(row=1, column=0)
        
        if processed_image is not None and processed_image.size > 0:
            if len(processed_image.shape) == 2:
                processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
            else:
                processed_rgb = processed_image
                
            processed_img = Image.fromarray(processed_rgb)
            processed_imgtk = ImageTk.PhotoImage(processed_img)
            
            processed_label = ttk.Label(result_window, image=processed_imgtk)
            processed_label.image_ref = processed_imgtk
            processed_label.grid(row=0, column=1, padx=10, pady=10)
            ttk.Label(result_window, text="Preprocessed Image", 
                     font=("Arial", 10)).grid(row=1, column=1)
            
    def load_image(self):
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
            self.status_bar.config(text=f"Loading {os.path.basename(file_path)}...")
            thread = threading.Thread(target=self.load_and_process_image, args=(file_path,))
            thread.daemon = True
            thread.start()
            
    def load_and_process_image(self, file_path):
        try:
            image = cv2.imread(file_path)
            if image is None:
                self.root.after(0, self.show_error, "Could not read image file")
                return
                
            self.current_frame = image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            max_size = 600
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
                
            img = Image.fromarray(image_rgb)
            imgtk = ImageTk.PhotoImage(img)
            
            self.root.after(0, self.display_loaded_image, imgtk)
            self.process_image(image)
            
        except Exception as e:
            self.root.after(0, self.show_error, f"Error loading image: {str(e)}")
            
    def display_loaded_image(self, imgtk):
        self.video_label.config(image=imgtk)
        self.video_label.image_ref = imgtk
        self.status_bar.config(text="Image loaded. Processing...")
        
    def update_confidence_label(self, *args):
        self.confidence_label.config(text=f"{self.confidence_var.get():.2f}")
        
    def change_mode(self):
        self.processing_mode = self.mode_var.get()
        self.status_bar.config(text=f"Mode changed to {self.processing_mode}")
        
    def copy_to_clipboard(self):
        if self.extracted_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.extracted_text)
            self.status_bar.config(text="Text copied to clipboard")
            
    def save_to_file(self):
        if not self.extracted_text:
            messagebox.showwarning("No Text", "No text to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.extracted_text)
                self.status_bar.config(text=f"Text saved to {os.path.basename(file_path)}")
            except Exception as e:
                self.show_error(f"Error saving file: {str(e)}")
                
    def clear_text(self):
        self.extracted_text = ""
        self.text_display.delete(1.0, tk.END)
        self.char_count_label.config(text="Characters: 0")
        self.word_count_label.config(text="Words: 0")
        self.line_count_label.config(text="Lines: 0")
        self.status_bar.config(text="Text cleared")
        
    def show_install_instructions(self):
        response = messagebox.askyesno(
            "Installation Required",
            "EasyOCR is not installed. Would you like to install it now?\n\n"
            "This may take a few minutes and require an internet connection."
        )
        
        if response:
            self.install_easyocr()
        else:
            self.show_error("EasyOCR is required for this application")
        
    def install_easyocr(self):
        self.status_bar.config(text="Installing EasyOCR...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            
            self.status_bar.config(text="Installation complete. Initializing OCR...")
            self.init_ocr_background()
            
        except subprocess.CalledProcessError:
            self.show_error("Failed to install EasyOCR. Please install manually: pip install easyocr")
        except Exception as e:
            self.show_error(f"Installation error: {str(e)}")
            
    def show_error(self, message):
        messagebox.showerror("Error", message)
        self.status_bar.config(text="Error occurred")
        
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TextExtractorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()


if __name__ == "__main__":
    main()
