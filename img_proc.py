import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import copy


class ImageProcessor:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing Tool")
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.temp_image = None

        # UI 생성
        self.create_ui()

        # 이미지 처리 파이프라인
        self.processing_pipeline = []

    def create_ui(self):
        # 이미지 로드 프레임
        load_frame = ttk.Frame(self.master, padding=10)
        load_frame.grid(row=0, column=0, sticky="ew")

        load_button = ttk.Button(load_frame, text="Load Image", command=self.load_image)
        load_button.pack(side="left", padx=5)

        self.image_label = ttk.Label(load_frame, text="No Image Loaded")
        self.image_label.pack(side="left", padx=5)

        # 기능 프레임
        function_frame = ttk.Frame(self.master, padding=10)
        function_frame.grid(row=1, column=0, sticky="ew")

        # Blur
        blur_frame = ttk.LabelFrame(function_frame, text="Blur", padding=5)
        blur_frame.pack(fill="x", padx=5, pady=5)
        self.blur_type_var = tk.StringVar(value="Gaussian")
        blur_types = ["Gaussian", "Median", "Average", "fastNlMeansDenoisingColored"]
        blur_type_menu = ttk.Combobox(blur_frame, textvariable=self.blur_type_var, values=blur_types, state="readonly")
        blur_type_menu.pack(side="left", padx=5)
        self.blur_level_var = tk.IntVar(value=3)
        self.blur_level_scale = ttk.Scale(blur_frame, from_=1, to=15, orient="horizontal", command=self.update_blur_level, variable=self.blur_level_var)
        self.blur_level_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.blur_label = ttk.Label(blur_frame, text="Level: 3")
        self.blur_label.pack(side="left", padx=5)
        self.blur_button = ttk.Button(blur_frame, text="Apply", command=self.apply_blur)
        self.blur_button.pack(side="left", padx=5)
        self.blur_view_button = ttk.Button(blur_frame, text="View", command=self.view_blur)
        self.blur_view_button.pack(side="left", padx=5)
        self.remove_blur_button = ttk.Button(blur_frame, text="Remove", command=self.remove_blur)
        self.remove_blur_button.pack(side="left", padx=5)

        # Contour
        contour_frame = ttk.LabelFrame(function_frame, text="Contour", padding=5)
        contour_frame.pack(fill="x", padx=5, pady=5)
        self.contour_type_var = tk.StringVar(value="Canny")
        contour_types = ["Canny", "Laplacian"]
        contour_type_menu = ttk.Combobox(contour_frame, textvariable=self.contour_type_var, values=contour_types, state="readonly")
        contour_type_menu.pack(side="left", padx=5)
        self.contour_level_var = tk.IntVar(value=100)
        self.contour_level_scale = ttk.Scale(contour_frame, from_=1, to=200, orient="horizontal", command=self.update_contour_level, variable=self.contour_level_var)
        self.contour_level_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.contour_label = ttk.Label(contour_frame, text="Level: 100")
        self.contour_label.pack(side="left", padx=5)
        self.contour_button = ttk.Button(contour_frame, text="Apply", command=self.apply_contour)
        self.contour_button.pack(side="left", padx=5)
        self.contour_view_button = ttk.Button(contour_frame, text="View", command=self.view_contour)
        self.contour_view_button.pack(side="left", padx=5)
        self.remove_contour_button = ttk.Button(contour_frame, text="Remove", command=self.remove_contour)
        self.remove_contour_button.pack(side="left", padx=5)

        # Contour Detection
        contour_detect_frame = ttk.LabelFrame(function_frame, text="Contour Detection", padding=5)
        contour_detect_frame.pack(fill="x", padx=5, pady=5)
        self.contour_detect_level_var = tk.IntVar(value=100)
        self.contour_detect_level_scale = ttk.Scale(contour_detect_frame, from_=1, to=200, orient="horizontal", command=self.update_contour_detect_level, variable=self.contour_detect_level_var)
        self.contour_detect_level_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.contour_detect_label = ttk.Label(contour_detect_frame, text="Level: 100")
        self.contour_detect_label.pack(side="left", padx=5)
        self.contour_detect_button = ttk.Button(contour_detect_frame, text="Apply", command=self.apply_contour_detection)
        self.contour_detect_button.pack(side="left", padx=5)
        self.contour_detect_view_button = ttk.Button(contour_detect_frame, text="View", command=self.view_contour_detection)
        self.contour_detect_view_button.pack(side="left", padx=5)
        self.remove_contour_detection_button = ttk.Button(contour_detect_frame, text="Remove", command=self.remove_contour_detection)
        self.remove_contour_detection_button.pack(side="left", padx=5)


        
        # Threshold
        threshold_frame = ttk.LabelFrame(function_frame, text="Threshold", padding=5)
        threshold_frame.pack(fill="x", padx=5, pady=5)
        self.threshold_type_var = tk.StringVar(value="Binary")
        threshold_types = ["Binary", "Binary Inverse", "Trunc", "Tozero", "Tozero Inverse"]
        threshold_type_menu = ttk.Combobox(threshold_frame, textvariable=self.threshold_type_var, values=threshold_types, state="readonly")
        threshold_type_menu.pack(side="left", padx=5)
        self.threshold_level_var = tk.IntVar(value=127)
        self.threshold_level_scale = ttk.Scale(threshold_frame, from_=0, to=255, orient="horizontal", command=self.update_threshold_level, variable=self.threshold_level_var)
        self.threshold_level_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.threshold_label = ttk.Label(threshold_frame, text="Level: 127")
        self.threshold_label.pack(side="left", padx=5)
        self.threshold_button = ttk.Button(threshold_frame, text="Apply", command=self.apply_threshold)
        self.threshold_button.pack(side="left", padx=5)
        self.threshold_view_button = ttk.Button(threshold_frame, text="View", command=self.view_threshold)
        self.threshold_view_button.pack(side="left", padx=5)
        self.remove_threshold_button = ttk.Button(threshold_frame, text="Remove", command=self.remove_threshold)
        self.remove_threshold_button.pack(side="left", padx=5)
        
        # Histogram
        histogram_frame = ttk.LabelFrame(function_frame, text="Histogram", padding=5)
        histogram_frame.pack(fill="x", padx=5, pady=5)
        self.histogram_equalization_var = tk.BooleanVar(value=False)
        histogram_check = ttk.Checkbutton(histogram_frame, text="Equalization", variable=self.histogram_equalization_var, command=self.update_histogram_equalization)
        histogram_check.pack(side="left", padx=5)
        self.histogram_view_button = ttk.Button(histogram_frame, text="View", command=self.view_histogram)
        self.histogram_view_button.pack(side="left", padx=5)

        # Dilation/Erosion
        dilation_erosion_frame = ttk.LabelFrame(function_frame, text="Dilation/Erosion", padding=5)
        dilation_erosion_frame.pack(fill="x", padx=5, pady=5)
        self.dilation_erosion_type_var = tk.StringVar(value="Dilation")
        dilation_erosion_types = ["Dilation", "Erosion"]
        dilation_erosion_type_menu = ttk.Combobox(dilation_erosion_frame, textvariable=self.dilation_erosion_type_var, values=dilation_erosion_types, state="readonly")
        dilation_erosion_type_menu.pack(side="left", padx=5)
        self.dilation_erosion_level_var = tk.IntVar(value=3)
        self.dilation_erosion_level_scale = ttk.Scale(dilation_erosion_frame, from_=1, to=10, orient="horizontal", command=self.update_dilation_erosion_level, variable=self.dilation_erosion_level_var)
        self.dilation_erosion_level_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.dilation_erosion_label = ttk.Label(dilation_erosion_frame, text="Level: 3")
        self.dilation_erosion_label.pack(side="left", padx=5)
        self.dilation_erosion_button = ttk.Button(dilation_erosion_frame, text="Apply", command=self.apply_dilation_erosion)
        self.dilation_erosion_button.pack(side="left", padx=5)
        self.dilation_erosion_view_button = ttk.Button(dilation_erosion_frame, text="View", command=self.view_dilation_erosion)
        self.dilation_erosion_view_button.pack(side="left", padx=5)
        self.remove_dilation_erosion_button = ttk.Button(dilation_erosion_frame, text="Remove", command=self.remove_dilation_erosion)
        self.remove_dilation_erosion_button.pack(side="left", padx=5)

        # Hough Line
        hough_line_frame = ttk.LabelFrame(function_frame, text="Hough Line", padding=5)
        hough_line_frame.pack(fill="x", padx=5, pady=5)
        self.hough_angle_var = tk.IntVar(value=90)
        hough_angle_scale = ttk.Scale(hough_line_frame, from_=0, to=180, orient="horizontal", command=self.update_hough_angle, variable=self.hough_angle_var)
        hough_angle_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.hough_angle_label = ttk.Label(hough_line_frame, text="Angle: 90")
        self.hough_angle_label.pack(side="left", padx=5)
        self.hough_thickness_var = tk.IntVar(value=1)
        hough_thickness_scale = ttk.Scale(hough_line_frame, from_=1, to=5, orient="horizontal", command=self.update_hough_thickness, variable=self.hough_thickness_var)
        hough_thickness_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.hough_thickness_label = ttk.Label(hough_line_frame, text="Thickness: 1")
        self.hough_thickness_label.pack(side="left", padx=5)
        self.hough_line_button = ttk.Button(hough_line_frame, text="Apply", command=self.apply_hough_line)
        self.hough_line_button.pack(side="left", padx=5)
        self.hough_line_view_button = ttk.Button(hough_line_frame, text="View", command=self.view_hough_line)
        self.hough_line_view_button.pack(side="left", padx=5)
        self.remove_hough_line_button = ttk.Button(hough_line_frame, text="Remove", command=self.remove_hough_line)
        self.remove_hough_line_button.pack(side="left", padx=5)

        # 이미지 표시 영역
        image_display_frame = ttk.Frame(self.master, padding=10)
        image_display_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")

        self.image_canvas = tk.Canvas(image_display_frame)
        self.image_canvas.pack(fill="both", expand=True)

        # 이미지 표시 프레임 조정
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        
    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.processed_image = self.original_image.copy()
            self.temp_image = self.original_image.copy()
            self.update_image_display(self.processed_image)
            self.image_label.config(text=f"Image Loaded: {self.image_path.split('/')[-1]}")
            self.processing_pipeline = {}
            
    def update_image_display(self, image):
        if image is not None:
            image = Image.fromarray(image)
            image = image.resize((640, 480))  # 이미지 크기 조정
            photo = ImageTk.PhotoImage(image)
            self.image_canvas.config(width=640, height=480)
            self.image_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.image_canvas.image = photo
        
    def update_blur_level(self, event):
        self.blur_label.config(text=f"Level: {self.blur_level_var.get()}")

    def update_contour_level(self, event):
        self.contour_label.config(text=f"Level: {self.contour_level_var.get()}")
    
    def update_contour_detect_level(self, event):
        self.contour_detect_label.config(text=f"Level: {self.contour_detect_level_var.get()}")

    def update_threshold_level(self, event):
        self.threshold_label.config(text=f"Level: {self.threshold_level_var.get()}")
    
    def update_histogram_equalization(self):
        pass
    
    def update_dilation_erosion_level(self, event):
        self.dilation_erosion_label.config(text=f"Level: {self.dilation_erosion_level_var.get()}")
        
    def update_hough_angle(self, event):
        self.hough_angle_label.config(text=f"Angle: {self.hough_angle_var.get()}")
    
    def update_hough_thickness(self, event):
        self.hough_thickness_label.config(text=f"Thickness: {self.hough_thickness_var.get()}")

    def apply_blur(self):
        if self.original_image is not None:
            self.processing_pipeline["blur"] = {"type": self.blur_type_var.get(), "level": self.blur_level_var.get()}
            self.apply_processing_pipeline()
           
            print(self.processing_pipeline)
    
    def view_blur(self):
        if self.original_image is not None:
            temp_image = self.temp_image.copy()
            blur_type = self.blur_type_var.get()
            blur_level = self.blur_level_var.get()
            if blur_type == "Gaussian":
                temp_image = cv2.GaussianBlur(temp_image, (blur_level * 2 + 1, blur_level * 2 + 1), 0)
            elif blur_type == "Median":
                temp_image = cv2.medianBlur(temp_image, blur_level * 2 + 1)
            elif blur_type == "Average":
                temp_image = cv2.blur(temp_image, (blur_level * 2 + 1, blur_level * 2 + 1))
            elif blur_type == "fastNlMeansDenoisingColored":
                temp_image = cv2.fastNlMeansDenoisingColored(temp_image, None, blur_level, blur_level, 7, 21)
            
            self.update_image_display(temp_image)

    def apply_contour(self):
         if self.original_image is not None:
            self.processing_pipeline["contour"] = {"type": self.contour_type_var.get(), "level": self.contour_level_var.get()}
            self.apply_processing_pipeline()

    def view_contour(self):
        if self.original_image is not None:
            temp_image = self.temp_image.copy()
            contour_type = self.contour_type_var.get()
            contour_level = self.contour_level_var.get()
            
            # 이미지 채널 수 확인하여 RGB 이미지인지 판별
            if len(temp_image.shape) == 3 and temp_image.shape[2] == 3:  # RGB 이미지인 경우
                temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
            
            if contour_type == "Canny":
                temp_image = cv2.Canny(temp_image, contour_level, contour_level * 2)
            elif contour_type == "Laplacian":
                temp_image = cv2.Laplacian(temp_image, cv2.CV_64F, ksize=3)
                temp_image = np.uint8(np.absolute(temp_image))
            
            self.update_image_display(temp_image)

    def apply_contour_detection(self):
         if self.original_image is not None:
            self.processing_pipeline["contour_detection"] = {"level": self.contour_detect_level_var.get()}
            self.apply_processing_pipeline()

    def view_contour_detection(self):
        if self.original_image is not None:
            temp_image = self.temp_image.copy()
            if len(temp_image.shape) == 3 and temp_image.shape[2] == 3:  # RGB 이미지인 경우
                gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = temp_image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.contour_detect_level_var.get(), self.contour_detect_level_var.get() * 2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            temp_image = temp_image.copy()
            cv2.drawContours(temp_image, contours, -1, (0, 255, 0), 2)
            self.update_image_display(temp_image)

    def apply_threshold(self):
         if self.original_image is not None:
            self.processing_pipeline["threshold"] = {"type": self.threshold_type_var.get(), "level": self.threshold_level_var.get()}
            self.apply_processing_pipeline()

    def view_threshold(self):
        if self.original_image is not None:
            temp_image = self.temp_image.copy()
            if len(temp_image.shape) == 3 and temp_image.shape[2] == 3:  # RGB 이미지인 경우
                gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = temp_image
            threshold_type = self.threshold_type_var.get()
            threshold_level = self.threshold_level_var.get()
            if threshold_type == "Binary":
                _, temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY)
            elif threshold_type == "Binary Inverse":
                _, temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY_INV)
            elif threshold_type == "Trunc":
                 _, temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_TRUNC)
            elif threshold_type == "Tozero":
                 _, temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_TOZERO)
            elif threshold_type == "Tozero Inverse":
                _, temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_TOZERO_INV)
            
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2RGB)
            self.update_image_display(temp_image)
    
    def apply_histogram(self):
        pass
    
    def view_histogram(self):
        if self.original_image is not None:
            temp_image = self.temp_image.copy()
            if len(temp_image.shape) == 3 and temp_image.shape[2] == 3:  # RGB 이미지인 경우
                gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = temp_image
            if self.histogram_equalization_var.get():
                gray = cv2.equalizeHist(gray)
                temp_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                self.update_image_display(temp_image)
                
            else:
                plt.figure(figsize=(6, 4))
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                plt.plot(hist)
                plt.title('Histogram')
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
                canvas = FigureCanvasTkAgg(plt.gcf(), master=self.master)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.place(x=0, y=220)
                plt.show(block = False)

    def apply_dilation_erosion(self):
         if self.original_image is not None:
            self.processing_pipeline["dilation_erosion"] = {"type": self.dilation_erosion_type_var.get(), "level": self.dilation_erosion_level_var.get()}
            self.apply_processing_pipeline()
    
    def view_dilation_erosion(self):
        if self.original_image is not None:
            temp_image = self.temp_image.copy()
            dilation_erosion_type = self.dilation_erosion_type_var.get()
            dilation_erosion_level = self.dilation_erosion_level_var.get()
            kernel = np.ones((dilation_erosion_level, dilation_erosion_level), np.uint8)
            if dilation_erosion_type == "Dilation":
                temp_image = cv2.dilate(temp_image, kernel, iterations=1)
            elif dilation_erosion_type == "Erosion":
                temp_image = cv2.erode(temp_image, kernel, iterations=1)
            
            self.update_image_display(temp_image)

    def apply_hough_line(self):
        if self.original_image is not None:
            self.processing_pipeline["hough_line"] = {"angle": self.hough_angle_var.get(), "thickness": self.hough_thickness_var.get()}
            self.apply_processing_pipeline()
    def view_hough_line(self):
        if self.original_image is not None:
            temp_image = self.temp_image.copy()
            if len(temp_image.shape) == 3 and temp_image.shape[2] == 3:  # RGB 이미지인 경우
                gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = temp_image
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    
                    angle_threshold = abs(self.hough_angle_var.get() - int(np.degrees(theta)))
                    
                    if angle_threshold < 10:
                        cv2.line(temp_image, (x1, y1), (x2, y2), (0, 255, 0), self.hough_thickness_var.get())
                    
            self.update_image_display(temp_image)
            
    def remove_blur(self):
        if "blur" in self.processing_pipeline:
            del self.processing_pipeline["blur"]  # 딕셔너리에서 blur 항목 제거
            self.apply_processing_pipeline() # 파이프라인 다시 적용

    def remove_contour(self):
        if "contour" in self.processing_pipeline:
            del self.processing_pipeline["contour"]
            self.apply_processing_pipeline()

    def remove_contour_detection(self):
        if "contour_detection" in self.processing_pipeline:
            del self.processing_pipeline["contour_detection"]
            self.apply_processing_pipeline()

    def remove_threshold(self):
        if "threshold" in self.processing_pipeline:
            del self.processing_pipeline["threshold"]
            self.apply_processing_pipeline()

    def remove_dilation_erosion(self):
        if "dilation_erosion" in self.processing_pipeline:
            del self.processing_pipeline["dilation_erosion"]
            self.apply_processing_pipeline()

    def remove_hough_line(self):
        if "hough_line" in self.processing_pipeline:
            del self.processing_pipeline["hough_line"]
            self.apply_processing_pipeline()
    
    def apply_processing_pipeline(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.temp_image = self.original_image.copy()

            if "blur" in self.processing_pipeline:
                blur_params = self.processing_pipeline["blur"]
                blur_type = blur_params["type"]
                blur_level = blur_params["level"]
                if blur_type == "Gaussian":
                    self.processed_image = cv2.GaussianBlur(self.processed_image, (blur_level * 2 + 1, blur_level * 2 + 1), 0)
                    self.temp_image = cv2.GaussianBlur(self.temp_image, (blur_level * 2 + 1, blur_level * 2 + 1), 0)
                elif blur_type == "Median":
                    self.processed_image = cv2.medianBlur(self.processed_image, blur_level * 2 + 1)
                    self.temp_image = cv2.medianBlur(self.temp_image, blur_level * 2 + 1)
                elif blur_type == "Average":
                    self.processed_image = cv2.blur(self.processed_image, (blur_level * 2 + 1, blur_level * 2 + 1))
                    self.temp_image = cv2.blur(self.temp_image, (blur_level * 2 + 1, blur_level * 2 + 1))
                elif blur_type == "fastNlMeansDenoisingColored":
                    self.processed_image = cv2.fastNlMeansDenoisingColored(self.processed_image, None, blur_level, blur_level, 7, 21)
                    self.temp_image = cv2.fastNlMeansDenoisingColored(self.temp_image, None, blur_level, blur_level, 7, 21)
            
            if "contour" in self.processing_pipeline:
                contour_params = self.processing_pipeline["contour"]
                contour_type = contour_params["type"]
                contour_level = contour_params["level"]
                
                if len(self.processed_image.shape) == 3 and self.processed_image.shape[2] == 3:  # RGB 이미지인 경우
                    self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                if len(self.temp_image.shape) == 3 and self.temp_image.shape[2] == 3:  # RGB 이미지인 경우
                    self.temp_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2GRAY)
                
                if contour_type == "Canny":
                        self.processed_image = cv2.Canny(self.processed_image, contour_level, contour_level * 2)
                        self.temp_image = cv2.Canny(self.temp_image, contour_level, contour_level * 2)
                elif contour_type == "Laplacian":
                        self.processed_image = cv2.Laplacian(self.processed_image, cv2.CV_64F, ksize=3)
                        self.processed_image = np.uint8(np.absolute(self.processed_image))
                        self.temp_image = cv2.Laplacian(self.temp_image, cv2.CV_64F, ksize=3)
                        self.temp_image = np.uint8(np.absolute(self.temp_image))
            
            if "contour_detection" in self.processing_pipeline:
                params = self.processing_pipeline["contour_detection"]
                contour_detect_level = params["level"]
                if len(self.processed_image.shape) == 3 and self.processed_image.shape[2] == 3:  # RGB 이미지인 경우
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.processed_image
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, contour_detect_level, contour_detect_level * 2)


                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(self.processed_image, contours, -1, (0, 255, 0), 2)

                if len(self.temp_image.shape) == 3 and self.temp_image.shape[2] == 3:  # RGB 이미지인 경우
                    gray = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.temp_image
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, contour_detect_level, contour_detect_level * 2)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(self.temp_image, contours, -1, (0, 255, 0), 2)
            if "threshold" in self.processing_pipeline:
                params = self.processing_pipeline["threshold"]
                threshold_type = params["type"]
                threshold_level = params["level"]
                if len(self.processed_image.shape) == 3 and self.processed_image.shape[2] == 3:  # RGB 이미지인 경우
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.processed_image
                if threshold_type == "Binary":
                    _, self.processed_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY)
                elif threshold_type == "Binary Inverse":
                    _, self.processed_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY_INV)
                elif threshold_type == "Trunc":
                    _, self.processed_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_TRUNC)
                elif threshold_type == "Tozero":
                    _, self.processed_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_TOZERO)
                elif threshold_type == "Tozero Inverse":
                    _, self.processed_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_TOZERO_INV)
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)

                if len(self.temp_image.shape) == 3 and self.temp_image.shape[2] == 3:  # RGB 이미지인 경우
                    gray = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.temp_image
                if threshold_type == "Binary":
                    _, self.temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY)
                elif threshold_type == "Binary Inverse":
                    _, self.temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY_INV)
                elif threshold_type == "Trunc":
                    _, self.temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_TRUNC)
                elif threshold_type == "Tozero":
                    _, self.temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_TOZERO)
                elif threshold_type == "Tozero Inverse":
                    _, self.temp_image = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_TOZERO_INV)
                self.temp_image = cv2.cvtColor(self.temp_image, cv2.COLOR_GRAY2RGB)
            
            if "dilation_erosion" in self.processing_pipeline:
                params = self.processing_pipeline["dilation_erosion"]
                dilation_erosion_type = params["type"]
                dilation_erosion_level = params["level"]
                kernel = np.ones((dilation_erosion_level, dilation_erosion_level), np.uint8)
                if dilation_erosion_type == "Dilation":
                    self.processed_image = cv2.dilate(self.processed_image, kernel, iterations=1)
                    self.temp_image = cv2.dilate(self.temp_image, kernel, iterations=1)
                elif dilation_erosion_type == "Erosion":
                    self.processed_image = cv2.erode(self.processed_image, kernel, iterations=1)
                    self.temp_image = cv2.erode(self.temp_image, kernel, iterations=1)
            
            if "hough_line" in self.processing_pipeline:
                params = self.processing_pipeline["hough_line"]
                angle = params["angle"]
                thickness = params["thickness"]
                if len(self.processed_image.shape) == 3 and self.processed_image.shape[2] == 3:  # RGB 이미지인 경우
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.processed_image
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
                if lines is not None:
                    for line in lines:
                        rho, theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        
                        angle_threshold = abs(angle - int(np.degrees(theta)))
                        if angle_threshold < 10:
                            cv2.line(self.processed_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)

                if len(self.temp_image.shape) == 3 and self.temp_image.shape[2] == 3:  # RGB 이미지인 경우
                    gray = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.temp_image
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
                if lines is not None:
                    for line in lines:
                        rho, theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        angle_threshold = abs(angle - int(np.degrees(theta)))
                        
                        if angle_threshold < 10:
                            cv2.line(self.temp_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            self.update_image_display(self.processed_image)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()