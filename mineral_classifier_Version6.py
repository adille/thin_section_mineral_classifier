import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage import measure, filters
from scipy import ndimage
import datetime
import json
import csv
from scipy import stats
import tifffile

class MineralClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Mineral Thin Section Classifier")
        self.root.geometry("1400x800")  # Wider to accommodate the third panel
        
        self.current_image_path = None
        self.images_paths = []
        self.current_image_index = 0
        self.selected_pixels = []
        self.mineral_colors = {}  # Dictionary to store mineral colors
        self.current_image = None
        self.current_image_array = None
        self.current_display_image = None  # For zoomed image display
        self.zoom_level = 1.0  # Initial zoom level
        self.output_folder = None  # For saving classification results
        
        # Create main frames
        self.left_frame = tk.Frame(self.root, width=550, height=800)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.center_frame = tk.Frame(self.root, width=550, height=800)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = tk.Frame(self.root, width=300, height=800)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Left frame components - Image display and navigation
        self.btn_select_folder = tk.Button(self.left_frame, text="Select Folder", command=self.select_folder)
        self.btn_select_folder.pack(pady=10)
        
        # Zoom controls
        self.zoom_frame = tk.Frame(self.left_frame)
        self.zoom_frame.pack(pady=5)
        
        self.zoom_out_btn = tk.Button(self.zoom_frame, text="🔍-", command=self.zoom_out)
        self.zoom_out_btn.grid(row=0, column=0, padx=5)
        
        self.zoom_reset_btn = tk.Button(self.zoom_frame, text="Reset Zoom", command=self.reset_zoom)
        self.zoom_reset_btn.grid(row=0, column=1, padx=5)
        
        self.zoom_in_btn = tk.Button(self.zoom_frame, text="🔍+", command=self.zoom_in)
        self.zoom_in_btn.grid(row=0, column=2, padx=5)
        
        self.zoom_label = tk.Label(self.zoom_frame, text="Zoom: 100%")
        self.zoom_label.grid(row=0, column=3, padx=5)
        
        # Image canvas with scrollbars
        self.image_frame = tk.Frame(self.left_frame, width=530, height=580)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.v_scrollbar = ttk.Scrollbar(self.image_frame)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(self.image_frame, width=530, height=580,
                               xscrollcommand=self.h_scrollbar.set,
                               yscrollcommand=self.v_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        # Add mouse wheel binding for zoom
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # For Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # For Linux, scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # For Linux, scroll down
        # Add panning with middle mouse button
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_image)
        
        # Navigation buttons
        self.navigation_frame = tk.Frame(self.left_frame)
        self.navigation_frame.pack(pady=10)
        
        self.btn_prev = tk.Button(self.navigation_frame, text="Previous", command=self.previous_image)
        self.btn_prev.grid(row=0, column=0, padx=5)
        
        self.label_image_counter = tk.Label(self.navigation_frame, text="0/0")
        self.label_image_counter.grid(row=0, column=1, padx=5)
        
        self.btn_next = tk.Button(self.navigation_frame, text="Next", command=self.next_image)
        self.btn_next.grid(row=0, column=2, padx=5)
        
        # Progress bar
        self.progress_frame = tk.Frame(self.left_frame)
        self.progress_frame.pack(pady=10, fill=tk.X, padx=10)
        
        self.progress_label = tk.Label(self.progress_frame, text="Progress:")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, 
                                            length=400, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Center frame components - Controls and results
        self.control_frame = tk.Frame(self.center_frame)
        self.control_frame.pack(pady=10, fill=tk.X)
        
        self.selected_pixels_label = tk.Label(self.control_frame, text="Selected Pixels:")
        self.selected_pixels_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.selected_pixels_listbox = tk.Listbox(self.control_frame, width=50, height=5)
        self.selected_pixels_listbox.grid(row=1, column=0, padx=5, pady=5, columnspan=2)
        
        self.label_name_entry_label = tk.Label(self.control_frame, text="Mineral Name:")
        self.label_name_entry_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.mineral_name_entry = tk.Entry(self.control_frame, width=20)
        self.mineral_name_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.add_mineral_btn = tk.Button(self.control_frame, text="Add Mineral", command=self.add_mineral)
        self.add_mineral_btn.grid(row=3, column=0, padx=5, pady=5)
        
        self.clear_selections_btn = tk.Button(self.control_frame, text="Clear Selections", command=self.clear_selections)
        self.clear_selections_btn.grid(row=3, column=1, padx=5, pady=5)
        
        # Mineral selection save/load
        self.save_load_frame = tk.Frame(self.control_frame)
        self.save_load_frame.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        
        self.save_selections_btn = tk.Button(self.save_load_frame, text="Save Selections", command=self.save_mineral_selections)
        self.save_selections_btn.grid(row=0, column=0, padx=5)
        
        self.load_selections_btn = tk.Button(self.save_load_frame, text="Load Selections", command=self.load_mineral_selections)
        self.load_selections_btn.grid(row=0, column=1, padx=5)
        
        # Classification model selection
        self.model_frame = tk.LabelFrame(self.center_frame, text="Classification Model")
        self.model_frame.pack(pady=10, fill=tk.X, padx=10)
        
        self.model_var = tk.StringVar(value="knn")
        
        self.knn_radio = tk.Radiobutton(self.model_frame, text="K-Nearest Neighbors", 
                                         variable=self.model_var, value="knn")
        self.knn_radio.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.svm_radio = tk.Radiobutton(self.model_frame, text="Support Vector Machine", 
                                         variable=self.model_var, value="svm")
        self.svm_radio.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.rf_radio = tk.Radiobutton(self.model_frame, text="Random Forest", 
                                        variable=self.model_var, value="rf")
        self.rf_radio.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.kmeans_radio = tk.Radiobutton(self.model_frame, text="K-Means", 
                                           variable=self.model_var, value="kmeans")
        self.kmeans_radio.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Classification button and reset button
        self.buttons_frame = tk.Frame(self.center_frame)
        self.buttons_frame.pack(pady=10)
        
        self.classify_btn = tk.Button(self.buttons_frame, text="Classify Image", command=self.classify_image)
        self.classify_btn.grid(row=0, column=0, padx=5)
        
        self.reset_results_btn = tk.Button(self.buttons_frame, text="Reset Results", command=self.reset_results)
        self.reset_results_btn.grid(row=0, column=1, padx=5)
        
        # Save results checkbox
        self.save_results_var = tk.BooleanVar(value=True)
        self.save_results_check = tk.Checkbutton(self.center_frame, text="Save Results", 
                                               variable=self.save_results_var)
        self.save_results_check.pack(pady=5)
        
        # Minerals frame
        self.minerals_frame = tk.LabelFrame(self.center_frame, text="Identified Minerals")
        self.minerals_frame.pack(pady=10, fill=tk.X, padx=10)
        
        self.minerals_listbox = tk.Listbox(self.minerals_frame, width=50, height=5)
        self.minerals_listbox.pack(pady=5, padx=5, fill=tk.X)
        
        # Results frame for displaying classification results
        self.results_frame = tk.LabelFrame(self.center_frame, text="Classification Results")
        self.results_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=10)
        
        # Right frame - Help and parameter descriptions
        self.help_frame = tk.LabelFrame(self.right_frame, text="Parameter Descriptions")
        self.help_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=10)
        
        # Create a scrollable text widget for parameter descriptions
        self.help_canvas = tk.Canvas(self.help_frame)
        self.help_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.help_scrollbar = ttk.Scrollbar(self.help_frame, orient="vertical", command=self.help_canvas.yview)
        self.help_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.help_canvas.configure(yscrollcommand=self.help_scrollbar.set)
        self.help_canvas.bind('<Configure>', lambda e: self.help_canvas.configure(scrollregion=self.help_canvas.bbox("all")))
        
        self.help_content_frame = tk.Frame(self.help_canvas)
        self.help_canvas.create_window((0, 0), window=self.help_content_frame, anchor="nw")
        
        # Add parameter descriptions
        self.add_parameter_descriptions()
        
        # Carbon and Other settings frames at the bottom of right panel
        self.settings_frame = tk.Frame(self.right_frame)
        self.settings_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # Carbon detection settings
        self.carbon_frame = tk.LabelFrame(self.settings_frame, text="Carbon Detection Settings")
        self.carbon_frame.pack(pady=5, fill=tk.X)
        
        self.carbon_threshold_label = tk.Label(self.carbon_frame, text="Carbon Threshold:")
        self.carbon_threshold_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.carbon_threshold_var = tk.IntVar(value=30)  # Default threshold value
        self.carbon_threshold_scale = tk.Scale(self.carbon_frame, variable=self.carbon_threshold_var,
                                              from_=0, to=100, orient=tk.HORIZONTAL, length=150)
        self.carbon_threshold_scale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.carbon_blob_size_label = tk.Label(self.carbon_frame, text="Min Blob Size:")
        self.carbon_blob_size_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.carbon_blob_size_var = tk.IntVar(value=100)  # Default blob size
        self.carbon_blob_size_scale = tk.Scale(self.carbon_frame, variable=self.carbon_blob_size_var,
                                              from_=10, to=1000, orient=tk.HORIZONTAL, length=150)
        self.carbon_blob_size_scale.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Other category threshold
        self.other_frame = tk.LabelFrame(self.settings_frame, text="Other Category Settings")
        self.other_frame.pack(pady=5, fill=tk.X)
        
        self.other_threshold_label = tk.Label(self.other_frame, text="Distance Threshold:")
        self.other_threshold_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.other_threshold_var = tk.DoubleVar(value=50.0)  # Default threshold value
        self.other_threshold_scale = tk.Scale(self.other_frame, variable=self.other_threshold_var,
                                            from_=10.0, to=200.0, resolution=5.0, orient=tk.HORIZONTAL, length=150)
        self.other_threshold_scale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Variables for panning
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.original_image = None  # Store original PIL image
        
        # Initialize classifier
        self.classifier = None
        self.scaler = None

    def add_parameter_descriptions(self):
        """Add descriptive text for all parameters to the help panel"""
        # Title
        title_label = tk.Label(self.help_content_frame, text="Parameter Guide", font=("Arial", 12, "bold"))
        title_label.pack(pady=5, anchor="w")
        
        # General Usage
        usage_frame = tk.LabelFrame(self.help_content_frame, text="General Usage")
        usage_frame.pack(pady=5, fill=tk.X, padx=5)
        
        usage_text = (
            "1. Select a folder with mineral images\n"
            "2. Click on pixels to select minerals\n"
            "3. Name and add minerals\n"
            "4. Choose classification model\n"
            "5. Adjust thresholds if needed\n"
            "6. Click 'Classify Image'\n"
            "7. Click 'Reset Results' when finished"
        )
        
        usage_label = tk.Label(usage_frame, text=usage_text, justify=tk.LEFT)
        usage_label.pack(pady=5, anchor="w")
        
        # Models section
        models_frame = tk.LabelFrame(self.help_content_frame, text="Classification Models")
        models_frame.pack(pady=5, fill=tk.X, padx=5)
        
        models_text = (
            "KNN (K-Nearest Neighbors):\n"
            "- Best general-purpose option\n"
            "- Good with multiple samples\n\n"
            "SVM (Support Vector Machine):\n"
            "- Best for similar colors with\n  distinct boundaries\n\n"
            "Random Forest:\n"
            "- Best for varied textures\n"
            "- Handles subtle patterns well\n\n"
            "K-Means:\n"
            "- Simple unsupervised clustering\n"
            "- Quick exploratory analysis"
        )
        
        models_label = tk.Label(models_frame, text=models_text, justify=tk.LEFT)
        models_label.pack(pady=5, anchor="w")
        
        # Carbon Threshold
        carbon_frame = tk.LabelFrame(self.help_content_frame, text="Carbon Threshold")
        carbon_frame.pack(pady=5, fill=tk.X, padx=5)
        
        carbon_text = (
            "Controls how dark a pixel must be\n"
            "to be considered carbon (graphite).\n\n"
            "- Low values (10-20): Only very dark\n  pixels identified as carbon\n"
            "- High values (40-50): More medium\n  darkness pixels included\n"
            "- Works with grayscale values (0-255)\n"
            "- Default: 30"
        )
        
        carbon_label = tk.Label(carbon_frame, text=carbon_text, justify=tk.LEFT)
        carbon_label.pack(pady=5, anchor="w")
        
        # Blob Size
        blob_frame = tk.LabelFrame(self.help_content_frame, text="Min Blob Size")
        blob_frame.pack(pady=5, fill=tk.X, padx=5)
        
        blob_text = (
            "Controls the maximum size of black\n"
            "areas that can be carbon.\n\n"
            "- Small values: More black areas\n  classified as carbon\n"
            "- Large values: Only small diffuse\n  black areas are carbon\n"
            "- Size is in pixels\n"
            "- Default: 100 pixels"
        )
        
        blob_label = tk.Label(blob_frame, text=blob_text, justify=tk.LEFT)
        blob_label.pack(pady=5, anchor="w")
        
        # Distance Threshold
        distance_frame = tk.LabelFrame(self.help_content_frame, text="Distance Threshold")
        distance_frame.pack(pady=5, fill=tk.X, padx=5)
        
        distance_text = (
            "Controls how strictly pixels must\n"
            "match known minerals.\n\n"
            "- Low values (10-30): Strict matching,\n  more pixels classified as 'Other'\n"
            "- High values (100+): Lenient matching,\n  more pixels assigned to minerals\n"
            "- For KNN: Euclidean distance in RGB space\n"
            "- For other models: Converted threshold\n"
            "- Default: 50"
        )
        
        distance_label = tk.Label(distance_frame, text=distance_text, justify=tk.LEFT)
        distance_label.pack(pady=5, anchor="w")
        
        # Tips
        tips_frame = tk.LabelFrame(self.help_content_frame, text="Tips")
        tips_frame.pack(pady=5, fill=tk.X, padx=5)
        
        tips_text = (
            "• Select multiple samples for each mineral\n"
            "• Zoom in for accurate pixel selection\n"
            "• Save selections for consistent analysis\n"
            "• Try different models for best results\n"
            "• Adjust thresholds incrementally\n"
            "• Check the confidence intervals\n"
            "• Use 'Other' category for unknown minerals"
        )
        
        tips_label = tk.Label(tips_frame, text=tips_text, justify=tk.LEFT)
        tips_label.pack(pady=5, anchor="w")

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Mineral Images")
        if not folder_path:
            return
            
        # Get all image files from the selected folder
        self.images_paths = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                self.images_paths.append(os.path.join(folder_path, file))
                
        if not self.images_paths:
            messagebox.showinfo("No Images", "No image files found in the selected folder.")
            return
            
        # Create output folder for classification results
        self.output_folder = os.path.join(folder_path, "mineral_classification_results")
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Reset variables
        self.current_image_index = 0
        self.selected_pixels = []
        self.mineral_colors = {}
        self.zoom_level = 1.0
        self.update_selected_pixels_display()
        self.update_minerals_display()
        self.update_zoom_label()
        
        # Reset results
        self.reset_results()
        
        # Display the first image
        self.display_current_image()

    def display_current_image(self):
        if not self.images_paths:
            return
            
        self.current_image_path = self.images_paths[self.current_image_index]
        
        try:
            # Open and store the image
            self.original_image = Image.open(self.current_image_path)
            self.current_image_array = np.array(self.original_image)
            
            # Apply the current zoom level
            self.apply_zoom()
            
            # Update image counter
            self.label_image_counter.config(text=f"{self.current_image_index + 1}/{len(self.images_paths)}")
            
            # Reset selected pixels for the new image
            self.selected_pixels = []
            self.update_selected_pixels_display()
            
            # Look for saved mineral selections for this image
            if self.output_folder:
                base_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
                selections_file = os.path.join(self.output_folder, f"{base_filename}_selections.json")
                if os.path.exists(selections_file):
                    self.load_mineral_selections(selections_file)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def apply_zoom(self):
        if self.original_image is None:
            return
            
        # Get original size
        orig_width, orig_height = self.original_image.size
        
        # Calculate new size based on zoom level
        new_width = int(orig_width * self.zoom_level)
        new_height = int(orig_height * self.zoom_level)
        
        # Resize the image
        if self.zoom_level == 1.0:
            # Use original image at 100% zoom for best quality
            resized_img = self.original_image.copy()
        else:
            # Resize according to zoom level
            resized_img = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Update the PhotoImage and display
        self.current_display_image = ImageTk.PhotoImage(resized_img)
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Clear previous image and markers
        self.canvas.delete("all")
        
        # Display the new image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_display_image)
        
        # Redraw markers for selected pixels
        self.redraw_markers()

    def redraw_markers(self):
        if not self.selected_pixels or self.original_image is None:
            return
            
        # For each selected pixel, calculate its position in the zoomed image
        # and draw a marker
        marker_radius = 5
        for x, y, _ in self.selected_pixels:
            # Calculate the position in the zoomed image
            zoomed_x = int(x * self.zoom_level)
            zoomed_y = int(y * self.zoom_level)
            
            self.canvas.create_oval(
                zoomed_x - marker_radius, zoomed_y - marker_radius,
                zoomed_x + marker_radius, zoomed_y + marker_radius,
                outline="yellow", width=2
            )

    def zoom_in(self):
        if self.original_image is None:
            return
            
        # Increase zoom level
        self.zoom_level *= 1.25
        if self.zoom_level > 10.0:  # Set a maximum zoom level
            self.zoom_level = 10.0
            
        self.update_zoom_label()
        self.apply_zoom()

    def zoom_out(self):
        if self.original_image is None:
            return
            
        # Decrease zoom level
        self.zoom_level /= 1.25
        if self.zoom_level < 0.25:  # Set a minimum zoom level
            self.zoom_level = 0.25
            
        self.update_zoom_label()
        self.apply_zoom()

    def reset_zoom(self):
        if self.original_image is None:
            return
            
        # Reset to 100%
        self.zoom_level = 1.0
        self.update_zoom_label()
        self.apply_zoom()

    def update_zoom_label(self):
        zoom_percentage = int(self.zoom_level * 100)
        self.zoom_label.config(text=f"Zoom: {zoom_percentage}%")

    def on_mousewheel(self, event):
        if self.original_image is None:
            return
            
        # Handle mousewheel zoom
        # For Windows
        if event.num == 5 or event.delta < 0:
            self.zoom_out()
        if event.num == 4 or event.delta > 0:
            self.zoom_in()

    def start_pan(self, event):
        # Start panning with middle mouse button
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan_image(self, event):
        # Calculate how much to move
        dx = self.pan_start_x - event.x
        dy = self.pan_start_y - event.y
        
        # Move canvas
        self.canvas.xview_scroll(dx, "units")
        self.canvas.yview_scroll(dy, "units")
        
        # Reset start position
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def next_image(self):
        if not self.images_paths:
            return
            
        self.current_image_index = (self.current_image_index + 1) % len(self.images_paths)
        self.display_current_image()

    def previous_image(self):
        if not self.images_paths:
            return
            
        self.current_image_index = (self.current_image_index - 1) % len(self.images_paths)
        self.display_current_image()

    def on_canvas_click(self, event):
        if self.original_image is None:
            return
            
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert canvas coordinates to original image coordinates
        x_orig = int(canvas_x / self.zoom_level)
        y_orig = int(canvas_y / self.zoom_level)
        
        # Make sure coordinates are within image bounds
        orig_width, orig_height = self.original_image.size
        x_orig = max(0, min(x_orig, orig_width - 1))
        y_orig = max(0, min(y_orig, orig_height - 1))
        
        # Get RGB value at that point
        pixel_color = self.current_image_array[y_orig, x_orig]
        
        # Add to selected pixels
        self.selected_pixels.append((x_orig, y_orig, pixel_color))
        
        # Update display
        self.update_selected_pixels_display()
        
        # Show a marker on the clicked position
        marker_radius = 5
        zoomed_x = int(x_orig * self.zoom_level)
        zoomed_y = int(y_orig * self.zoom_level)
        
        self.canvas.create_oval(
            zoomed_x - marker_radius, zoomed_y - marker_radius,
            zoomed_x + marker_radius, zoomed_y + marker_radius,
            outline="yellow", width=2
        )

    def update_selected_pixels_display(self):
        self.selected_pixels_listbox.delete(0, tk.END)
        for i, (x, y, color) in enumerate(self.selected_pixels):
            self.selected_pixels_listbox.insert(tk.END, f"{i+1}: Position ({x},{y}), RGB: {color}")

    def add_mineral(self):
        if not self.selected_pixels:
            messagebox.showinfo("No Selection", "Please select at least one pixel first.")
            return
            
        mineral_name = self.mineral_name_entry.get().strip()
        if not mineral_name:
            messagebox.showinfo("Missing Name", "Please enter a mineral name.")
            return
            
        # Get colors from selected pixels
        colors = np.array([color for _, _, color in self.selected_pixels])
        avg_color = np.mean(colors, axis=0).astype(int)
        
        # Add to mineral colors dictionary
        self.mineral_colors[mineral_name] = {
            'color': avg_color,
            'samples': [(x, y, color.tolist()) for x, y, color in self.selected_pixels]
        }
        
        # Update the minerals display
        self.update_minerals_display()
        
        # Clear selection
        self.selected_pixels = []
        self.update_selected_pixels_display()
        self.mineral_name_entry.delete(0, tk.END)
        
        # Redisplay the image (to clear markers)
        self.apply_zoom()

    def update_minerals_display(self):
        self.minerals_listbox.delete(0, tk.END)
        for name, data in self.mineral_colors.items():
            color = data['color']
            samples_count = len(data['samples'])
            self.minerals_listbox.insert(tk.END, f"{name}: RGB: {color} (Samples: {samples_count})")

    def clear_selections(self):
        self.selected_pixels = []
        self.update_selected_pixels_display()
        self.apply_zoom()  # Redisplay to clear markers

    def reset_results(self):
        """Clear classification results and reset the results frame"""
        # Clear results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # Reset progress bar
        self.progress_bar["value"] = 0
        self.root.update_idletasks()
        
        # Add a message to the results frame
        message_label = tk.Label(self.results_frame, 
                                text="No classification results yet.\nClick 'Classify Image' to analyze the current image.",
                                justify=tk.CENTER)
        message_label.pack(expand=True, pady=20)

    def save_mineral_selections(self):
        """Save the current mineral selections to a JSON file"""
        if not self.mineral_colors or not self.output_folder:
            messagebox.showinfo("No Data", "No mineral selections to save or no output folder selected.")
            return
            
        try:
            # Create a dictionary of mineral selections
            selections_data = {
                'image_path': self.current_image_path,
                'minerals': {}
            }
            
            for name, data in self.mineral_colors.items():
                selections_data['minerals'][name] = {
                    'color': data['color'].tolist(),
                    'samples': data['samples']
                }
                
            # Generate filename based on the current image
            base_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
            output_file = os.path.join(self.output_folder, f"{base_filename}_selections.json")
            
            # Save to JSON file
            with open(output_file, 'w') as f:
                json.dump(selections_data, f, indent=2)
                
            messagebox.showinfo("Save Successful", f"Mineral selections saved to:\n{output_file}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save selections: {str(e)}")

    def load_mineral_selections(self, file_path=None):
        """Load mineral selections from a JSON file"""
        try:
            # If no file path is provided, open a file dialog
            if file_path is None or not isinstance(file_path, str):
                file_path = filedialog.askopenfilename(
                    title="Load Mineral Selections",
                    filetypes=[("JSON Files", "*.json")],
                    initialdir=self.output_folder if self.output_folder else "."
                )
                
            if not file_path:
                return
                
            # Load from JSON file
            with open(file_path, 'r') as f:
                selections_data = json.load(f)
                
            # Clear current selections
            self.mineral_colors = {}
            
            # Load minerals
            for name, data in selections_data['minerals'].items():
                samples = [(s[0], s[1], np.array(s[2])) for s in data['samples']]
                
                self.mineral_colors[name] = {
                    'color': np.array(data['color']),
                    'samples': samples
                }
                
            # Update the minerals display
            self.update_minerals_display()
            
            messagebox.showinfo("Load Successful", "Mineral selections loaded successfully.")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load selections: {str(e)}")

    def train_classifier(self):
        """Train a classifier using the selected mineral samples based on the selected algorithm"""
        # Collect all samples and their labels
        X_samples = []
        y_labels = []
        
        for idx, (name, data) in enumerate(self.mineral_colors.items()):
            for x, y, color in data['samples']:
                X_samples.append(color)
                y_labels.append(idx)
        
        # Convert to numpy arrays
        X = np.array(X_samples)
        y = np.array(y_labels)
        
        # Normalize features for better performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and train the classifier based on selection
        model_type = self.model_var.get()
        
        if model_type == "knn":
            # K-Nearest Neighbors
            classifier = KNeighborsClassifier(n_neighbors=min(3, len(X)))
        elif model_type == "svm":
            # Support Vector Machine
            classifier = SVC(probability=True)
        elif model_type == "rf":
            # Random Forest
            classifier = RandomForestClassifier(n_estimators=100)
        elif model_type == "kmeans":
            # K-Means
            classifier = KMeans(n_clusters=len(self.mineral_colors))
        else:
            # Default to KNN
            classifier = KNeighborsClassifier(n_neighbors=min(3, len(X)))
        
        # Train the classifier
        classifier.fit(X_scaled, y)
        
        # Save the classifier and scaler
        self.classifier = classifier
        self.scaler = scaler
        
        return X_scaled, y, classifier, scaler

    def classify_image(self):
        if not self.mineral_colors:
            messagebox.showinfo("No Minerals", "Please define at least one mineral first.")
            return
        
        if self.current_image_array is None:
            messagebox.showinfo("No Image", "No image is currently loaded.")
            return
        
        # Clear previous results
        self.reset_results()
        
        # Reset progress bar
        self.progress_bar["value"] = 0
        self.root.update_idletasks()
        
        # Train the classifier using the selected mineral samples
        X_scaled, y, classifier, scaler = self.train_classifier()
        
        # Reshape the image array for processing
        h, w, d = self.current_image_array.shape
        pixels = self.current_image_array.reshape((h * w, d))
        
        # Create a mask for the carbon (special handling)
        carbon_mask = self.detect_carbon(self.current_image_array)
        
        # Scale the pixels for the classifier
        pixels_scaled = scaler.transform(pixels)
        
        # Create a classification result array and a distance/probability array
        result = np.zeros(h * w, dtype=np.int32)
        confidence = np.zeros(h * w, dtype=np.float32)
        
        # Process in batches to update progress bar
        batch_size = 10000
        num_batches = max(1, (h * w) // batch_size)
        
        # Get the distance threshold for "Other" category
        other_threshold = self.other_threshold_var.get()
        
        # Classification approach depends on the model
        model_type = self.model_var.get()
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, h * w)
            
            # Skip pixels that are already classified as carbon
            if np.any(carbon_mask.reshape(h * w)[start_idx:end_idx]):
                carbon_indices = np.where(carbon_mask.reshape(h * w)[start_idx:end_idx])[0] + start_idx
                result[carbon_indices] = len(self.mineral_colors)  # Carbon class is after all minerals
                confidence[carbon_indices] = 1.0  # High confidence for carbon
            
            # Get non-carbon pixels in this batch
            non_carbon_indices = np.where(~carbon_mask.reshape(h * w)[start_idx:end_idx])[0] + start_idx
            
            if len(non_carbon_indices) > 0:
                # Classify non-carbon pixels based on model type
                batch_pixels = pixels_scaled[non_carbon_indices]
                
                if model_type == "knn":
                    # KNN: Use distances to determine confidence
                    dists, indices = classifier.kneighbors(batch_pixels)
                    predictions = classifier.predict(batch_pixels)
                    
                    # Store distances (lower is better)
                    mean_dists = dists.mean(axis=1)
                    # Convert distance to confidence (inverse relationship)
                    conf = np.exp(-mean_dists / 50)  # Exponential decay of confidence with distance
                    confidence[non_carbon_indices] = conf
                    
                    # Assign minerals to pixels within the distance threshold
                    within_threshold = mean_dists < other_threshold
                    result[non_carbon_indices[within_threshold]] = predictions[within_threshold]
                    
                    # Assign "Other" category to pixels beyond the threshold
                    result[non_carbon_indices[~within_threshold]] = len(self.mineral_colors) + 1  # "Other" class
                    confidence[non_carbon_indices[~within_threshold]] = 0.1  # Low confidence for "Other"
                    
                elif model_type == "kmeans":
                    # K-Means: Use distance to cluster centers
                    predictions = classifier.predict(batch_pixels)
                    distances = np.min(classifier.transform(batch_pixels), axis=1)
                    
                    # Convert distance to confidence
                    conf = np.exp(-distances / 50)
                    confidence[non_carbon_indices] = conf
                    
                    # Assign clusters to pixels within the distance threshold
                    within_threshold = distances < other_threshold
                    result[non_carbon_indices[within_threshold]] = predictions[within_threshold]
                    
                    # Assign "Other" category to pixels beyond the threshold
                    result[non_carbon_indices[~within_threshold]] = len(self.mineral_colors) + 1
                    confidence[non_carbon_indices[~within_threshold]] = 0.1
                    
                else:  # SVM and Random Forest
                    # Use probability estimates for confidence
                    predictions = classifier.predict(batch_pixels)
                    proba = classifier.predict_proba(batch_pixels)
                    
                    # Get highest probability for each prediction
                    max_proba = np.max(proba, axis=1)
                    confidence[non_carbon_indices] = max_proba
                    
                    # Assign minerals to pixels with sufficient confidence
                    within_threshold = max_proba > (1.0 - other_threshold/200)  # Convert distance to probability threshold
                    result[non_carbon_indices[within_threshold]] = predictions[within_threshold]
                    
                    # Assign "Other" category to low confidence pixels
                    result[non_carbon_indices[~within_threshold]] = len(self.mineral_colors) + 1
                    confidence[non_carbon_indices[~within_threshold]] = 0.1
            
            # Update progress bar
            self.progress_bar["value"] = (i + 1) / num_batches * 100
            self.root.update_idletasks()
        
        # Reshape back to image shape
        result_image = result.reshape((h, w))
        confidence_image = confidence.reshape((h, w))
        
        # Calculate percentages and confidence intervals
        total_pixels = h * w
        percentages = {}
        pixel_counts = {}
        confidence_intervals = {}
        
        # Helper function to calculate confidence interval
        def confidence_interval(proportion, n, confidence=0.95):
            """Calculate binomial proportion confidence interval"""
            if n == 0 or proportion == 0:
                return 0, 0
            
            z = stats.norm.ppf(1 - (1 - confidence) / 2)
            interval = z * np.sqrt((proportion * (1 - proportion)) / n)
            return max(0, proportion - interval), min(1, proportion + interval)
        
        # Add mineral percentages
        for idx, (name, _) in enumerate(self.mineral_colors.items()):
            mineral_pixels = np.sum(result_image == idx)
            mineral_proportion = mineral_pixels / total_pixels
            percentages[name] = mineral_proportion * 100
            pixel_counts[name] = mineral_pixels
            
            # Calculate confidence interval (95%)
            lower_ci, upper_ci = confidence_interval(mineral_proportion, total_pixels)
            confidence_intervals[name] = (lower_ci * 100, upper_ci * 100)
        
        # Add carbon percentage if detected
        carbon_count = np.sum(carbon_mask)
        if carbon_count > 0:
            carbon_proportion = carbon_count / total_pixels
            percentages["Carbon (Graphite)"] = carbon_proportion * 100
            pixel_counts["Carbon (Graphite)"] = carbon_count
            
            lower_ci, upper_ci = confidence_interval(carbon_proportion, total_pixels)
            confidence_intervals["Carbon (Graphite)"] = (lower_ci * 100, upper_ci * 100)
        
        # Add "Other" category percentage
        other_count = np.sum(result_image == len(self.mineral_colors) + 1)
        if other_count > 0:
            other_proportion = other_count / total_pixels
            percentages["Other"] = other_proportion * 100
            pixel_counts["Other"] = other_count
            
            lower_ci, upper_ci = confidence_interval(other_proportion, total_pixels)
            confidence_intervals["Other"] = (lower_ci * 100, upper_ci * 100)
        
        # Display results
        fig = plt.Figure(figsize=(10, 6))
        
        # Create a colorful visualization of the classification
        ax1 = fig.add_subplot(121)
        max_category = len(percentages)
        cmap = plt.cm.get_cmap('tab10', max_category)
        classification_img = ax1.imshow(result_image, cmap=cmap, vmin=0, vmax=max_category-1)
        ax1.set_title('Classification')
        ax1.axis('off')
        
        # Add color bar
        cbar = fig.colorbar(classification_img, ax=ax1, ticks=range(max_category))
        cbar.set_ticklabels(list(percentages.keys()))
        
        # Create a pie chart of percentages with error bars
        ax2 = fig.add_subplot(122)
        wedges, texts, autotexts = ax2.pie(
            percentages.values(), 
            labels=percentages.keys(), 
            autopct='%1.1f%%',
            textprops={'fontsize': 9}
        )
        ax2.set_title('Mineral Percentages')
        ax2.axis('equal')
        
        # Adjust layout
        fig.tight_layout()
        
        # Display the figure in the results frame
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create a text representation of results with confidence intervals
        results_text = tk.Text(self.results_frame, height=10, width=50)
        results_text.pack(fill=tk.X, pady=5, padx=5)
        
        results_text.insert(tk.END, "Results with 95% Confidence Intervals:\n")
        for name, percentage in percentages.items():
            lower, upper = confidence_intervals[name]
            pixel_count = pixel_counts[name]
            results_text.insert(tk.END, f"{name}: {percentage:.2f}% ({lower:.2f}% - {upper:.2f}%), Pixels: {pixel_count}\n")
        
        # Save results if the checkbox is checked
        if self.save_results_var.get() and self.output_folder:
            self.save_classification_results(fig, result_image, confidence_image, percentages, pixel_counts, confidence_intervals)

    def save_classification_results(self, fig, result_image, confidence_image, percentages, pixel_counts, confidence_intervals):
        """Save classification results to output folder"""
        if not self.output_folder:
            return
            
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
        
        # Create a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the classification figure
        fig_filename = os.path.join(self.output_folder, f"{base_filename}_classification_{timestamp}.png")
        fig.savefig(fig_filename, dpi=300)
        
        # Save the classification data as CSV with confidence intervals
        data_filename = os.path.join(self.output_folder, f"{base_filename}_data_{timestamp}.csv")
        with open(data_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Mineral", "Percentage", "Lower_CI", "Upper_CI", "Pixel_Count"])
            for name, percentage in percentages.items():
                lower, upper = confidence_intervals[name]
                pixel_count = pixel_counts[name]
                writer.writerow([name, percentage, lower, upper, pixel_count])
        
        # Save the classification image as a separate file (without legend)
        # Create a new figure for just the classified image without legend
        plt.figure(figsize=(10, 10))
        plt.imshow(result_image, cmap=plt.cm.get_cmap('tab10', len(percentages)))
        plt.axis('off')
        
        # Save the image
        img_filename = os.path.join(self.output_folder, f"{base_filename}_classified_{timestamp}.png")
        plt.savefig(img_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save as TIFF file without legend
        tiff_filename = os.path.join(self.output_folder, f"{base_filename}_classified_{timestamp}.tiff")
        tifffile.imwrite(tiff_filename, result_image.astype(np.uint8))
        
        # Save confidence map as an additional visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(confidence_image, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Confidence')
        plt.title('Classification Confidence')
        plt.axis('off')
        
        # Save the confidence map
        conf_filename = os.path.join(self.output_folder, f"{base_filename}_confidence_{timestamp}.png")
        plt.savefig(conf_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        messagebox.showinfo("Results Saved", f"Classification results saved to:\n{self.output_folder}")

    def detect_carbon(self, image):
        """
        Detect carbon (graphite) in the image using PIL and scikit-image.
        Carbon appears as diffuse black areas.
        """
        # Convert to grayscale using PIL
        pil_image = Image.fromarray(image)
        gray_image = ImageOps.grayscale(pil_image)
        gray = np.array(gray_image)
        
        # Threshold for dark areas
        threshold = self.carbon_threshold_var.get()
        min_blob_size = self.carbon_blob_size_var.get()
        
        # Binary threshold
        binary = gray < threshold
        
        # Label connected regions
        labeled_array, num_features = ndimage.label(binary)
        
        # Analyze regions and filter by size
        carbon_mask = np.zeros_like(binary, dtype=bool)
        
        # Calculate sizes of labeled regions
        sizes = ndimage.sum(binary, labeled_array, range(1, num_features + 1))
        
        # Filter regions by size
        for i, size in enumerate(sizes):
            if size < min_blob_size:
                carbon_mask[labeled_array == i + 1] = True
        
        return carbon_mask


if __name__ == "__main__":
    root = tk.Tk()
    app = MineralClassifier(root)
    root.mainloop()