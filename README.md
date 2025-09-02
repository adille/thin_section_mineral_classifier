# Mineral Thin Section Classifier

This application helps geologists and mineralogists classify minerals in thin section images and calculate their proportions.

## Features

- Select a folder containing mineral thin section images (TIFF, PNG, JPG)
- Navigate through images with next/previous buttons
- Zoom in/out for detailed examination of thin sections
- Click on pixels to select mineral colors
- Special handling for carbon (graphite) detection in diffuse black areas
- Multiple classification models (KNN, SVM, Random Forest, K-Means)
- Save and load mineral selections for each image
- "Other" category for pixels that don't match any known minerals
- Progress bar for classification processing
- Reset button to clear results when finished
- Results showing percentage of each mineral with visualization
- Confidence intervals for mineral proportions
- Classified TIFF images without legends
- Complete CSV statistics including pixel counts
- Automatic saving of classification results to a subfolder
- Parameter descriptions panel with guidance

## Installation

### Prerequisites
- Python 3.7 or higher
- Required Python packages:

```
pip install numpy pillow matplotlib scikit-learn scikit-image scipy tifffile
```

### Running the Application

1. Download the `mineral_classifier.py` file
2. Run the application:

```
python mineral_classifier.py
```

## Usage Instructions

1. **Select Folder**: Click "Select Folder" to choose a directory containing your mineral thin section images.

2. **Navigate Images**: Use "Previous" and "Next" buttons to browse through the images in the folder.

3. **Zoom and Navigate**:
   - Use the 🔍+ and 🔍- buttons to zoom in and out
   - Use the mouse wheel to zoom in and out
   - Pan by using the scrollbars or middle-click and drag
   - Click "Reset Zoom" to return to 100% view

4. **Select Mineral Samples**:
   - Click on pixels in the image that represent a specific mineral
   - Enter the mineral name in the "Mineral Name" field
   - Click "Add Mineral" to register this mineral type
   - Repeat for all minerals you want to identify
   - **Important**: Select multiple samples for each mineral to capture diversity

5. **Save/Load Selections**:
   - Click "Save Selections" to save the current mineral selections for this image
   - Click "Load Selections" to load previously saved mineral selections
   - Selections are automatically saved to and loaded from the results subfolder

6. **Choose Classification Model**:
   - K-Nearest Neighbors (KNN): Best for general mineral classification
   - Support Vector Machine (SVM): Good for complex boundary distinctions
   - Random Forest: Handles varied mineral textures well
   - K-Means: Simple unsupervised clustering approach

7. **Adjust Settings**:
   - Carbon Threshold: Controls darkness threshold for carbon detection
   - Min Blob Size: Controls size threshold for carbon areas
   - Distance Threshold: Controls strictness of mineral matching

8. **Classify Image**:
   - Click "Classify Image" to process the current image
   - The progress bar shows classification status
   - View results in the center panel showing:
     - Color-coded classification map
     - Pie chart of mineral proportions
     - Text percentages for each mineral with confidence intervals

9. **Reset Results**:
   - Click "Reset Results" to clear the current classification results

10. **Save Results**:
    - Check "Save Results" to automatically save classification data
    - Results are saved in a "mineral_classification_results" subfolder
    - Saved files include:
      - Classification image with color map
      - Clean TIFF image of classification (no legend)
      - CSV file with mineral percentages, confidence intervals, and pixel counts
      - Confidence map visualization

## Interface Layout

The application has three main panels:

1. **Left Panel**:
   - Image display with zoom controls
   - Navigation buttons
   - Progress bar

2. **Center Panel**:
   - Mineral selection controls
   - Classification model options
   - Classification results

3. **Right Panel**:
   - Parameter descriptions and help
   - Threshold adjustment controls
   - Tips for effective use

## Parameter Descriptions

### Carbon Threshold (0-100)
Controls how dark a pixel must be to be considered as potential carbon.
- Lower values (10-20): Only very dark pixels identified as carbon
- Higher values (40-50): More medium darkness pixels included
- Default: 30

### Min Blob Size (10-1000)
Controls the maximum size of black areas that can be classified as carbon.
- Small values: More black areas classified as carbon
- Large values: Only small diffuse black areas are carbon
- Default: 100 pixels

### Distance Threshold (10-200)
Controls how strictly pixels must match known minerals.
- Lower values (10-30): Strict matching, more pixels classified as "Other"
- Higher values (100+): Lenient matching, more pixels assigned to minerals
- Default: 50

## Output Files

The application saves several output files for each classified image:

1. **Classification Image (.png)**: 
   - Color-coded visualization with legend and pie chart

2. **Classified TIFF (.tiff)**:
   - Clean classification map without legend
   - For direct comparison with original image

3. **Data CSV (.csv)**:
   - Mineral percentages
   - Confidence intervals (95%)
   - Pixel counts for each class

4. **Confidence Map (.png)**:
   - Visualization of classification confidence
   - Shows areas of uncertainty in the classification

## Notes

- For best results, select multiple sample pixels for each mineral type
- Select samples from different areas to capture mineral variation
- Ensure proper lighting and focus in your thin section images
- Use the zoom feature to accurately select mineral pixels
- Experiment with different classification models for optimal results
- Check the parameter descriptions panel for guidance
