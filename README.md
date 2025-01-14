# Image Processing Tool

This Python application provides a graphical user interface (GUI) for performing various image processing operations using OpenCV and other libraries.  It allows users to load images, apply different processing techniques, adjust parameters, and view the results in real-time.

## Features

* **Image Loading:** Load images in common formats (jpg, jpeg, png, bmp) using a file dialog.
* **Blurring:** Apply Gaussian, Median, or Average blur with adjustable kernel size.
* **Contour Detection:**  Perform contour detection using Canny or Laplacian methods.  Adjust threshold levels for edge detection.  Contours are highlighted with green outlines.
* **Thresholding:** Apply various thresholding techniques (Binary, Binary Inverse, Trunc, Tozero, Tozero Inverse) with an adjustable threshold level.
* **Histogram Display and Equalization:** Display the image histogram and apply histogram equalization to enhance contrast.
* **Dilation/Erosion:** Perform morphological operations (dilation and erosion) with an adjustable kernel size.
* **Hough Line Detection:** Detect lines in the image using the Hough transform.  Adjust the angle of detection and line thickness.
* **Real-time Preview:**  The "View" button next to most functions allows users to see the effect of that operation *before* applying it permanently. This non-destructive preview helps in fine-tuning parameters.
* **Apply/Remove:**  Click "Apply" to add the current settings of a function to the processing pipeline. Click "Remove" to remove that function from the processing pipeline. The image is re-processed each time "Apply" or "Remove" is clicked.
* **Processing Pipeline:** Operations are applied sequentially in a processing pipeline.  This allows for combining multiple effects.  The "Apply" button adds an operation to the pipeline, while "Remove" takes it out. The pipeline is re-applied whenever a change is made.



## Usage

1. **Load Image:** Click the "Load Image" button to select an image file.
2. **Adjust Parameters:** Use the sliders and dropdown menus to adjust the parameters for each image processing function.
3. **View:** Click the "View" button to see the effect of the selected function with the current parameters without permanently changing the image.
4. **Apply:** Click the "Apply" button to apply the selected function and parameters to the image.  This adds it to the processing pipeline.
5. **Remove:** Click the "Remove" button to remove the last applied operation.



## Dependencies

* Python 3
* OpenCV (`cv2`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)
* Pillow (`PIL`)
* Tkinter (`tkinter`)


## Installation

1. Ensure you have Python 3 installed.
2. Install the required libraries using pip:
   ```bash
   pip install opencv-python numpy matplotlib pillow
   ```
   (Tkinter is usually included with Python installations.)

## Running the Application

1. Save the code as `img_proc.py`.
2. Navigate to the directory containing `img_proc.py` in your terminal.
3. Run the script using the command:
   ```bash
   python img_proc.py
   ```

## Example

To apply a Gaussian blur and then Canny edge detection:

1. Load an image.
2. In the "Blur" section, select "Gaussian" and adjust the level. Click "Apply".
3. In the "Contour" section, select "Canny" and adjust the level. Click "Apply".
4. The resulting image will display the blurred image with highlighted edges.


## Future Improvements

* **More Image Processing Functions:** Add more image processing functionalities, such as edge sharpening, color adjustments, and image transformations.
* **Saving Processed Images:** Implement the ability to save the processed images to a file.
* **Undo/Redo Functionality:** Add undo and redo functionality for easier experimentation.
* **Improved Histogram View:**  Make the histogram view more interactive and informative.
* **Region of Interest Selection:** Allow users to select a specific region of interest for applying processing functions.


This README provides a comprehensive overview of the Image Processing Tool, its features, and how to use it.  It serves as documentation for users and developers who may want to contribute to or extend the application's functionalities.