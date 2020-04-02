## Objectives

* Apply Hough transform for detecting parametric shapes like circles and lines.
* Apply Harris operator for detecting corners.
* Apply Active Contour Model for semi-supervised shape delineation.

## Deadline

**Thrusday 2/4/2020 11:59 PM**


## Deliverables

### A) Computer Vision Functions

You need to implement Python functions which will support the following tasks:

1. For all given images; detect edges using Canny edge detector, detect lines and circles located in these images (if any) using Hough transform. Superimpose the detected shapes on the images.
2. For given images; detect corners using Harris operator and experiment with different techniques to extract the real corners (e.g experiment with thresholding, local thresholding, non-maxima supression, local maxima extraction).
3. For given images; initialize the contour for a given object and evolve the Active Contour Model (snake) using the greedy algorithm. Represent the output as chain code and compute the perimeter and the area inside these contours.

You should implement these tasks **without depending on OpenCV library or alike**.


Organize your implementation among the following files:

1. `CV404Hough.py`: this will include your implementation for Hough transform for lines and circles (requirement 1).
2. `CV404Harris.py`: this will include your implementation for Harris operator for corners detection (requirement 2).
3. `CV404ActiveContour.py`: this will include your implementation for Snakes Algorithm (requirement 3).

### B) GUI Integration

Integrate your functions in part (A) to the following Qt MainWindow design.
