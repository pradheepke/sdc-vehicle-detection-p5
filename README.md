# Vehicle Detection Project

[//]: # (Image References)

In this project, I implemented a classification, detection and tracking pipeline for identifying cars. Main steps:
 - Feature extraction
 - Classification model training and evaluation
 - Sliding window search to do detection
 - Tracking to help improve the accuracy by removing some false positives


## Details
### Feature extraction
 - I used color histogram binning using 32 bins for each color channel. I used the HSV color space. 
 - I used HoG features. In terms of parameters, I stuck to one set only here, which seems to be commonly used in literature. I kept cells_per_block = 2, pixels_per_cell = 8, orientations = 9. 
 - I also added features from spatial binning, set at size 32 x 32.

I found that just using color histogram features was pretty good in terms of accuracy (97%), adding HoG features improved the accuracy on an held-out test set to 99%. 

Totally I was using 8460 features. 

### Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell #3 of the IPython notebook, function `extract_hog_features()`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

TODO

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HoG parameters.

For HoG parameters, I used `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. Given that the prediction accuracy seemed pretty good I did not explore other choices here. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HoG features (and color features if you used them).

I trained a linear SVM using `scikit.svm.LinearSVC`. SVMs give a good tradeoff between accuracy and speed, and from the data that I used for training and testing, I got 99% test set accuracy. I varied the parameter C, using grid search. And found that the regularization parameter C = [10, 100] to provide best test set accuracy.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the `find_cars(...)` function. The main brunt of the work here is doing this efficiently by doing the sliding window on the HoG feature map instead of cropping on the original image and then calling HoG extractor multiple times. HoG returns a feature map that is structured as an n-dimensional array like: (block_x, block_y, cell_pos_in_block_x, cell_pos_in_block_y, orientation_bin). To map feature map to original image, we need to do some arithmetic to translate the block co-ordinates to pixel co-ordinates. Bulk of the `find_cars` code is doing this translation. Once we do extract the hog features, we crop the image and compute color histograms and spatial binning features (we could possibly make this also more efficient in a future iteration). I just used one scale: 64 pixels in this iteration. I would like to try one more higher scale (say, 96) to catch larger cars and one smaller (say 32) to catch cars in the distance.
 
![alt text][image3]

Some example test images showing it is working:

![alt text][image4]
---

### Video Implementation

This is the final [link to my video result](./project_video.mp4).

#### Heatmaps and voting to reduce false positives and reducing multiple overlapping detections

To address duplicates and false positives, I used the heatmap + voting technique. That is, count number of detected boxes each pixel is part of, and keep only pixels that were part of more than 1 box. 

I then used `scipy.ndimage.measurements.label()` to identify individual blobs and constructed smallest bounding rectangles to cover the area of each blob detected.

### Tracking
Even after this there was a fair amount of false positives, as seen in this short clip:


I implemented a basic tracking method: look back last N frames, and look for overlapping boxes (after the heatmap integration is done) in the last N frames. Only keep those boxes that have some amount of overlap with at least one box in the last N frames. This method is implemented with the method `process_img_with_tracking()`. The helper function `area_intersection` is used to determine if two boxes overlap or not. This helps in reducing the number of false positives significantly as well, and helped produce the final video. 

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

