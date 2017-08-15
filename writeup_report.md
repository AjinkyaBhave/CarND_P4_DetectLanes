## Writeup 
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_output.png 		
[image2]: ./output_images/undist_output.png 			
[image3]: ./output_images/threshold_output.png
[image4]: ./output_images/projected_straight_output.png 	
[image5]: ./output_images/projected_curved_output.png			
[image6]: ./output_images/hist_win_search.png 			
[image7]: ./output_images/lines_win_search.png 			
[image8]: ./output_images/lines_focus_search.png 			
[image9]: ./output_images/detect_lane_output.png 			
[video1]: ./output_video/project_video_output.mp4 		

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This document.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in *calibrate_camera.py* (lines 10-43). The code has been modified from the Udacity example located [here](https://github.com/udacity/CarND-Camera-Calibration).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Calibration Output][image1]

The calculated camera parameters are stored in a pickled file called *camera_params.p* and used in the lane detection pipeline later.

### Pipeline (single images)
The complete pipeline is implemented in the following files:
1. *track_lanes.py*: Reads video and camera parameters, defines image pipeline, and draws final lane lines on output video.
2. *detect_lanes.py*: fits lane lines in a single image using sliding window or focused search, checks goodness of fit, and outlier robustness measures.
3. *process_image.py*: low-level image processing for gradient and color thresholds, perspective transformation, and lane pixel identification.
4. *calibrate_camera.py*: Calculates and saves intrinsic parameters from checker board images.
#### 1. Provide an example of a distortion-corrected image.

The image below shows the application of the distortion correction to one of the test images (straight_lines2.jpg).
![Road Transformed][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I experimented with various combinations of color and gradient thresholds to come up with the most robust method of detecting yellow and white lines under different lighting conditions and videos. Initially, the s-channel from HSV was used for yellow lines with the V-channel for white lines. However, that was noisy for the challenge video images. I finally converged on the solution as:
- B-channel from LAB space for robust detection of yellow lines
- Sobel filter in the x-direction for line detection
- L-channel from HSL space on Sobel gradient image and normalisation for intensity

The low-level pipeline is implemented in process_image.py in threshold_image() (lines 98-106). The associated functions are also implemented in the same file. Here's an example of my output for this step. (note: this is from the first video frame of *project_video.mp4*)

![Binary Output][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is implemented in *process_image.py* in the function *view_road_top()* (lines 144-165). The source and destination image points are defined in lines 6-21 of the same file.I chose to fix the source and destination points as follows, in clockwise direction from top left:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 470      | 350, 1        | 
| 720, 470      | 875, 1        |
| 1130, 670     | 900, 720      |
| 200, 670      | 320, 720      |

I chose to crop the bottom part of the image (y-coordinate of 670) to remove any specular reflection artifacts that appear on the car bonnet. This works well in practice and helps the lane fitting in the rest of the pipeline.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto the *straight_lines2* test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Top View Straight][image4]

I also checked the transform on curved road segments from the video.

![Top View Curved][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane lines detection, fitting, and checking is implemented in multiple functions in the file *detect_lanes.py*. The main function is *fit_lane()* (lines 148-233) which calls a function for window-based search or focused search, checks for lane fit, and outlier detection.

The initial window-based search is in *search_window()*, which is based on the code in the tutorial. I have modified the code to search for the first peak in a small area around the image midpoint. This helps to avoid spurious detections at the image boundaries from road edges, other cars, fences, and bright dust surfaces. The result of this modification can be seen in the clean peaks detected in the window histogram in the image below.

![Line Pixels Histogram][image6]

I also check for a minimum number of pixels to be detected before accepting the pixels as belonging to a line. An example binary image with the results of *search_window()* is shown below.

![Window Search Result][image7]

After the lane lines have been detected once, subsequent detections are carried out around the last detected lane centres (lines 166-170) in *fit_lane()*. Once the lines are detected, I check for goodness of fit in *check_lane_fit()* (lines 235-298) and reject outliers. I fit a second-order polynomial to the final detected pixels in lines (250,271). An example binary image with the results of the focused search and lane fit checking is shown below.

![Focused Search Result][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used the code given in the tutorial to calculate the radius of curvature in function *find_curvature()* (lines 300-316) I modified the code to return a left or right radius, based on the *lane_line* input.

I calculate the position of vehicle in the lane in *find_offset()* (lines 318-328). I assume that the camera is placed so that the image midpoint is the vehicle centre and calculate the lane centre based on the left and right lane coordinates. The difference between the lane centre and image midpoint is the vehicle offset.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in *track_lanes.py* in the function *draw_lanes()* (lines 33-67). Here is an example of my result on a video image:

![Detected Lane][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
The output videos are named appropriately and placed in the *./output_video* folder.

Here is a [link to my project video result](./output_video/project_video_output.mp4)

Here is a [link to my challenge video result](./output_video/challenge_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
