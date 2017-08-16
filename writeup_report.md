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

This document. Two changes were requested in Udacity review:
1. Draw final lines on undistorted image, instead of original image.

Solution: Changed the input to *draw_lanes()* to give undistorted image to draw on (line 75). Was earlier giving original image mistakenly. Fixed.

2. Improve lane detection in some frames. 

Solution: Used suggestions in review and implemented better thresholds for HSV and RGB space with Sobel. Please see *threshold_image()* in *process_image.py*. Final video shows no large deviations in tracked lanes throughout video. Fixed.

Here is a [link to my resubmitted project video result](./output_video/project_video_output_rev.mp4)

Rest of the writeup is the same, except for the thresholded binary output image, and changes to line numbers in code.

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

I experimented with various combinations of color and gradient thresholds to come up with the most robust method of detecting yellow and white lines under different lighting conditions and videos. Initially, the B-channel from LAB was used for yellow lines with the L-channel  from HSV for white lines. However, that was noisy for the sections of road with changed lighting.
After review from Udacity, I finally converged on the solution as:
- RGB thresholds for robust detection of yellow lines (lines 96-102)
- HSV thresholds for robust detection of white lines (lines 104-109)
- Sobel filter in the x-direction for line detection (lines 23-40)

The low-level pipeline is implemented in process_image.py in *threshold_image()* (lines 111-144). The associated functions are also implemented in the same file. Here's an example of my output for this step. (note: this is from the first video frame of *project_video.mp4*)

![Binary Output][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is implemented in *process_image.py* in the function *view_road_top()* (lines 144-165). The source and destination image points are defined in lines 6-21 of the same file.I chose to fix the source and destination points as follows, in clockwise direction from top left:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 565, 470      | 320, 1        | 
| 720, 470      | 900, 1        |
| 1120, 720     | 900, 720      |
| 200, 720      | 320, 720      |

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

Notice how the algorithm rejects the bright patch of road at the top left of the image in both window and focused search because of limiting the initial search to a margin around the image midpoint.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used the code given in the tutorial to calculate the radius of curvature in function *find_curvature()* (lines 298-314) I modified the code to return a left or right radius, based on the *lane_line* input.

I calculate the position of vehicle in the lane in *find_offset()* (lines 316-326). I assume that the camera is placed so that the image midpoint is the vehicle centre and calculate the lane centre based on the left and right lane coordinates. The difference between the lane centre and image midpoint is the vehicle offset.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in *track_lanes.py* in the function *draw_lanes()* (lines 33-67). Here is an example of my result on a video image:

![Detected Lane][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
The output videos are named appropriately and placed in the *./output_video* folder.

Here is a [link to my original project video result](./output_video/project_video_output.mp4)

Here is a [link to my resubmitted project video result](./output_video/project_video_output_rev.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The performance of the approach on the project and challenge videos shows that the algorithm is robust to moderate changes in lighting, shadows, lost lines, and outliers. This is because I implemented multiple robustness checks in the code in *check_lane_fit()*. The parameters for this logic are defined in lines 7-28.

1. I use the *Line()* class (lines 35-64) to maintain a history of various detected parameters for each lane line, including the current and average fits over time. This helps to track the lines over the video, smoothen the detection plotting, and allows outlier detection more easily. 

2. I rejected lines if a minimum number of pixels were not detected (lines 245,265). This helped avoid the algorithm from fitting a curve to small, isolated clusters of pixels, leading to very wrong curvatures.

3. I check the calculated radius of curvature (lines 261,282) to make sure it is above a minimum threshold, based on empirical observation and using the U.S. highway specifications as the upper limit. This helps to reject lines that are changing too rapidly to be valid detections.

4. I check that the difference in radius magnitude between current and previous calculations is within a reasonable deviation (lines 258, 279). This prevents sudden changes in lane curvature and allows better outlier rejection.

5. I check that the calculated lane width is within reasonable limits (lines 286-296), otherwise I use the previous detected lines. This prevents the lane from suddenly becoming too small or too large, especially when lines are lost or there are shadows on the road.

In spite of all these checks, I was not able to successfully navigate the challenge videos. After analysing the result, I believe the different lighting conditions, extensive shadows, and sudden changes in curvature confuse the simple tracking scheme that I have implemented. A better approach would be to have a model of the lane that is tracked using a Kalman or particle filter. The same image processing pipeline would work but the filter would help to robustly estimate where the lines would be since there would be an underlying curvature and lane width model to help it reject impossible values. I do not believe my current approach can successfully complete the final video since there are too many corner cases to hard code. Unfortunately, I ran out of time to implement a more principled approach using lane model and associated estimation algorithm.