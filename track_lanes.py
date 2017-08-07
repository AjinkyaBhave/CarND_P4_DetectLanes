import numpy as np
import pickle
import glob
import os
from collections import deque
from functools import reduce
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from process_image import *
from detect_lanes import *

# File and directory paths
params_file   = 'camera_params.p'
video_input   = 'project_video.mp4'
video_output  = 'project_video_output.mp4'
img_dir       = 'test_images/'
img_file      = 'straight_lines1.jpg'
video_img_dir =  img_dir+'project_video/'

# Read camera intrinsic matrix and distortion coefficients
# Created initially with calibrate_camera.py
with open(params_file, mode='rb') as f:
    params = pickle.load(f)
mtx  = params['intrinsic']
dist = params['distortion']

# Number of past camera images to store for smoothing
n_prev_frames = 10
# Contains history of last n_prev_frames camera images
img_queue = deque(maxlen=n_prev_frames)

def draw_lanes(img_undist, img_top, left_fit, right_fit, visualise=False):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_top).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    img_out = np.copy(img_undist)

    if (len(left_fit) == (left_line.fit_degree + 1)) and (len(right_fit) == (right_line.fit_degree + 1)):
        # Generate x and y values for plotting
        y_fit = np.linspace(0, img_top.shape[0] - 1, img_top.shape[0])
        leftx_fit = left_fit[0] * y_fit ** 2 + left_fit[1] * y_fit + left_fit[2]
        rightx_fit = right_fit[0] * y_fit ** 2 + right_fit[1] * y_fit + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([leftx_fit, y_fit]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx_fit, y_fit])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Calculate inverse homography using source and destination pixels
        Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
        # Combine the result with the original image
        img_out = cv2.addWeighted(img_undist, 1, newwarp, 0.3, 0)
        text_str = 'Left Radius: {0:.2f} m '.format(left_line.radius) + 'Right Radius: {0:.2f} m '.format(right_line.radius)\
                   + 'Offset: {0:.2f} m '.format(left_line.centre_offset)
        cv2.putText(img_out, text_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if visualise:
            plt.imshow(img_out)
            plt.show()

    return img_out

# Pipeline to process camera image to isolate lane markings
def image_pipeline(img_max, img, visualise=False):
    img_undist = undistort_image(img_max, mtx=mtx, dist=dist, visualise=False)
    img_thresh = threshold_image(img_undist, visualise=visualise)
    img_top = view_road_top(img_thresh, img_max, visualise=visualise)
    left_fit, right_fit = fit_lane(img_top, visualise=visualise)
    img_out = draw_lanes(img, img_top, left_fit, right_fit, visualise=visualise)
    return img_out

def track_lanes(img):
    # Add newest camera frame from right end of queue
    img_queue.append(img)
    # If queue has enough images to begin processing
    if len(img_queue) == n_prev_frames:
        img_max = reduce(np.maximum, np.asarray(img_queue))
        #plt.imshow(img_max)
        #plt.show()
        img_out = image_pipeline(img_max, img)
        # Remove oldest camera frame from left end of queue
        img_queue.popleft()
        return img_out
    else:
        return img

if __name__ == '__main__':

    # Run on video file if true else run on test images
    TEST_ON_VIDEO = 1
    # No lane lines are detected at start of driving
    left_line.detected = False
    right_line.detected = False

    if TEST_ON_VIDEO:
        # Video is at 25 FPS
        clip = VideoFileClip(video_input).subclip(0,2)
        clip_output = clip.fl_image(track_lanes) #NOTE: this function expects color images!!
        clip_output.write_videofile(video_output, audio=False)
    else:
        if not os.listdir(video_img_dir):
            v_start = 0
            v_end   = 1
            video_times = np.linspace(v_start, v_end, n_prev_frames+1)
            clip = VideoFileClip(video_input).subclip(v_start, v_end)
            for vt in video_times:
                video_img_file = video_img_dir + 'video{:3.2}.jpg'.format(vt)
                clip.save_frame(video_img_file, vt)

        # Read camera frames from disk
        img_files = glob.glob(video_img_dir+'video*.jpg')
        for img_file in img_files:
            img = mpimg.imread(img_file)
            track_lanes(img)
