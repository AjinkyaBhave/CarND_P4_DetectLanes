import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from process_image import *

# File and directory paths
params_file = './camera_cal/camera_params.p'
video_file  = ''
img_dir     = './test_images/'
img_file    = 'straight_lines1.jpg'
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
#white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)

# Read camera intrinsic matrix and distortion coefficients
# Created initially with calibrate_camera.py
with open(params_file, mode='rb') as f:
    params = pickle.load(f)
mtx = params['intrinsic']
dist = params['distortion']

# Pipeline to process camera image to isolate lane markings
def pipeline(img):
    img_undist = undistort(img, mtx=mtx, dist=dist, visualise=True)
    img_thresh = threshold_image(img_undist, visualise=True)
    img_top = view_road_top(img, img_thresh, visualise=True)
    return img_top

img = mpimg.imread(img_dir + img_file)
pipeline(img)

