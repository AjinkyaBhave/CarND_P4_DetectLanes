import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from process_image import *

# File and directory paths
params_file  = 'camera_params.p'
video_file   = 'project_video.mp4'
video_output = 'project_video_output.mp4'
img_dir      = 'test_images/'
img_file     = 'test3.jpg'

# Pipeline to process camera image to isolate lane markings
def pipeline(img):
    visualise = True
    img_undist = undistort_image(img, mtx=mtx, dist=dist, visualise=visualise)
    img_thresh = threshold_image(img_undist, visualise=visualise)
    img_top = view_road_top(img, img_thresh, visualise=visualise)
    return img_top

# Read camera intrinsic matrix and distortion coefficients
# Created initially with calibrate_camera.py
with open(params_file, mode='rb') as f:
    params = pickle.load(f)
mtx  = params['intrinsic']
dist = params['distortion']

'''clip1 = VideoFileClip(video_file).subclip(0,5)
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(video_output, audio=False)
'''
img = mpimg.imread(img_dir + img_file)
pipeline(img)

