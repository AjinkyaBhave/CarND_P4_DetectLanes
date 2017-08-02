import pickle
import glob
from functools import reduce
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from process_image import *
from detect_lanes import *

# File and directory paths
params_file  = 'camera_params.p'
video_file   = 'project_video.mp4'
video_output = 'project_video_output.mp4'
img_dir      = 'test_images/'
img_file     = 'straight_lines1.jpg'

# Pipeline to process camera image to isolate lane markings
def pipeline(img):
    visualise = False
    img_undist = undistort_image(img, mtx=mtx, dist=dist, visualise=visualise)
    img_thresh = threshold_image(img_undist, visualise=visualise)
    img_top = view_road_top(img_thresh, img, visualise=True)
    find_lane_pixels(img_top, visualise=True)
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
# Number of past images to store for lane detection
n_steps = 4
img_height = 720
img_width = 1280
img_channels = 3
img_buffer = np.zeros((n_steps, img_height, img_width, img_channels), dtype=np.uint8)
# Make a list of calibration images
images = glob.glob(img_dir+'curved_lines*.jpg')
i=0
for image in images:
    img = mpimg.imread(image)
    img_buffer[i] = img
    i+=1
img = reduce(np.maximum, img_buffer)
plt.imshow(img)
plt.show()
pipeline(img)

