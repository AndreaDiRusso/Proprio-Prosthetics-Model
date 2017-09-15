import numpy as np
import cv2, scipy.misc, pdb
from subprocess import call

filePath = 'E:\\Google Drive\\Github\\tempdata\\Biomechanical Model\\figures\\'

call(['ffmpeg', '-f', 'rawvideo', '-pixel_format', 'rgb24',
    '-video_size', '3000x1500', '-framerate','1',
    '-i', 'flywheel_design_out.raw', '-vf', 'vflip', 'mujoco_video.avi'],
    cwd=filePath)
