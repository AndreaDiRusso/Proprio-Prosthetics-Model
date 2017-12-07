import cv2, scipy.misc, pdb
import numpy as np

filePath = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/'

animalVideoName = 'T_1_orig.avi'
animalVideoPath = filePath + animalVideoName
animalVideoPathOut = filePath + 'out_' + animalVideoName

mujocoVideoName = 'T_1_filtered_video.avi'
mujocoVideoPath = filePath + mujocoVideoName
mujocoVideoPathOut = filePath + 'out_mujoco_video.avi'

frVideoName = 'T_1_filtered_fr_animation.mp4'
frVideoPath = filePath + frVideoName
frVideoPathOut = filePath + 'out_fr_animation.avi'

xyzVideoName = 'T_1_filtered_kinematics_animation.mp4'
xyzVideoPath = filePath + xyzVideoName
xyzVideoPathOut = filePath + 'out_xyz_animation.avi'

animalVideo = cv2.VideoCapture()
mujocoVideo = cv2.VideoCapture()
frVideo = cv2.VideoCapture()
xyzVideo = cv2.VideoCapture()

success = animalVideo.open(animalVideoPath)
#pdb.set_trace()
ah = int(animalVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
aw = int(animalVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Animal footage. Width = %d Heigth = %d' % (aw, ah))

success = mujocoVideo.open(mujocoVideoPath)
mh = int(mujocoVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
mw = int(mujocoVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Mujoco footage. Width = %d Heigth = %d' % (mw, mh))

success = frVideo.open(frVideoPath)
fh = int(frVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
fw = int(frVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
print('fr footage. Width = %d Heigth = %d' % (fw, fh))

success = xyzVideo.open(xyzVideoPath)
xh = int(xyzVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
xw = int(xyzVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
print('xyz footage. Width = %d Heigth = %d' % (xw, xh))

#Cropping
x1a = 400 # top left coordinates
y1a = 0
x2a = aw - 400 # botom right coordinates
y2a = ah
aw_prime = x2a - x1a
ah_prime = y2a - y1a

x1m = 250 # top left coordinates
y1m = 0
x2m = mw - 250 # botom right coordinates
y2m = mh
mw_prime = x2m - x1m
mh_prime = y2m - y1m

w = xw + 1800 + fw
h = fh

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
fps=animalVideo.get(cv2.CAP_PROP_FPS)/4
destinationPath = filePath + 'merged_video.avi'

out = cv2.VideoWriter(destinationPath,fourcc, fps, (w,h), isColor=True)

outAnimal = cv2.VideoWriter(animalVideoPathOut,fourcc, fps, (1800,900), isColor=True)
outMujoco = cv2.VideoWriter(mujocoVideoPathOut,fourcc, fps, (1800,900), isColor=True)
outFr     = cv2.VideoWriter(frVideoPathOut,fourcc, fps, (fw,fh), isColor=True)
outXyz    = cv2.VideoWriter(xyzVideoPathOut,fourcc, fps, (xw,xh), isColor=True)

count = 0
start_idx = 50
end_idx = 300

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)            # create windows, use WINDOW_AUTOSIZE for a fixed window size
font = cv2.FONT_HERSHEY_DUPLEX
#advance animal video to the proper frame
while cv2.waitKey(1) != ord(' ') and animalVideo.isOpened() and count < 0:  # until the Esc key is pressed or webcam connection is lost
    blnFrameReadSuccessfully, imgAnimal = animalVideo.read()                  # read next frame
    if not blnFrameReadSuccessfully or imgAnimal is None:                     # if frame was not read successfully
        print("error: frame not read\n")                                         # print error message to std out
        break                                                                 # exit while loop (which exits program)
    count += 1
# end while

while cv2.waitKey(1) != ord(' ') and animalVideo.isOpened() and mujocoVideo.isOpened():  # until the Esc key is pressed or webcam connection is lost
    animalFrameReadSuccessfully, imgAnimal = animalVideo.read()                  # read next frame
    mujocoFrameReadSuccessfully, imgMujoco = mujocoVideo.read()
    frFrameReadSuccessfully, imgFr = frVideo.read()                  # read next frame
    xyzFrameReadSuccessfully, imgXyz = xyzVideo.read()

    # weird black band removal
    #imgXyz[:,-15:-1] = 0
    #imgXyz[:,-1] = 0

    #pdb.set_trace()
    if not animalFrameReadSuccessfully or\
        not mujocoFrameReadSuccessfully or\
        not frFrameReadSuccessfully or\
        not xyzFrameReadSuccessfully or\
        imgAnimal is None or imgMujoco is None or imgFr is None or imgXyz is None:
        break

    imgAnimal = imgAnimal[y1a:y2a, x1a:x2a]

    imgMujoco = imgMujoco[y1m:y2m, x1m:x2m]

    imgAnimal = cv2.resize(imgAnimal, (1800,900))
    imgMujoco = cv2.resize(imgMujoco, (1800,900))

    imgOut = np.concatenate((imgAnimal, imgMujoco), axis = 0)
    #pdb.set_trace()
    imgOut = np.concatenate((imgXyz, imgOut, imgFr), axis = 1)


    #cv2.putText(imgOut,'Experimental Marker Position',(2250,870), font, 1,(0,0,0),1,cv2.LINE_AA)
    #cv2.putText(imgOut,'Computed Marker Position',(2250, 820), font, 1,(0,0,0),1,cv2.LINE_AA)
    #pdb.set_trace()
    #cv2.imshow("Preview", imgOut)
    cv2.imshow('Preview',imgXyz)
    out.write(imgOut)

    outAnimal.write(imgAnimal)
    outMujoco.write(imgMujoco)
    outFr.write(imgFr)
    outXyz.write(imgXyz)

# end while
animalVideo.release()

outAnimal.release()
outMujoco.release()
outFr.release()
outXyz.release()

cv2.destroyAllWindows()
out.release()
