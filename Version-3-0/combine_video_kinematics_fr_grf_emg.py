import cv2, scipy.misc, pdb
import numpy as np

filePath = 'W:/ENG_Neuromotion_Shared/group/MI locomotion data/Biomechanical Model/Compressed_q19d20131124tkCORRdsNORMt004/'

animalVideoName = 'footage-cropped.avi'
animalVideoPath = filePath + animalVideoName
animalVideoPathOut = filePath + 'out_' + animalVideoName

mujocoVideoName = 'Q19_20131124_pre_CORR_004_KIN_processed_filtered_video.avi'
mujocoVideoPath = filePath + mujocoVideoName
mujocoVideoPathOut = filePath + 'out_mujoco_video.avi'

frVideoName = 'Q19_20131124_pre_CORR_004_KIN_processed_filtered_fr_animation.mp4'
frVideoPath = filePath + frVideoName
frVideoPathOut = filePath + 'out_fr_animation.avi'

xyzVideoName = 'Q19_20131124_pre_CORR_004_KIN_processed_filtered_kinematics_animation.mp4'
xyzVideoPath = filePath + xyzVideoName
xyzVideoPathOut = filePath + 'out_xyz_animation.avi'

emgVideoName = 'Array_Q19_20131124_emg_animation.mp4'
emgVideoPath = filePath + emgVideoName
emgVideoPathOut = filePath + 'out_emg_animation.avi'

conVideoName = 'Q19_20131124_pre_CORR_004_KIN_processed_filtered_confrc_animation.mp4'
conVideoPath = filePath + conVideoName
conVideoPathOut = filePath + 'out_con_animation.avi'

torqVideoName = 'Q19_20131124_pre_CORR_004_KIN_processed_filtered_qfrc_inverse_animation.mp4'
torqVideoPath = filePath + torqVideoName
torqVideoPathOut = filePath + 'out_torq_animation.avi'

animalVideo = cv2.VideoCapture()
mujocoVideo = cv2.VideoCapture()
frVideo = cv2.VideoCapture()
xyzVideo = cv2.VideoCapture()
emgVideo = cv2.VideoCapture()
torqVideo = cv2.VideoCapture()
conVideo = cv2.VideoCapture()

success = animalVideo.open(animalVideoPath)
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

success = emgVideo.open(emgVideoPath)
eh = int(emgVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
ew = int(emgVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
print('emg footage. Width = %d Heigth = %d' % (ew, eh))

success = conVideo.open(conVideoPath)
ch = int(conVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
cw = int(conVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
print('con footage. Width = %d Heigth = %d' % (cw, ch))

success = torqVideo.open(torqVideoPath)
th = int(torqVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
tw = int(torqVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
print('torq footage. Width = %d Heigth = %d' % (tw, th))

#Cropping
x1a = 0 # top left coordinates
y1a = 0
x2a = aw # botom right coordinates
y2a = ah
aw_prime = x2a - x1a
ah_prime = y2a - y1a

x1m = 0 # top left coordinates
y1m = 0
x2m = mw # botom right coordinates
y2m = mh
mw_prime = x2m - x1m
mh_prime = y2m - y1m

w = xw + 1800 + fw + ew + cw
h = fh

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps=animalVideo.get(cv2.CAP_PROP_FPS)/4
destinationPath = filePath + 'merged_video.avi'

out = cv2.VideoWriter(destinationPath,fourcc, fps, (w,h), isColor=True)

outAnimal  = cv2.VideoWriter(animalVideoPathOut,fourcc, fps, (1800,900), isColor=True)
outMujoco  = cv2.VideoWriter(mujocoVideoPathOut,fourcc, fps, (1800,900), isColor=True)
outFr      = cv2.VideoWriter(frVideoPathOut,fourcc, fps, (fw,fh), isColor=True)
outXyz     = cv2.VideoWriter(xyzVideoPathOut,fourcc, fps, (xw,xh), isColor=True)
outEmg     = cv2.VideoWriter(emgVideoPathOut,fourcc, fps, (ew,eh), isColor=True)
outCon     = cv2.VideoWriter(conVideoPathOut,fourcc, fps, (cw,ch), isColor=True)
outTorq    = cv2.VideoWriter(torqVideoPathOut,fourcc, fps, (tw,th), isColor=True)

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)            # create windows, use WINDOW_AUTOSIZE for a fixed window size
font = cv2.FONT_HERSHEY_DUPLEX

while cv2.waitKey(1) != ord(' ') and animalVideo.isOpened() and mujocoVideo.isOpened():  # until the Esc key is pressed or webcam connection is lost
    animalFrameReadSuccessfully, imgAnimal = animalVideo.read()                  # read next frame
    mujocoFrameReadSuccessfully, imgMujoco = mujocoVideo.read()
    frFrameReadSuccessfully,     imgFr     = frVideo.read()                  # read next frame
    xyzFrameReadSuccessfully,    imgXyz    = xyzVideo.read()
    emgFrameReadSuccessfully,    imgEmg    = emgVideo.read()
    conFrameReadSuccessfully,    imgCon    = conVideo.read()
    torqFrameReadSuccessfully,   imgTorq   = torqVideo.read()

    # weird black band removal
    #imgXyz[:,-15:-1] = 0
    #imgXyz[:,-1] = 0

    #pdb.set_trace()
    if not animalFrameReadSuccessfully or\
        not mujocoFrameReadSuccessfully or\
        not frFrameReadSuccessfully or\
        not xyzFrameReadSuccessfully or\
        not emgFrameReadSuccessfully or\
        not conFrameReadSuccessfully or\
        not torqFrameReadSuccessfully or\
        imgAnimal is None or imgMujoco is None or imgFr is None or imgXyz is None or imgEmg is None or imgCon is None or imgTorq is None:
        break

    imgAnimal = imgAnimal[y1a:y2a, x1a:x2a]

    imgMujoco = imgMujoco[y1m:y2m, x1m:x2m]

    imgAnimal = cv2.resize(imgAnimal, (1800,900))
    imgMujoco = cv2.resize(imgMujoco, (1800,900))

    #pdb.set_trace()
    imgTorqDist = cv2.resize(imgTorq, (600, int(1800 * 4/5) ))
    imgConDist = cv2.resize(imgCon, (600, 1800 - int(1800 * 4/5) ))

    #pdb.set_trace()
    imgConTorq = np.concatenate((imgTorqDist, imgConDist), axis = 0)

    #pdb.set_trace()

    imgOut = np.concatenate((imgAnimal, imgMujoco), axis = 0)
    #pdb.set_trace()
    imgOut = np.concatenate((imgXyz, imgOut, imgFr, imgEmg, imgConTorq), axis = 1)
    #pdb.set_trace()

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
