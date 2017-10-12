import numpy as np
import cv2, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--modelVideoFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1.avi')
parser.add_argument('--origVideoFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_orig.avi')
parser.add_argument('--outputFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_stacked.avi')

args = parser.parse_args()
outputFile = args.outputFile
origVideoFile = args.origVideoFile
modelVideoFile = args.modelVideoFile

model = cv2.VideoCapture(modelVideoFile)
orig = cv2.VideoCapture(origVideoFile)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
stackHeight, stackWidth = (2*model.get(cv2.CAP_PROP_FRAME_HEIGHT),
    model.get(cv2.CAP_PROP_FRAME_WIDTH))
output = cv2.VideoWriter(outputFile, fourcc, orig.get(cv2.CAP_PROP_FPS),
    (int(stackWidth), int(stackHeight)))

assert model.get(cv2.CAP_PROP_FRAME_COUNT) == orig.get(cv2.CAP_PROP_FRAME_COUNT)

go = True
showViewer = False
while(go):
    # Capture frame-by-frame
    gotFrame, frame = model.read()
    gotOrigFrame, origFrame = orig.read()

    go = gotFrame and gotOrigFrame
    if go:
        height, width = frame.shape[:2]
        origFrame = cv2.resize(origFrame, (width, height),
            interpolation = cv2.INTER_CUBIC)

        both = np.vstack((frame, origFrame))
        # Display the resulting frame
        output.write(both)
        if showViewer:
            cv2.imshow('frame', both)
            cv2.waitKey(0)

# When everything done, release the capture
model.release()
orig.release()
output.release()
cv2.destroyAllWindows()
