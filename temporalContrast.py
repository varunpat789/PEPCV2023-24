import cv2
import numpy as np

#GOAL: Experiment with comparing consecutive frames for faster/lower power drain object detection

capture = cv2.VideoCapture(0)

ret, newFrame = capture.read()    
newFrameShape = np.shape(newFrame)

oldFrame = b = np.full(newFrameShape[0:2], 0, np.int8)

temporalFrame = np.full(newFrameShape[0:2], 0, np.int8)

def toTemporal(i):
    if i < -30:  #off event
        return 0
    elif i > 30: #on event
        return 1
    else:        #not enough to decide
        return .5
    
applyall = np.vectorize(toTemporal)

while True:
    ret, newFrame = capture.read()
    newFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY).astype(np.int8)
    
    difference = cv2.subtract(oldFrame,newFrame)
    temporalFrame = applyall(difference).astype(np.float32)

    cv2.imshow('temporalFrame', temporalFrame)

    oldFrame = newFrame

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()