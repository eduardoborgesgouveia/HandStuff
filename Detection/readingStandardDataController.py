
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
import matplotlib.animation as animation
from segmentationUtils import segmentationUtils
import matplotlib.patches as patches



path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/Standard files/multi_objects.mp4'



video = cv.VideoCapture(path)

plotMask = True

if plotMask:
    fig, axarr = plt.subplots(1,2)
    axarr[0].set_title('standard image')
    axarr[1].set_title('watershed mask')
else:
    fig,axarr = plt.subplots(1)
    handle = None


while(video.isOpened()):
    ret, frame = video.read()
    if frame.shape != 0:
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        watershedImage, mask, detection = segmentationUtils.watershed(frame)
        
        if plotMask:
            axarr[0].imshow(watershedImage)
            axarr[1].imshow(mask)
        else:
            if handle is None:
                handle = plt.imshow(watershedImage)
            else:
                handle.set_data(watershedImage)

        
        plt.pause(0.01)
        plt.draw()


video.release()
cv.destroyAllWindows()