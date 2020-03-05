
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
import matplotlib.animation as animation
from segmentationUtils import segmentationUtils
import matplotlib.patches as patches


def main():
            
    path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/Standard files/multi_objects.mp4'
    #path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/Standard files/four_objects.mp4'



    video = cv.VideoCapture(path)

    plotMask = False

    if plotMask:
        fig, axarr = plt.subplots(1,2)
        axarr[0].set_title('standard image')
        axarr[1].set_title('watershed mask')
    else:
        fig,axarr = plt.subplots(1)
        handle = None

    off_set_text = 3
    rects = []
    texts = []
    while(video.isOpened()):
        ret, frame = video.read()
        if frame.shape != 0:
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            watershedImage, mask, detection = segmentationUtils.watershed(frame,minimumSizeBox=0.5,smallBBFilter=True,centroidDistanceFilter = True, mergeOverlapingDetectionsFilter = True)

            if plotMask:
                axarr[0].imshow(watershedImage)
                axarr[1].imshow(mask)
            else:
                if handle is None:
                    handle = plt.imshow(watershedImage)
                else:
                    handle.set_data(watershedImage)
            cleanFigure(rects,texts)
            for j in range(len(detection)):
                rect = patches.Rectangle((detection[j][1],detection[j][0]),detection[j][3],detection[j][2],linewidth=1,edgecolor='r',facecolor='none')
                plt.gca().add_patch(rect)
                #the append is necessary to make the predictions not visible after the refresh of the frame
                rects.append(rect)
            plt.pause(0.01)
            plt.draw()


    video.release()
    cv.destroyAllWindows()


def cleanFigure(rects,texts):
    for s in range(len(rects)):
        rects[s].set_visible(False)

    for s in range(len(texts)):
        texts[s].set_visible(False)

if __name__ == "__main__":
	main()