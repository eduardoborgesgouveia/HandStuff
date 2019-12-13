
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
from classifierTools import classifierTools
import matplotlib.animation as animation
from segmentationUtils import segmentationUtils
import matplotlib.patches as patches

def main():
        
    # path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/standardized data/Mug.aedat'
    #path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/two_objects.aedat'
    path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/key.aedat'
    #path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/key_2.aedat'
    #path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/multi_objects_2.aedat'

    model = classifierTools.openModel('model/model.json',
                                        'model/model.h5')

    t, x, y, p = aedatUtils.loadaerdat(path)

    tI=50000 #50 ms

    totalImages = []
    totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI)
    font = {'family': 'serif',
            'color':  'red',
            'weight': 'normal',
            'size': 8,
    }
    off_set_text = 3
    plotMask = False

    if plotMask:
        fig, axarr = plt.subplots(1,2)
        textPlot = plt.text(0,0,"")
        axarr[0].set_title('neuromorphic image')
        axarr[1].set_title('watershed mask')
    else:
        fig,axarr = plt.subplots(1)
        textPlot = plt.text(0,0,"")
        handle = None

    count = []
    rects = []
    texts = []
    for f in totalImages:

        cleanFigure(rects,texts)

        watershedImage, mask, detection = segmentationUtils.watershed(f,'--avg --median --neuromorphic')
        watershedImage = watershedImage.astype(np.uint8)
        
        
        if plotMask:
            axarr[0].imshow(np.dstack([watershedImage,watershedImage,watershedImage]))
            axarr[1].imshow(mask)
        else:
            if handle is None:
                handle = plt.imshow(np.dstack([watershedImage,watershedImage,watershedImage]))                
            else:
                handle.set_data(np.dstack([watershedImage,watershedImage,watershedImage]))


        #draw the bounding boxes and write the classification
        for j in range(len(detection)):
            imageRoi = getROI(detection[j],watershedImage)
            predict = classify(imageRoi,model)
            #patches receive (y,x), length and width
            rect = patches.Rectangle((detection[j][1],detection[j][0]),detection[j][3],detection[j][2],linewidth=1,edgecolor='r',facecolor='none')
            plt.gca().add_patch(rect)
            text = plt.gca().text(detection[j][1], detection[j][0]-off_set_text, predict, fontdict = font)
            #the append is necessary to make the predictions not visible after the refresh of the frame
            rects.append(rect)
            texts.append(text)
        
        plt.pause(0.01)
        plt.draw()


def cleanFigure(rects,texts):
    for s in range(len(rects)):
        rects[s].set_visible(False)

    for s in range(len(texts)):
        texts[s].set_visible(False)

#this function get the original image and extract the ROI
def getROI(detection,image):
    dim = (128,128)
    crop_img = image[(detection[0]+1):detection[0]+detection[2],(detection[1]+1):detection[1]+detection[3]]
    crop_img = cv.resize(crop_img, dim, interpolation = cv.INTER_AREA)
    crop_img = crop_img.reshape(1, 128, 128, 1)
    return crop_img

def classify(image,model):
    resp, objectSet = classifierTools.predictObject(image, model)
    predict = objectSet[resp][1]
    return predict

if __name__ == "__main__":
	main()