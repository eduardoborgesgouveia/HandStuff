
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
from classifierTools import classifierTools
import matplotlib.animation as animation
from segmentationUtils import segmentationUtils
import matplotlib.patches as patches

# path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/standardized data/Mug.aedat'
#path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/two_objects.aedat'
path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/key.aedat'
#path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/key_2.aedat'
#path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/multi_objects_2.aedat'

model = classifierTools.openModel('model/model.json',
							        'model/model.h5')
dim = (128,128)
t, x, y, p = aedatUtils.loadaerdat(path)

tI=50000 #50 ms

totalImages = []
totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI)

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
for f in totalImages:
    for txt in fig.texts:
        txt.remove()

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


    for a in range(len(detection)):
        crop_img = watershedImage[(detection[a][0]+1):detection[a][0]+detection[a][2],(detection[a][1]+1):detection[a][1]+detection[a][3]]
        crop_img = cv.resize(crop_img, dim, interpolation = cv.INTER_AREA)
        crop_img = crop_img.reshape(1, 128, 128, 1)
        resp, objectSet = classifierTools.predictObject(crop_img, model)
        predict = objectSet[resp][1]
        #plt.text(detection[a][0], detection[a][1], predict, fontsize=12)

    
    plt.pause(0.01)
    plt.draw()


