
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
import matplotlib.animation as animation
from segmentationUtils import segmentationUtils

#path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/standardized data/Mouse.aedat'
path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/two_objects.aedat'

t, x, y, p = aedatUtils.loadaerdat(path)

tI=50000 #50 ms

totalImages = []
totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI)

fig = plt.figure()
ims = []
handle = None
for f in totalImages:
    
    
    watershedImage, mask = segmentationUtils.watershed(f,'--avg --median --neuromorphic')
    
    im=watershedImage
    
    im = im.astype(np.uint8)
    if handle is None:
        handle = plt.imshow(np.dstack([im,im,im]))
    else:
        handle.set_data(np.dstack([im,im,im]))

    plt.pause(0.01)
    plt.draw()


print('fim')

