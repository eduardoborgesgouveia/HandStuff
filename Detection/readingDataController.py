
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
import matplotlib.animation as animation
from segmentationUtils import segmentationUtils

path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/standardized data/Mouse.aedat'

t, x, y, p = aedatUtils.loadaerdat(path)

tI=100000 #50 ms

totalImages = []
totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI)

fig = plt.figure()
ims = []
handle = None
for f in totalImages:
   
    watershedImage, mask = segmentationUtils.watershed(f,'--avg --median --neuromorphic')
    
    im=watershedImage
    if handle is None:
        handle = plt.imshow(im)
    else:
        handle.set_data(im)

    plt.pause(.01)
    plt.draw()


print('fim')

