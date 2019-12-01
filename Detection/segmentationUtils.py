
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class segmentationUtils:

    '''
    parameters:
        imagem - desired image
        options - optional parameter who is a string with the desired options.
            avaiable options:
                '--avg' - apply average filter to image
                '--median' - apply median filter to image
                '--neuromorphic' - is the declaration of neuromorphic image or else is a RGB image
    '''
    def watershed(imagem,options=None):

        opt = []
        if options != None:
            options = "".join(options.split())
            opt = options.split('--')

        if opt.__contains__('neuromorphic'):
            imagem = 255 * imagem # Now scale by 255
            img = imagem.astype(np.uint8)
            img = cv.cvtColor(imagem,cv.COLOR_RGB2GRAY)
        else:
            img = cv.cvtColor(imagem,cv.COLOR_RGB2GRAY)
           
        for i in range(len(opt)):
            if opt[i].__contains__('avg'):
                img = cv.blur(img, (5, 5))
            elif opt[i].__contains__('median'):
                img = cv.medianBlur(img, 5)
        
       
        
        ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        img2 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        markers = cv.watershed(img2,markers)
        img2[markers == -1] = [255,0,0]
   

        return img2, markers

    '''
    this method run a demo for watershed segmentation technique. 
    this will plot 4 images:
        - 1 standard image (original)
        - 1 standard image (watershed segmentation)
        - 1 neuromorphic image (original | probabily 100 ms event agroupation)
        - 1 neuromorphic image (watershed segmentation + filter of avg and median)
    '''
    def watershed_demo():
        neuromorphicImage = cv.imread('/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Detection/assets/testes/Mouse_71.png')
        standardImage = cv.imread('/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Detection/assets/testes/standard_mouse.jpeg')
        watershedNeuromorphicImage, neuromorphicMask = segmentationUtils.watershed(neuromorphicImage,'--avg --median --neuromorphic')
        watershedStandardImage, standardMask = segmentationUtils.watershed(standardImage)

        f, axarr = plt.subplots(2,3)
        axarr[0,0].set_title('neuromorphic image [original]')
        axarr[0,0].imshow(neuromorphicImage)

        axarr[0,1].set_title('neuromorphic image [watershed]')
        axarr[0,1].imshow(watershedNeuromorphicImage)

        axarr[0,2].set_title('neuromorphic - mask')
        axarr[0,2].imshow(neuromorphicMask)

        axarr[1,0].set_title('standard image [original]')
        axarr[1,0].imshow(standardImage)

        axarr[1,1].set_title('standard image [watershed]')
        axarr[1,1].imshow(watershedStandardImage)

        axarr[1,2].set_title('standard - mask')
        axarr[1,2].imshow(standardMask)

        
        plt.show()



       