
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from iou import iou
from classifierTools import classifierTools
from skimage.morphology import erosion, dilation, opening, closing, white_tophat,square,convex_hull_image
from filterUtils import filterUtils

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
            img = imagem.astype(np.uint8)
            img[img == 255] = 0
            if len(img.shape) == 3:
                img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        else:
            img = cv.cvtColor(imagem,cv.COLOR_RGB2GRAY)
                
        ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        # noise removal
        kernel = np.ones((1,1),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_ELLIPSE,kernel, iterations = 3)
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=5)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,0)
        ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),255,0)
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

        detections = segmentationUtils.makeRectDetection(markers)        
        #imagem = segmentationUtils.drawRect(imagem,detections)
        detections = segmentationUtils.getCoordinatesFromPoints(detections)
        return imagem, markers, detections

    '''
    this method was make in order to receive a mask from multiple detection using the watershed method
    and make a rectangular bounding box ao redor of the detections.
    '''
    def makeRectDetection(mask):
        #make sure that the edges of the image is not being marked
        mask[0,:] = 1
        mask[:,0] = 1
        mask[mask.shape[0]-1,:] = 1
        mask[:, mask.shape[1]-1] = 1
        unique = np.unique(mask)
        unique = unique[unique != -1]
        unique = unique[unique != 1]
        objects = []
        for i in range(len(unique)):
            positions = np.where(mask == unique[i])
            x = min(positions[0])
            y = min(positions[1])
            lastX = max(positions[0])
            lastY = max(positions[1])
            width = lastX - x
            height = lastY - y 
            objects.append([x, y, width, height])
        print(len(objects))
        if len(objects)<5:
            objects = segmentationUtils.getPointsFromCoordinates(objects)
            objects = segmentationUtils.filterDetections(objects)
        else:
            objects = []

        return objects


    def filterDetections(detections):
        flag = True
        retorno = detections[:]
        while(flag):
            flag, pos = segmentationUtils.checkIntersec(retorno)
            retorno = segmentationUtils.mergeDetections(retorno,pos)
        return retorno

    def checkIntersec(coordinates):
        count = len(coordinates)
        register = 0
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if j > i:
                    area = iou.bb_intersection_over_union(coordinates[i],coordinates[j])
                    if area > 0.0 and area != 1.0:
                        return True, [i,j]
                    
        return False, None

    def drawRect(img, detections,lineWidth=None):
        bbColor = 8
        detections = segmentationUtils.getCoordinatesFromPoints(detections)
        if lineWidth == None:
            lineWidth = round(0.01*img.shape[0])
        if len(img.shape) == 3:
            bbColor = [255,0,0]
        for i in range(len(detections)):
            img[detections[i][0],detections[i][1]] = bbColor
            img[detections[i][0]:detections[i][0]+lineWidth,detections[i][1]:(detections[i][1]+detections[i][3])] = bbColor
            img[detections[i][0]:(detections[i][0]+detections[i][2]),detections[i][1]:detections[i][1]+lineWidth] = bbColor
            img[(detections[i][0]+detections[i][2]):(detections[i][0]+detections[i][2])+lineWidth,detections[i][1]:(detections[i][1]+detections[i][3])] = bbColor
            img[detections[i][0]:(detections[i][0]+detections[i][2]),(detections[i][1]+detections[i][3]):(detections[i][1]+detections[i][3])+lineWidth] = bbColor
        return img

    '''
        If one or more rectangular detections has a IOU the bounding boxes are merged and
    became just one
    '''
    def mergeDetections(detections,pos):
        retorno = detections
        if pos != None:
            coordinates = detections[:]
            retorno = []
            X1 = max(coordinates[pos[0]][0],coordinates[pos[0]][2],coordinates[pos[1]][0],coordinates[pos[1]][2])
            X2 = min(coordinates[pos[0]][0],coordinates[pos[0]][2],coordinates[pos[1]][0],coordinates[pos[1]][2])
            Y1 = max(coordinates[pos[0]][1],coordinates[pos[0]][3],coordinates[pos[1]][1],coordinates[pos[1]][3])
            Y2 = min(coordinates[pos[0]][1],coordinates[pos[0]][3],coordinates[pos[1]][1],coordinates[pos[1]][3])
            width = X1 - X2        
            height = Y1 - Y2

            coordinates.remove(detections[pos[0]])
            coordinates.remove(detections[pos[1]])

            retorno = coordinates
            retorno.append([X2, Y2, X1, Y1])                   
        return retorno

    def getPointsFromCoordinates(detections):
        objects = []
        for i in range(len(detections)):
            x1 = detections[i][0]
            y1 = detections[i][1]
            x2 = detections[i][0] + detections[i][2]
            y2 = detections[i][1] + detections[i][3]
            objects.append([x1, y1, x2, y2])
        return objects
    def getCoordinatesFromPoints(detections):
        objects = []
        for i in range(len(detections)):
            if detections[i][2] - detections[i][0] > 1 and detections[i][3] - detections[i][1] > 1:
                x1 = detections[i][0]
                y1 = detections[i][1]
                width = detections[i][2] - x1
                lenght = detections[i][3] - y1
                objects.append([x1, y1, width, lenght])
        return objects


    '''
    this method run a demo for watershed segmentation technique. 
    this will plot 4 images:
        - 1 standard image (original)
        - 1 standard image (watershed segmentation)
        - 1 neuromorphic image (original | probabily 100 ms event agroupation)
        - 1 neuromorphic image (watershed segmentation + filter of avg and median)
    '''
    def watershed_demo():
        font = {'family': 'serif',
                'color':  'red',
                'weight': 'normal',
                'size': 8,
        }
        off_set_text = 3

        dim = (128,128)
        
        neuromorphicImage = cv.imread('/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Detection/assets/testes/Mouse_22.png')
        standardImage = cv.imread('/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Detection/assets/testes/standard_mouse.jpeg')
        watershedStandardImage, standardMask,standardDetection = segmentationUtils.watershed(standardImage)
        
        neuromorphicImage = filterUtils.avg(neuromorphicImage)
        neuromorphicImage = filterUtils.median(neuromorphicImage)

        watershedNeuromorphicImage, neuromorphicMask,neuromorphicDetection = segmentationUtils.watershed(neuromorphicImage,'--neuromorphic')
        
      
        f, axarr = plt.subplots(2,3)
        axarr[0,0].set_title('neuromorphic image [original]')
        axarr[0,0].imshow(neuromorphicImage)

        axarr[0,1].set_title('neuromorphic - mask')
        axarr[0,1].imshow(neuromorphicMask)

        crop_neuromorphic = neuromorphicImage[neuromorphicDetection[0][0]+1:neuromorphicDetection[0][0]+neuromorphicDetection[0][2] , neuromorphicDetection[0][1]+1:neuromorphicDetection[0][1]+neuromorphicDetection[0][3]]

        axarr[0,2].set_title('croped bounding box')
        axarr[0,2].imshow(crop_neuromorphic)


        axarr[1,0].set_title('standard image [original]')
        axarr[1,0].imshow(standardImage)

        axarr[1,1].set_title('standard - mask')
        axarr[1,1].imshow(standardMask)

        crop_standard = standardImage[standardDetection[0][0]+(round(0.01*standardImage.shape[0])):standardDetection[0][0]+standardDetection[0][2] , standardDetection[0][1]+(round(0.01*standardImage.shape[0])):standardDetection[0][1]+standardDetection[0][3]]

        axarr[1,2].set_title('croped bounding box')
        axarr[1,2].imshow(crop_standard)
        

        model = classifierTools.openModel('model/model.json',
							              'model/model.h5')


        crop_img = cv.resize(crop_neuromorphic, dim, interpolation = cv.INTER_AREA)
        crop_img = cv.cvtColor(crop_img,cv.COLOR_RGB2GRAY)
        crop_img = crop_img.reshape(1, 128, 128, 1)
        resp, objectSet = classifierTools.predictObject(crop_img, model)
        predict = objectSet[resp][1]
        axarr[0,0].text(neuromorphicDetection[0][1], neuromorphicDetection[0][0]-off_set_text, predict, fontdict = font)

        plt.show()

if __name__ == "__main__":
	segmentationUtils.watershed_demo()