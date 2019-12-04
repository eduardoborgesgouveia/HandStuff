
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from iou import iou

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
            img = 255 * imagem # Now scale by 255
            img = img.astype(np.uint8)
            if len(img.shape) == 3:
                img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
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
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 1)
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=20)
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

        imTeste = segmentationUtils.drawRect(imagem,detections)

        return imTeste, markers

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

        objects = segmentationUtils.getPointsFromCoordinates(objects)
        objects = segmentationUtils.filterDetections(objects)

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
                    else:
                        register += 1
                        if register == count:
                            return False, None
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
    # def mergeDetections(detections):
    #     coordinates = detections
    #     retorno = []
    #     if len(detections) == 1:
    #         return detections
    #     else:
    #         for i in range(len(detections)):
    #             for j in range(len(detections)):
    #                 if j > i:
    #                     area = iou.bb_intersection_over_union(coordinates[i],coordinates[j])
    #                     if area > 0.0:
    #                         X1 = max(coordinates[i][0],coordinates[i][2],coordinates[j][0],coordinates[j][2])
    #                         X2 = min(coordinates[i][0],coordinates[i][2],coordinates[j][0],coordinates[j][2])
    #                         Y1 = max(coordinates[i][1],coordinates[i][3],coordinates[j][1],coordinates[j][3])
    #                         Y2 = min(coordinates[i][1],coordinates[i][3],coordinates[j][1],coordinates[j][3])
    #                         width = X1 - X2
    #                         height = Y1 - Y2
    #                         retorno.append([X2, Y2, width, height])
    #                     else:
    #                         retorno.append(detections[i])
    #     return retorno

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



       