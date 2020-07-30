import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy 
from openAEDAT import aedatUtils
from segmentationUtils import segmentationUtils


def main():
        
    #caso você queira que os retangulos sejam desenhados na tela:
    rectFlag = False


    #Caminho para o arquivo .aedat
    #path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/standardized data/banana_1.aedat'
    path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/standardized data/anything.aedat'
    #carregando o arquivo aedat
    t, x, y, p = aedatUtils.loadaerdat(path)
    
    #determinando o intervalo de tempo para agrupamento dos eventos
    tI=33000 #33 ms

    #carregando todos os eventos agrupados em frames
    totalImages = []
    totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI)

    #variável que armazena as bounding boxes
    detections = []

    fig,axarr = plt.subplots(1)
    textPlot = plt.text(0,0,"")
    handle = None
    rects = []

    teste = np.zeros_like(totalImages[0])

    for f in totalImages:
    
        f = f.astype(np.uint8)
        imagem = copy.deepcopy(f)
        
        watershedImage, mask, detection, opening, sure_fg, sure_bg, markers = segmentationUtils.watershed(imagem,'--neuromorphic',minimumSizeBox=0.5,smallBBFilter=True,centroidDistanceFilter = True, mergeOverlapingDetectionsFilter = True,flagCloserToCenter=True)
        #detection contém todas as bounding box
        #cada bounding box tem o formato:
            # x -- detection[0]
            # y -- detection[1]
            # width -- detection[2]
            # length -- detection[3]
        
       
        
       
        cleanFigure(rects)
        if rectFlag: 
            for j in range(len(detection)):  
                #patches receive (y,x), length and width
                rect = patches.Rectangle((detection[j][1],detection[j][0]),detection[j][3],detection[j][2],linewidth=1,edgecolor='r',facecolor='none')
                plt.gca().add_patch(rect)
                #the append is necessary to make the predictions not visible after the refresh of the frame
                rects.append(rect)

        teste = segmentationUtils.getROI(detection,f)
        if handle is None:      
            handle = plt.imshow(np.dstack([teste,teste,teste]))                
        else:
            handle.set_data(np.dstack([teste,teste,teste]))
    
        plt.pause(tI/1000000)
        plt.draw()
    
    
        


def cleanFigure(rects = [],texts = []):
    for s in range(len(rects)):
        rects[s].set_visible(False)

    for s in range(len(texts)):
        texts[s].set_visible(False)



if __name__ == "__main__":
	main()