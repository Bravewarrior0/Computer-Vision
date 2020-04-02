from os import listdir
from os.path import isfile , join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage, signal, interpolate
import math
import sys

def compute_energy(pointsX, pointsY, alpha, beta, gamma, grad_normalized): #compute continuity energy
    #print(pointsX)
    #print(pointsY)
    newPointsX=np.zeros(pointsX.shape)
    newPointsY=np.zeros(pointsY.shape)    
    
    distance=0 #compute average distance 
    print(pointsX) #padded to include before and after
    #print(len(pointsX))# length of padded points
    print(len(pointsX)-2)# length of points
    #for ind in pointsX[1:len(pointsX)-1]: #loop over points values
    for ind in range(1,len(pointsX)-1): #loop over indices strarting from index 1(in the padded array)
        print(ind)
        #print('\n')
        distance+=np.sqrt((pointsX[ind]-pointsX[ind+1]) ** 2+(pointsY[ind]-pointsY[ind+1]) ** 2)
    
    distance/=(len(pointsX)-2) # average contour distance 
    
    for ind in range(1,len(pointsX)-1): #compute elastic energy VECTOR #cover all Core points
        # having circular previous and circ next
        #print(pointsX[ind])        
        #if False: #block comment
        elas_En=np.zeros(5) # For the Current contour point and its 4-neighbors
        curv_En=np.zeros(5) # For the Current contour point and its 4-neighbors
        Grad_En=np.zeros(5) # For the Current contour point and its 4-neighbors
        
        allpoints_4=get_4neighbors(pointsX[ind],pointsY[ind]) # retrieve neighbors
        #print(allpoints_4)
        Grad_En =  ndimage.map_coordinates(grad_normalized, np.transpose(allpoints_4))#spline
        for neigh in range(len(allpoints_4)): #cover all neighbors           
            elas_En[neigh]=distance-(np.sqrt((allpoints_4[neigh,0]-pointsX[ind-1]) ** 2+\
                                (allpoints_4[neigh,1]-pointsY[ind-1]) ** 2)) #neighbors
            #print(allpoints_4[neigh,0],allpoints_4[neigh,1],pointsX[ind-1],pointsY[ind-1],\
            #     pointsX[ind+1],pointsY[ind+1])
            #print(elas_En)
            curv_En[neigh]=np.sqrt((2*allpoints_4[neigh,0]-pointsX[ind+1]-pointsX[ind-1]) **2 +\
                        (2*allpoints_4[neigh,1]-pointsY[ind+1]-pointsY[ind-1]) **2)            
            
        elas_En=alpha*elas_En
        curv_En=beta*curv_En
        Grad_En=gamma*Grad_En

        total_En=elas_En+curv_En+Grad_En
        indMin=np.argmin(total_En)
        #print(pointsX[ind], pointsY[ind])
        newPointsX[ind]=allpoints_4[indMin,0]
        newPointsY[ind]=allpoints_4[indMin,1]
        
        #print(pointsX[ind], pointsY[ind])
        #print(newPointsX[ind], newPointsY[ind])

        #print(allpoints_4)
        #print(total_En) 
        #print(indMin)
        #print(len(grad_normalized))        
        
        #print('\n')
    #print(distance)
    return newPointsX, newPointsY


def gaussian_Filter_AC(sigma, shape):
    filter=np.zeros(shape)
    filter[1,1]=1
    return ndimage.gaussian_filter(filter, sigma)


def normalize(array, newMin, newMax):
    minArr=array.min()
    maxArr=array.max()
    return ((array-minArr)/(maxArr-minArr))*(newMax-newMin)+newMin    



def circ_replicate(array): #repeat put last as first, and first as last again (circ)
    rows=len(array)
    try:
        columns=len(array[0])
        #print(rows)
        #print(columns)
        arr_replicated=np.tile(array,(3,3)) # copy one more column and one more row
        arr_replicated=arr_replicated[rows-1:2*rows+1, columns-1:2*columns+1]
    except: #1D vectors
        arr_replicated=np.tile(array,3) # copy one more column and one more row
        arr_replicated=arr_replicated[rows-1:2*rows+1] # truncate extraPlus rows 
    
    return arr_replicated



def get_4neighbors(pointsX, pointsY): #
    pointsOut= np.array([[pointsX, pointsY], [pointsX-1, pointsY], [pointsX+1, pointsY],\
                         [pointsX, pointsY-1], [pointsX, pointsY+1]    ] )
    return pointsOut