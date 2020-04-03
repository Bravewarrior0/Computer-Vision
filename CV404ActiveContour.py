from os import listdir
from os.path import isfile , join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage, signal, interpolate
import math
import sys

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
        arr_replicated=np.tile(array,(3,3)) # copy one more column and one more row
        arr_replicated=arr_replicated[rows-1:2*rows+1, columns-1:2*columns+1]
    except: #1D vectors
        arr_replicated=np.tile(array,3) # copy one more column and one more row
        arr_replicated=arr_replicated[rows-1:2*rows+1] # truncate extraPlus rows 
    
    return arr_replicated



def get_8neighbors(pointsX, pointsY): #
    pointsOut= np.array([[pointsX, pointsY], [pointsX-1, pointsY], [pointsX+1, pointsY],\
                         [pointsX, pointsY-1], [pointsX, pointsY+1], [pointsX-1,pointsY-1],[pointsX-1,pointsY+1],\
                             [pointsX+1,pointsY-1], [pointsX+1,pointsY+1]   ] )
    return pointsOut

def norm(arr):
    return np.sqrt(arr[0]**2 + arr[1]**2 )
def compute_energy(pointsX, pointsY, alpha, beta, gamma, grad_normalized): #compute continuity energy
    #print(pointsX)
    #print(pointsY)
    newPointsX=np.zeros(pointsX.shape)
    newPointsY=np.zeros(pointsY.shape)  
    Emin =1000000000  
    c = np.zeros(len(pointsX))
    th = .7
    
    distance=0 #compute average distance 
    for ind in range(1,len(pointsX)-1): #loop over indices strarting from index 1(in the padded array)
        distance+=np.sqrt((pointsX[ind]-pointsX[ind+1]) ** 2+(pointsY[ind]-pointsY[ind+1]) ** 2)
    
    distance/=(len(pointsX)-2) # average contour distance 
    
    for ind in range(1,len(pointsX)-1): #compute elastic energy VECTOR #cover all Core points
    
        elas_En=np.zeros(9) 
        curv_En=np.zeros(9) 
        Grad_En=np.zeros(9) 
        
        allpoints_8=get_8neighbors(pointsX[ind],pointsY[ind]) # retrieve neighbors
        Grad_En =  ndimage.map_coordinates(grad_normalized, np.transpose(allpoints_8))#spline
        for neigh in range(len(allpoints_8)): #cover all neighbors           
            elas_En[neigh]=distance-(np.sqrt((allpoints_8[neigh,0]-pointsX[ind-1]) ** 2+\
                                (allpoints_8[neigh,1]-pointsY[ind-1]) ** 2)) #neighbors
           
            curv_En[neigh]=np.sqrt((2*allpoints_8[neigh,0]-pointsX[ind+1]-pointsX[ind-1]) **2 +\
                        (2*allpoints_8[neigh,1]-pointsY[ind+1]-pointsY[ind-1]) **2)            
            
        elas_En=alpha*elas_En
        curv_En=beta*curv_En
        Grad_En=gamma*Grad_En

        total_En=elas_En+curv_En+Grad_En
        EminTotal = min(total_En)
        if EminTotal<Emin:
            Emin = EminTotal  
            indMin=np.argmin(EminTotal)  

        newPointsX[ind]=allpoints_8[indMin,0]
        newPointsY[ind]=allpoints_8[indMin,1]

        #Beta relaxation algo
        Ui = [ pointsX[ind]-pointsX[ind-1], pointsY[ind]-pointsY[ind-1] ]
        Ui_1 = [ pointsX[ind+1]-pointsX[ind], pointsY[ind+1]-pointsY[ind] ]
        vi = [pointsX[ind], pointsY[ind]]
        vi_11 = [pointsX[ind-1], pointsY[ind-1]]
        vi_12 = [pointsX[ind+1], pointsY[ind+1]]
        CC = (Ui/norm(Ui))  - (Ui_1/norm(Ui_1)) 

        c[ind] = norm(CC)**2
        

        mag = np.hypot(vi[0],vi[1])
        if ((c[ind] > c[ind-1]) and (c[ind]> c[ind+1])) and (c[ind] > th) and ( mag> mag-th):
            beta = 0
 
       
        
    return newPointsX, newPointsY


