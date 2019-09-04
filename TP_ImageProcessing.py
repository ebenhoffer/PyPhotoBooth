#Eben Hoffer
#15-112, Section I
# ehoffer

##### Image filtering/processing functions for photo booth "show mode"

import numpy as np
import cv2 as cv
import copy

def outline(data,frame):
    #edge detection via kernel convolution
    if len(frame.shape)<3:
        return False
    #noise reduction first, or it's a mess
    blurFrame = gaussianBlur(data,frame)
    frameBW = cv.cvtColor(blurFrame,cv.COLOR_BGR2GRAY)
    kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    (rows,cols,depth) = frame.shape
    #convolve kernel with every pixel
    target = np.empty([rows,cols], dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            target[i,j]=convolve(frameBW,kernel,i,j)
    return target
        
    
def convolve(frame,kernel,i,j,d=None):
    #colvolution algorithm- add kern cells to surrounding pixels, etc
    kernSize = len(kernel)
    #get the chunk of the image you're working with, to simplify it
    chunk = getChunk(frame,i,j,d,kernSize)
    sum = 0
    for a in range(kernSize):
        for b in range(kernSize):
            if d==None:
                sum += chunk[a][b]*kernel[a][b]
            else:
                sum += chunk[a][b][0]*kernel[a][b]
    return sum    

def gkern(l=5, sig=1.):
    #creates gaussian kernel with side length l and a sigma of sig
    #taken mostly intact from James199, stackOverFlow
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    kernel = kernel / np.sum(kernel)
    return np.ndarray.tolist(kernel)

def gaussianBlur(data,frame):
    if len(frame.shape)<3:
        return False
    kernel = gkern(data.blurMode,1.5)
    (rows,cols,depth) = frame.shape
    target = np.empty([rows,cols,depth], dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            for d in range(depth):
                target[i,j,d]=convolve(frame,kernel,i,j,d)
    return target
        

def getChunk(frame,i,j,d=None, kernSize=5):
    #gets a chunk of image of side-length sams as kernel
    h,w = (frame.shape[0]-1,frame.shape[1]-1)
    k = kernSize//2
    if d!=None:  
        #first see if you're an edge case. If not, easy.
        if i>=k and j>=k and i<h-k and j<w-k:
            chunkArray = frame[i-k:i+k+1,j-k:j+k+1,d:d+1]
        else: 
        #if you are on an edge, get as much of the chunk as possible...
            rowStart,colStart = -1,-1
            editFrame = copy.deepcopy(frame)
            if i<k: 
                editFrame = editFrame[:i+k+1,:,:]
                rowStart = 1
            elif i>(h-k): 
                editFrame = editFrame[i-k:,:,:]
                rowStart = 0
            else:
                editFrame = editFrame[i-k:i+k+1,:,:]
            if j<k: 
                editFrame = editFrame[:,:j+k+1,:]
                colStart = 1
            elif j>(w-k): 
                editFrame = editFrame[:,j-k:,:]
                colStart = 0
            else:
                editFrame = editFrame[:,j-k:j+k+1,:]
            #and send it to be mirrored up to target dimensions
            #rowstart defines which direction you mirror from to stretch kern
            editFrame=np.ndarray.tolist(editFrame)
            editFrame=chunkMirror(editFrame,kernSize,rowStart,colStart)
            return editFrame
    else:
        #case for bw (2d) matrices
        if i>0 and j>0 and i<h and j<w:
            chunkArray = frame[i-k:i+k+1,j-k:j+k+1]
        else:
            return [[0]*kernSize]*kernSize
    chunk = np.ndarray.tolist(chunkArray)
    return chunk

def chunkMirror(editFrame,kernSize,rowStart,colStart):
    #copy rows to empty space first
    if rowStart==1:
        row = 0
        while len(editFrame)<kernSize:
            editFrame.insert(0, copy.copy(editFrame[row]))
            row += 2
    elif rowStart==0:
        row = len(editFrame)-1
        while len(editFrame)<kernSize:
            editFrame.append(copy.copy(editFrame[row]))
            row -= 2
    #then copy columns into empty space
    if colStart==1:
        col = 0
        while len(editFrame[0])<kernSize:
            for r in editFrame:
                r.insert(0,copy.copy(r[col]))
            col += 2
    elif colStart==0:
        col = len(editFrame[0])-1
        while len(editFrame[0])<kernSize:
            for r in editFrame:
                r.append(copy.copy(r[col]))
            col -= 2
    return editFrame

def normalize(data,frame):
    #normalization type dispatcher
    if len(frame.shape)<3:
        return False
    if data.normalMode == 'hi':
        return incautiousNormalize(data,frame)
    else:
        return cautiousNormalize(data,frame)
        
def incautiousNormalize(data,frame):
    #maximize dynaic range between colors in each pixel - smallest color
    #val goes to zero, highest goes to 255, middle scales
    h,w = (frame.shape[0]-1,frame.shape[1]-1)
    editFrame = copy.deepcopy(frame)
    for i in range (h):
        for j in range(w):
            G = int(editFrame[i,j,0])
            B = int(editFrame[i,j,1])
            R = int(editFrame[i,j,2])
            mx = max(G,B,R)
            mn = min(G,B,R)
            if mx != mn:
                r = 255/(mx-mn)
                G = (G-mn) * r
                R = (R-mn) * r
                B = (B-mn) * r
            a= np.array([G,B,R], dtype='uint8')
            editFrame[i,j] = a
    return editFrame
    
def cautiousNormalize(data,frame):
    #a real actual normalization, per total of pixel colors, flattening color
    h,w = (frame.shape[0]-1,frame.shape[1]-1)
    editFrame = copy.deepcopy(frame)
    for i in range (h):
        for j in range(w):
            G = int(editFrame[i,j,0])
            B = int(editFrame[i,j,1])
            R = int(editFrame[i,j,2])
            total = G+B+R
            if total != 0:
                G = (G/total * 255)
                R = R/total * 255
                B = B/total * 255
            a= np.array([G,B,R], dtype='uint8')
            editFrame[i,j] = a
    return editFrame