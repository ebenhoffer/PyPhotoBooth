import numpy as np
import cv2 as cv
import tkinter
import PIL
import copy
from tkinter import *
from PIL import Image
from PIL import ImageTk
import time
eyeCheck = cv.CascadeClassifier('haarcascade_eye.xml')
faceCheck = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def init(data):
    data.canvas = np.zeros((512,512,3), np.uint8)
    data.winName = 'Photo Booth'
    data.window = cv.namedWindow(data.winName)
    data.cap = cv.VideoCapture(0)
    data.cvBG = None
    data.cvImages = []
    data.boothOn = False
    data.boothSession = 0
    data.fps = int(1000/30)
    data.time = time.time()
    if data.cap.isOpened():
        print('capture open')
    
def timerFired(data, key):
    return keyPressedMain(data,key)

def keyPressedMain(data, key):
    if key == ord('c'):
        photo1 = cvImage(data)
        photo1.faceGrab()
        data.cvImages.append(photo1)
    elif key == ord('d'):
        print('booth')
        data.boothOn = not data.boothOn
        photoBooth(data)
    elif key == ord('f'):
        for img in data.cvImages:
            cv.destroyAllWindows()
            cv.imshow(data.winName, img.img)
    elif key == ord('o'):
        cv.destroyAllWindows()
        for img in data.cvImages:
            newImg = outline(data,img.img)
            cv.imshow(data.winName,newImg)
    elif key == ord('n'):
        print('normal')
        for img in data.cvImages:
            newImg = normalize(data,img.img)
            cv.destroyAllWindows()
            cv.waitKey(0)
            cv.imshow(data.winName,newImg)
    elif key == 27:
        return False

def mousePressedAll(event, x,y,flags,param):
    pass


def getFrame(data):
    if (data.cap.isOpened()):
        ret, frame = data.cap.read()
        if ret:
            frame = cv.flip(frame, 1)
        else: return
    print('frame grabbed')
    return frame

    
def fitToWindow(data, frame):
    #frame is square, so frame should be stretched vert & cropped horiz
    fh,fw,fd = frame.shape
    if fh != data.height:
        scale = data.height/fh
        newW,newH = fw*scale,fh*scale
        frame = cv.resize(frame, (int(newW),int(newH)))
    if fw > data.width:
        x = (frame.shape[1]-data.width)//2
        frame = frame[0:frame.shape[0], x:x+data.width]
    frame = cv.flip(frame, 1)
    return frame
    
def redrawAll(canvas, data):
    if data.facesOnly:
        for cvImage in data.cvImages:
            for face in cvImage.outputFaces:
                canvas.create_image(0,0,image=face, anchor = NW)
    else:
        for cvImage in data.cvImages:
            output = cvImage.output
            canvas.create_image(0,0,image=output, anchor = NW)

def photoBooth(data):
    while data.boothOn:
        ret, frame = data.cap.read()
        if ret:
            frame = fitToWindow(data,frame)
            cv.imshow(data.winName, frame)
            drawPhotoBooth(data)
            keyPressedMain(data, cv.waitKey(data.fps))
        else: break

def outline(data,frame):
    blurFrame = gaussianBlur(data,frame)
    print('blur complete')
    cv.destroyAllWindows()
    cv.waitKey(1)
    cv.imshow(data.window,blurFrame)
    frameBW = cv.cvtColor(blurFrame,cv.COLOR_BGR2GRAY)
    kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    (rows,cols,depth) = frame.shape
    target = np.empty([rows,cols], dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            target[i,j]=convolve(frameBW,kernel,i,j)
    return target
        
# def bwConvolve(frame,kernel,i,j):
#     chunk = getChunk(frame,i,j)
#     sum = 0
#     for a in range(3):
#         for b in range(3):
#             sum += chunk[a][b]*kernel[a][b]
#     return sum
    
def convolve(frame,kernel,i,j,d=None):
    kernSize = len(kernel)
    chunk = getChunk(frame,i,j,d,kernSize)
    sum = 0
    for a in range(kernSize):
        for b in range(kernSize):
            if d==None:
                sum += chunk[a][b]*kernel[a][b]
            else:
                sum += chunk[a][b][0]*kernel[a][b]
    return sum    

def gkern(l=3, sig=1.):
    #creates gaussian kernel with side length l and a sigma of sig
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    kernel = kernel / np.sum(kernel)
    return np.ndarray.tolist(kernel)

def gaussianBlur(data,frame):
    kernel = gkern(5,1.5)
    (rows,cols,depth) = frame.shape
    target = np.empty([rows,cols,depth], dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            for d in range(depth):
                target[i,j,d]=convolve(frame,kernel,i,j,d)
    return target
        

def getChunk(frame,i,j,d=None, kernSize=3):
    h,w = (frame.shape[0]-1,frame.shape[1]-1)
    k = kernSize//2
    if d!=None:  
        if i>=k and j>=k and i<h-k and j<w-k:
            chunkArray = frame[i-k:i+k+1,j-k:j+k+1,d:d+1]
        else: 
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
            editFrame=np.ndarray.tolist(editFrame)
            editFrame=chunkMirror(editFrame,kernSize,rowStart,colStart)
            return editFrame
    else:
        if i>0 and j>0 and i<h and j<w:
            chunkArray = frame[i-k:i+k+1,j-k:j+k+1]
        else:
            return [[0]*kernSize]*kernSize
    chunk = np.ndarray.tolist(chunkArray)
    return chunk

def chunkMirror(editFrame,kernSize,rowStart,colStart):
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

    #cv.imwrite('%s.png'%name, frame)

def normalize(data,frame):
    h,w = (frame.shape[0]-1,frame.shape[1]-1)
    editFrame = copy.deepcopy(frame)
    for i in range (h):
        for j in range(w):
            G = int(editFrame[i,j,0])
            B = int(editFrame[i,j,1])
            R = int(editFrame[i,j,2])
            # print(G,B,R)
            total = G+B+R
            if total != 0:
                G = (G/total * 255)
                R = R/total * 255
                B = B/total * 255
            a= np.array([G,B,R], dtype='uint8')
            print(a)
            editFrame[i,j] = a
        print('normalized')
        return editFrame
        
    


def drawPhotoBooth(data):
    pass


class cvImage():
    def __init__(self, data):
        self.img = getFrame(data)
        self.img = fitToWindow(data,self.img)
        height, width, depth = self.img.shape
        self.height = height
        self.width = width
        self.maxDim = max(width,depth)
        self.bw = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.faces = []
        
    def outline(self):
        pass
    
    def faceGrab(self):
        faces = faceCheck.detectMultiScale(self.bw, 1.3, 5)
        for (x,y,w,h) in faces:
            faceBW = self.bw[y:y+h, x:x+w]
            faceColor = self.img[y:y+h, x:x+w]
            self.faces.append(faceColor)
            
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)

class photoBoothSet(cvImage):
    def __init__(self,data,frame):
        super().__init__(self,data)
        data.boothSession += 1
        self.name = 'boothSession_%d' %data.boothSession
        self.images = []
        self.images.append(frame)
        frameCount = 1
    def addImage(data, frame):
        self.images.append(frame)
        frameCount+=1

class confessionalSet(cvImage):
    def __init__(self,data,name):
        super().__init__(self,data)
        self.name = name



def runcv(width=512, height=512):
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    delay = 1
    init(data)
    
    background = np.zeros((width,height,3), np.uint8)
    cv.setMouseCallback(data.winName,mousePressedAll)
    
    cv.imshow(data.winName,background)
    while(1):
        
        if timerFired(data, cv.waitKey(delay)) == False:
            break
    
    cv.destroyAllWindows()
    cv.waitKey(1)
    print('bye!')


runcv(720,720)






def cvToTkinter(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(image))
    print('transformed to tkinter')
    return (img)