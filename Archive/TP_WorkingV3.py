import numpy as np
import cv2 as cv
import tkinter
import PIL
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
        data.boothOn = not data.boothOn
        photoBooth(data)
    elif key == ord('f'):
        for img in data.cvImages:
            cv.imshow(data.window,img.img)
            print (img.img)
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
    kernel = [[-1,-1,-1][-1,8,-1][-1,-1,-1]]
    (rows,cols,depth) = frame.shape
    target = np.empty([rows,cols,depth], dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            for d in range(depth):
                
                
                target[i,j,d]=convolve(frame,kernel,i,j,d)
    return target
        
def convolve(frame,kernel,i,j,d):
    chunk = getChunk(frame,i,j,d)
    sum = 0
    for a in range(3):
        for b in range(3):
            sum += chunk[a][b]*kernel[a][b]
    return sum
    
def getChunk(frame,i,j,d):
    w,h = (frame.shape[0],frame.shape[1])
    guide = [(-1,-1),(0,-1),(1,-1),
            (-1,0),(0,0),(1,0),
            (-1,1),(0,1),(1,1)]
    r = []
    for dx,dy in guide:
        if w>i+dx>=0 and h>i+dy>=0:
            result.append(frame[i+dx,j+dy,d])
        else:
            result.append(frame[i,j,d])
    
    chunk = [[0]*3]*3
    for i in range(2,-1):
        for j in range(2,-1):
            chunk[i][j]=r[-1]
            r.pop()
    reuturn chunk


    #cv.imwrite('%s.png'%name, frame)

def drawPhotoBooth(data):
    pass


# def kernelConvolve(data,kernel,frame):
#     image = copy.copy(frame)
#     h,w,d = image.shape
#     sum =
#     for j in range(h):
#         for i in range(w):
#             roi = frame[i-1:i+1,j-1:j+1]
#             
# 
# def pixelConvole(data,kernel,frame,i,j):
#     sum = 0
#     for fi in range ()


class cvImage():
    def __init__(self, data):
        self.img = getFrame(data)
        height, width, depth = self.img.shape
        self.height = height
        self.width = width
        self.maxDim = max(width,depth)
        self.bw = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.faces = []
        
    def outline(self):
        self.outline = cv.Canny(self.img, 50, 100)
    
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
    
    while(1):
        cv.imshow(data.winName,background)
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