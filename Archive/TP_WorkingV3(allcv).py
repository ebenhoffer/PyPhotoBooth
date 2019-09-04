import numpy as np
import cv2 as cv
import tkinter
import PIL
from tkinter import *
from PIL import Image
from PIL import ImageTk
eyeCheck = cv.CascadeClassifier('haarcascade_eye.xml')
faceCheck = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def run(width=512, height=512):
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.canvas = np.zeros((width,height,3), np.uint8)
    init(data)
    
    def redrawAllWrapper(data):
        redrawAll(data)
    def mousePressedWrapper(event, data):
        mousePressed(data)
        redrawAllWrapper(data)
    def keyPressedWrapper(event, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(data):
        timerFired(data)
        redrawAllWrapper(data)
        cv.waitKey(1)
        timerFiredWrapper(data)
    timerFiredWrapper(data)
    
def init(data):
    data.window = cv.namedWindow('Photo Booth')
    data.cap = cv.VideoCapture(0)
    data.cvBG = None
    data.cvImages = []
    data.display = False
    data.fps = int(1000/30)
    if data.cap.isOpened():
        print('capture open')
    
def timerFired(data):
    pass


def keyPressed(event, data):
    k = cv.waitKey(1)
        
    if k == ord('c'):
        name = 'joan'
        photo1 = cvImage(name,data)
        data.cvImages.append(photo1)
    elif k == ord('d'):
        displayVideo(data)

def mousePressed(event, data):
    pass


def getFrame(data, name):
    if (data.cap.isOpened()):
        ret, frame = data.cap.read()
        if ret:
            frame = cv.flip(frame, 1)
        else: return
    print('frame grabbed')
    cv.imwrite('%s.png'%name, frame)
    return frame
    
def displayVideo(data):
    data.display = True
    while data.display:
        ret, frame = data.cap.read()
        if ret:
            frame = cv.flip(frame, 1)
            cv.imshow(data.window, frame)
            k = cv.waitKey(data.fps)
            if k == ord('d'):
                break
        else: break
    data.display = False
    cv.destroyAllWindows()
    cv.waitKey(1)
    
def redrawAll(data):
    cv.imshow(data.window, data.canvas)
    for cvImage in data.cvImages:
        output = cvImage.output
        cv.imshow(output)


class cvImage():
    def __init__(self, name, data):
        self.name = name
        self.img = getFrame(data,'joan')
        height, width, depth = self.img.shape
        self.height = height
        self.width = width
        self.maxDim = max(width,depth)
        self.bw = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.faces = []
        
        self.output = cvToTkinter(data, self.img)
        
    def outline(self):
        self.outline = cv.Canny(self.img, 50, 100)
    
    
    def faceGrab(self):
        faces = faceCheck.detectMultiScale(self.bw, 1.3, 5)
        for (x,y,w,h) in faces:
            faceBW = self.bw[y:y+h, x:x+w]
            faceColor = self.img[y:y+h, x:x+w]
            self.faces.append(faceColor)
            
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)




def cvToTkinter(data,image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(image))
    print('transformed to tkinter')
    return (img)


run()