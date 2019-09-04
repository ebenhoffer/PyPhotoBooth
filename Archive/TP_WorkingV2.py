import numpy as np
import cv2 as cv
import tkinter
import PIL
from tkinter import *
from PIL import Image
from PIL import ImageTk
eyeCheck = cv.CascadeClassifier('haarcascade_eye.xml')
faceCheck = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def init(data):
    data.canvas = np.zeros((512,512,3), np.uint8)
    data.winName = 'Photo Booth'
    data.window = cv.namedWindow(data.winName)
    data.cap = cv.VideoCapture(0)
    data.cvBG = None
    data.cvImages = []
    data.display = False
    data.facesOnly = False
    data.fps = int(1000/30)
    if data.cap.isOpened():
        print('capture open')
    
def timerFired(data, key):
    keyPressed(data,key)


def keyPressed(data, key):
    if key == ord('c'):
        name = 'joan'
        photo1 = cvImage(name,data)
        photo1.faceGrab()
        data.cvImages.append(photo1)
    if key == ord('d'):
        displayVideo(data)
    if key == ord('f'):
        data.facesOnly = not data.facesOnly

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

def cvToTkinter(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(image))
    print('transformed to tkinter')
    return (img)
    
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
    
def redrawAll(canvas, data):
    if data.facesOnly:
        for cvImage in data.cvImages:
            for face in cvImage.outputFaces:
                canvas.create_image(0,0,image=face, anchor = NW)
    else:
        for cvImage in data.cvImages:
            output = cvImage.output
            canvas.create_image(0,0,image=output, anchor = NW)


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
        self.output = cvToTkinter(self.img)
        self.outputFaces = []
        
    def outline(self):
        self.outline = cv.Canny(self.img, 50, 100)
    
    def faceGrab(self):
        faces = faceCheck.detectMultiScale(self.bw, 1.3, 5)
        for (x,y,w,h) in faces:
            faceBW = self.bw[y:y+h, x:x+w]
            faceColor = self.img[y:y+h, x:x+w]
            self.faces.append(faceColor)
            self.outputFaces.append(cvToTkinter(faceColor))
            
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)

def runcv(width=512, height=512):
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    delay = 1
    init(data)
    
    background = np.zeros((width,height,3), np.uint8)
    cv.setMouseCallback(data.winName,mousePressed)
    while(1):
        cv.imshow(data.winName,background)
        k = timerFired(data, cv.waitKey(delay))
        if k == 27:
            break
    cv.destroyAllWindows()


def runtk(width=512, height=512):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    root = Tk()
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

runcv(720,720)