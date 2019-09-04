import numpy as np
import cv2 as cv
import tkinter
import PIL
import copy
from tkinter import *
from TP_ImageProcessing import *
from PIL import Image
from PIL import ImageTk
import time
import math
import os
import random
eyeCheck=cv.CascadeClassifier('haarCascades/haarcascade_eye.xml')
faceCheck=cv.CascadeClassifier\
    ('haarCascades/haarcascade_frontalface_default.xml')
sideCheck=cv.CascadeClassifier\
    ('haarCascades/haarcascade_profileface.xml')
dataGlobal = None
clicked = False

def getFrame(data):
    if (data.cap.isOpened()):
        ret, frame = data.cap.read()
        if ret:
            #frame = cv.flip(frame, 1)
            pass
        else: return
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
    return frame
    
def getSessionNum(data):
    highest = 0
    for filename in os.listdir(data.boothDir):
        if filename ==".DS_Store":
            continue
        curr = int(filename[-1])
        if curr == highest:
            highest = curr+1
    return highest

# def redrawAll(canvas, data):
#     if data.facesOnly:
#         for cvImage in data.cvImages:
#             for face in cvImage.outputFaces:
#                 canvas.create_image(0,0,image=face, anchor = NW)
#     else:
#         for cvImage in data.cvImages:
#             output = cvImage.output
#             canvas.create_image(0,0,image=output, anchor = NW)


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
    
    def faceGrab(self,bw,color):
        faces = faceCheck.detectMultiScale(bw, 1.3, 5)
        for (x,y,w,h) in faces:
            if max(w,h) == h:
                x=(x+w//2)-(h//2)
                faceColor = color[y:y+h, x:x+h]
            else:
                y=(y+h//2)-(w//2)
                faceColor = color[y:y+w, x:x+w]
            self.faces.append(faceColor)
        sideFaces = sideCheck.detectMultiScale(bw, 1.3, 5)
        for (x,y,w,h) in sideFaces:
            print('got a face')
            if max(w,h) == h:
                x=(x+w//2)-(h//2)
                faceColor = color[y:y+h, x:x+h]
            else:
                y=(y+h//2)-(w//2)
                faceColor = color[y:y+w, x:x+w]
            self.faces.append(faceColor)
        
            
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)

class photoBoothSet(cvImage):
    def __init__(self,data,session):
        # data.boothSession = getSessionNum(data)
        self.name = 'boothSession_%d' %data.boothSession
        self.images = []
        self.faces = []
        self.comp = None
        self.frameCount = 0
        self.dir = data.boothDir + '/' + self.name
        self.faceDir = data.faceDir+ '/' + self.name
        os.mkdir(self.dir)
        os.mkdir(self.faceDir)
    
    def addImage(self,frame):
        self.images.append(frame)
        self.frameCount+=1
        bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.faceGrab(bw,frame)
        
    def composite(self,data):
        divs = int(self.frameCount**.5)
        canvas = np.zeros((data.width,data.height,3), np.uint8)
        whichPhoto = 0
        for i in range(divs):
            for j in range(divs):
                img = self.images[whichPhoto]
                xy = data.width // divs
                imgScale = cv.resize(img,(xy,xy))
                dim = imgScale.shape[0]
                canvas[i*dim:(i+1)*dim,j*dim:(j+1)*dim]=imgScale
                whichPhoto+=1
        self.comp = canvas
        return self.comp
    
    def log(self,data):
        filename = data.entry
        cv.imwrite(self.dir+'/'+filename+'.png',self.comp)
        self.images = []
        faceNum=0
        for face in self.faces:
            filename =self.faceDir+'/face'+\
                str(data.boothSession)+'_'+str(faceNum)+'.png'
            cv.imwrite(filename,face)
            faceNum+=1
        self.faces=[]
        data.entry = ''
        

################ INITIALIZATION

def initTK(data):
    data.mode = 'splashScreen'
    initcv(data)
    data.loops = 0

def initcv(data,width=512, height=512):
    data.width = width
    data.height = height
    
    data.blank = np.zeros((width,height,3), np.uint8)
    
    data.canvas = np.zeros((512,512,3), np.uint8)
    data.winName = 'Photo Booth'
    data.window = cv.namedWindow(data.winName)
    data.cap = cv.VideoCapture(0)
    data.cvBG = None
    
    
    data.fps = int(1000/30)
    data.time = time.time()
    if data.cap.isOpened():
        print('capture open')
    
    
    data.splashScreen = cv.imread('images/system/splashScreen.png')
    data.splashScreen = fitToWindow(data, data.splashScreen)
    data.testPattern = cv.imread('images/system/testPattern.png')
    data.testPattern = fitToWindow(data,data.testPattern)
    data.thanksSplash = cv.imread('images/system/thanksSplash.png')
    data.thanksSplash = fitToWindow(data,data.thanksSplash)
    
    data.cpR = 25
    data.corners=[[0,0],[data.width,0],[data.width,data.height],
        [0,data.height]]
    data.circles = [
        [data.testPattern,tuple(data.corners[0]),data.cpR,(0,0,255),5,False],
        [data.testPattern,tuple(data.corners[1]),data.cpR,(0,255,0),5,False],
        [data.testPattern,tuple(data.corners[2]),data.cpR,(255,0,0),5,False],
        [data.testPattern,tuple(data.corners[3]),data.cpR,(255,255,0),5,False]
        ]
    data.warpKeySize = 5
    data.warpTolerance = 170
    data.lastLegalMouse = (0,0)
    
    background = getFrame(data)
    background = fitToWindow(data,background)
    background = cv.flip(background,1)
    data.background = background
    data.diff = False
    
    data.boothSession = -1
    data.buttonR = data.height//30
    data.recButtonC = (data.width//2,data.height-3*data.buttonR)
    data.frameWaitCount = 0
    data.photoDelay = 3
    data.framesPerBooth = 4
    data.countdownTime = 50
    data.takingPhoto = False
    data.curBoothSet = None
    
    data.boothDir = 'images/assets/boothImgs'
    data.faceDir = 'images/assets/faces'
    
    data.entry = ''
    data.entryLimit = 30
    data.entryFont = 'Courier %i'%(data.width//data.entryLimit)
    data.thankYou = False
    
    data.outGridNum = 3
    data.allFaces = {}
    data.allFaceNames = []
    data.outputInit = False
    data.outputGrid = None
    
    global dataGlobal
    dataGlobal = data
    

################ MODE SWITCH

def mousePressedata(event, data):
    if data.mode == 'splashScreen': splashMousePressed(event,data)
    elif data.mode == 'photoBooth': photoMousePressed(event,data)
    elif data.mode == 'takePicture': takePicMousePressed(event,data)
    elif data.mode == 'warp': warpMousePressed(event,data)
    elif data.mode == 'showBooth': showBoothMousePressed(event,data)
    elif data.mode == 'output': outputMousePressed(event,data)
    
def keyPressedata(event, data):
    if data.mode == 'splashScreen': splashKeyPressed(event,data)
    elif data.mode == 'photoBooth': photoKeyPressed(event,data)
    elif data.mode == 'takePicture': takePicKeyPressed(event,data)
    elif data.mode == 'warp': warpKeyPressed(event,data)
    elif data.mode == 'showBooth': showBoothKeyPressed(event,data)
    elif data.mode == 'output': outputKeyPressed(event,data)
    
def timerFiredata(data):
    if data.mode == 'splashScreen': splashTimerFired(data)
    elif data.mode == 'photoBooth': photoTimerFired(data)
    elif data.mode == 'takePicture': takePicTimerFired(data)
    elif data.mode == 'warp': warpTimerFired(data)
    elif data.mode == 'showBooth': showBoothTimerFired(data)
    elif data.mode == 'output': outputTimerFired(data)

def redrawAllTK(canvas, data):
    if data.mode == 'splashScreen': splashRedrawAll(canvas,data)
    elif data.mode == 'photoBooth': photoRedrawAll(canvas,data)
    elif data.mode == 'takePicture': takePicRedrawAll(canvas,data)
    elif data.mode == 'warp': warpRedrawAll(canvas,data)
    elif data.mode == 'showBooth': showBoothRedrawAll(canvas,data)
    elif data.mode == 'output': outputRedrawAll(canvas,data)
    
def cvMouse(data):
    if data.mode == 'splashScreen':
        cv.setMouseCallback(data.winName,splashCVMouse)
    elif data.mode == 'photoBooth':
        cv.setMouseCallback(data.winName,photoCVMouse)
    elif data.mode == 'takePicture':
        cv.setMouseCallback(data.winName,takePicCVMouse)
    elif data.mode == 'warp':
        cv.setMouseCallback(data.winName,warpCVMouse)
    elif data.mode == 'showBooth': 
        cv.setMouseCallback(data.winName,showBoothCVMouse)
    elif data.mode == 'output': 
        cv.setMouseCallback(data.winName,outputCVMouse)


################ SPLASH SCREEN MODE

def splashMousePressed(event,data):
    if event.x>0 and event.y>0:
        data.mode = 'warp'

def splashKeyPressed(event,data):
    if event.keysym == 'o':
        data.mode = 'output'
    elif event.keysym == 'w':
        data.mode = 'warp'
    
def splashTimerFired(data):
    pass
    
def splashRedrawAll(canvas,data):
    cv.imshow(data.winName,data.splashScreen)
    pass
    
def splashCVMouse(event, x,y,flags,param):
    global dataGlobal
    data = dataGlobal
    if event == cv.EVENT_LBUTTONDOWN:
        data.mode = 'photoBooth'

################ PHOTO BOOTH MODE

def photoMousePressed(event,data):
    pass

def photoKeyPressed(event,data):
    if event.keysym == 'r':
        data.diff = not data.diff
    if event.keysym == 'z':
        background = getFrame(data)
        background = fitToWindow(data,background)
        background = cv.flip(background,1)
        data.background = background
    
def photoTimerFired(data):
    pass
    
def photoRedrawAll(canvas,data):
    ret, frame = data.cap.read()
    if ret:
        frame = fitToWindow(data,frame)
        frame = cv.flip(frame, 1)
        if data.diff == True:
            frame = (frame - data.background)
            cv.imshow(data.winName, frame)
        else:
            frame = drawPhotoBooth(data,frame)
            cv.imshow(data.winName, frame)
            #keyPressedMain(data, cv.waitKey(data.fps))

def drawPhotoBooth(data,frame):
    cv.circle(frame,data.recButtonC,data.buttonR+2,(255,255,255),-1)
    cv.circle(frame,data.recButtonC,data.buttonR,(0,0,255),-1)
    return frame

def photoCVMouse(event, x,y,flags,param):
    global clicked
    global dataGlobal
    data = dataGlobal
    if event == cv.EVENT_LBUTTONDOWN:
        dx = data.recButtonC[0]-x
        dy = data.recButtonC[1]-y
        if (dx**2+dy**2)**.5<=data.buttonR:
            data.mode = 'takePicture'
        

################ TAKE PICTURE MODE

def takePicMousePressed(event,data):
    pass

def takePicKeyPressed(event,data):
    if event.keysym == 'r':
        data.diff = not data.diff
    if event.keysym == 'z':
        background = getFrame(data)
        background = fitToWindow(data,background)
        background = cv.flip(background,1)
        data.background = background
    
def takePicTimerFired(data):
    data.frameWaitCount += 1
    
def takePicRedrawAll(canvas,data):
    if data.takingPhoto == False:
        data.boothSession = getSessionNum(data)
        session = 'boothSession_%d' %data.boothSession
        data.curBoothSet = photoBoothSet(data,session)
        data.takingPhoto = True
    elif data.takingPhoto == True:
        runCapture(data,data.curBoothSet)
    elif data.takingPhoto == None:
        showSet(data,data.curBoothSet)

def takePicCVMouse(event, x,y,flags,param):
    global clicked
    global dataGlobal
    data = dataGlobal

def runCapture(data,boothSet):
    ret, frame = data.cap.read()
    if ret:
        frame = cv.flip(fitToWindow(data,frame),1)
        secsRemain=data.photoDelay-\
            (data.frameWaitCount//(data.countdownTime/data.fps))
        if secsRemain>0:
            frame = drawCountdown(data,frame,secsRemain)
            cv.imshow(data.winName, frame)
        else:
            boothSet.addImage(frame)
            whitescreen = np.zeros((data.width,data.height,3),np.uint8)
            whitescreen[:] = (255,255,255)
            
            photo = boothSet.images[boothSet.frameCount-1]
            opposite = whitescreen-photo
            for i in range (1,10):
                div = np.zeros((data.width,data.height,3),np.uint8)
                div[:] = (i,i,i)
                cv.imshow(data.winName,(photo+opposite//div))
                cv.waitKey(100)
            data.frameWaitCount = 0
            if boothSet.frameCount==data.framesPerBooth: 
                data.frameWaitCount = 0
                data.mode = 'showBooth'

    
def drawCountdown(data,frame,time):
    font = cv.FONT_HERSHEY_SCRIPT_COMPLEX
    cv.putText(frame,str(time),(0,data.height),font,int(data.height/21),
            (0,0,255),2,cv.LINE_AA)
    return frame

################# SHOWBOOTH MODE

def showBoothKeyPressed(event,data):
    k = event.keysym
    if k == 'Return' and len(data.entry)>0:
        data.curBoothSet.log(data)
        data.thankYou = True
    if k == 'BackSpace':
        if len(data.entry)>0:
            data.entry = data.entry[:-1]
    elif len(data.entry)>data.entryLimit:
        return
    elif k == 'space':
        data.entry = data.entry+' '
    elif k.isalnum() or k.isdigit() or k=='period' or k=='at':
        if len(k)==1:
            data.entry = data.entry+k

def showBoothMousePressed(event,data):
    pass  
    
def showBoothCVMouse(event, x,y,flags,param):
    pass

def showBoothTimerFired(data):
    if data.thankYou:
        print('thankyou')
        showThanks(data)
    else:
        showSet(data)

def showBoothRedrawAll(canvas,data):
    canvas.create_text(data.tkWidth//2,data.tkHeight//2,
        text=data.entry,anchor=CENTER,font=data.entryFont,)
    pass

def showSet(data):
    img = data.curBoothSet.composite(data)
    cv.imshow(data.winName,img)
    
def showThanks(data):
    cv.imshow(data.winName,data.thanksSplash)
    cv.waitKey(5000)
    initTK(data)

################# OUTPUT MODE

def outputKeyPressed(event,data):
    if event.keysym == 'Up' and (data.outGridNum+1)**2<=len(data.allFaces):
        data.outGridNum += 1
        data.outputGrid = makeOutGrid(data)
    elif event.keysym == 'Down' and data.outGridNum>1:
        data.outGridNum -= 1
        data.outputGrid = makeOutGrid(data)
    elif event.keysym == 'w':
        data.mode = 'warp'
    elif event.keysym == 's':
        data.mode = 'splashScreen'
        data.outputInit = False
    
    elif event.keysym == 'o':
        data.outputGrid = outline(data,data.outputGrid)
    elif event.keysym == 'n':
        data.outputGrid = normalize(data,data.outputGrid)
    elif event.keysym == 'b':
        data.outputGrid = gaussianBlur(data,data.outputGrid)
    elif event.keysym == 'c':
        data.outputGrid = cautiousNormalize(data,data.outputGrid)
    

def outputMousePressed(event,data):
    pass

def outputTimerFired(data):
    pass
    
def outputRedrawAll(canvas,data):
    if data.outputInit==False:
        getImages(data,data.faceDir)
        data.outputGrid = makeOutGrid(data)
        data.outputInit=True
    warpedOutput = warp(data,data.outputGrid)
    cv.imshow(data.winName, warpedOutput)

def outputCVMouse(event, x,y,flags,param):
    pass

def getImages(data,path,depth=0):
    for filename in os.listdir(path):
        if filename == '.DS_Store':
            continue
        subpath = path+'/'+filename
        if os.path.isdir(subpath):
            getImages(data,subpath,depth+1)
        else: 
            if subpath not in data.allFaces:
                img = cv.imread(subpath)
                data.allFaces[filename] = img
                data.allFaceNames.append(filename)

def makeOutGrid(data):
    divs = data.outGridNum
    if divs**2>len(data.allFaces):
        divs = int(len(data.allFaces)**.5)
    canvas = np.zeros((data.width,data.height,3), np.uint8)
    whichPhoto = 0
    random.shuffle(data.allFaceNames)
    for i in range(divs):
        for j in range(divs):
            img = data.allFaces[data.allFaceNames[whichPhoto]]
            xy = data.width // divs
            imgScale = cv.resize(img,(xy,xy))
            dim = imgScale.shape[0]
            canvas[i*dim:(i+1)*dim,j*dim:(j+1)*dim] = imgScale
            whichPhoto+=1
    return canvas


################# WARP MODE


def warpMousePressed(event,data):
    pass

def warpKeyPressed(event,data):
    if event.keysym == 'o':
        data.mode = 'output'
    elif event.keysym == 's':
        data.outputInit = False
        data.mode = 'splashScreen'
    elif event.keysym == 'Tab':
        c=0
        for i in range(len(data.circles)):
            c+=1
            if data.circles[i][5]:
                data.circles[i][5]=False
                next = (i+1)%len(data.circles)
                data.circles[next][5]=True
                break
            if c==4: 
                data.circles[0][5]=True
    elif event.keysym == 'Up':
        for circle in data.circles:
            if circle[5]==True and circle[1][1]>0:
                circle[1]=(circle[1][0],circle[1][1]-data.warpKeySize)
                if not legalCircleSetup(data):
                    circle[1]=(circle[1][0],circle[1][1]+data.warpKeySize)
    elif event.keysym == 'Down':
        for circle in data.circles:
            if circle[5] == True and circle[1][1]<data.height:
                circle[1]=(circle[1][0],circle[1][1]+data.warpKeySize)
                if not legalCircleSetup(data):
                    circle[1]=(circle[1][0],circle[1][1]-data.warpKeySize)
    elif event.keysym == 'Left':
        for circle in data.circles:
            if circle[5] == True and circle[1][0]>0:
                circle[1]=(circle[1][0]-data.warpKeySize,circle[1][1])
                if not legalCircleSetup(data):
                    circle[1]=(circle[1][0]+data.warpKeySize,circle[1][1])
    elif event.keysym == 'Right':
        for circle in data.circles:
            if circle[5] == True and circle[1][0]<data.width:
                circle[1]=(circle[1][0]+data.warpKeySize,circle[1][1])
                if not legalCircleSetup(data):
                    circle[1]=(circle[1][0]-data.warpKeySize,circle[1][1])

def legalCircleSetup(data):
    #test if no three adjacent coords create angle of greater than tolerance
    for i in range(len(data.circles)):
        next = (i+1)%len(data.circles)
        last = (i-1)%len(data.circles)
        nextX = data.circles[next][1][0]
        nextY = data.circles[next][1][1]
        lastX = data.circles[last][1][0]
        lastY = data.circles[last][1][1]
        curX = data.circles[i][1][0]
        curY = data.circles[i][1][1]
        A = pyth((nextX-lastX),(nextY-lastY))
        B = pyth((nextX-curX),(nextY-curY))
        C = pyth((lastX-curX),(lastY-curY))
        #law of cosines
        angle = math.acos((B**2+C**2-A**2)/(2*B*C))
        angle = math.degrees(angle)
        if angle>=data.warpTolerance:
            return False
    return True
    
def warpTimerFired(data):
    circleColor(data)
    
def warp(data,img):
    dstPoints = []
    for circle in data.circles:
        coord = list(circle[1])
        dstPoints.append(coord)
        x,y = [],[]
        x.append(circle[1][0])
        y.append(circle[1][1])

    w=max(x)-min(x)
    h=max(y)-min(y)
    dstPoints = np.float32(dstPoints)
    corners = np.float32(data.corners)
    t = cv.getPerspectiveTransform(corners,dstPoints)
    return cv.warpPerspective(img,t,(data.width,data.height))
    
    
def warpRedrawAll(canvas,data):
    tp = copy.copy(data.testPattern)
    tp = warp(data,tp)
    for n in range(len(data.circles)):
        a=data.circles[n]
        cv.circle(tp,a[1],a[2],a[3],a[4])
    cv.imshow(data.winName,tp)
    pass

def warpCVMouse(event, x,y,flags,param):
    global clicked
    global dataGlobal
    data = dataGlobal
    
    if event == cv.EVENT_LBUTTONDOWN:
        clicked = True
        data.time = time.time()
        circleSelector(data,x,y)
    
    elif event == cv.EVENT_LBUTTONUP:
        clicked = False
        t = (time.time() - data.time)
        if t > .25:
            for circle in data.circles:
                circle[5]=False
        
    elif event == cv.EVENT_MOUSEMOVE:
        if clicked:
            for circle in data.circles:
                if x==0 or x==data.width-1 or y==0 or y==data.height-1:
                    circle[5] = False
                if circle[5]:
                    circle[1]=(x,y)
                    if legalCircleSetup(data):
                        data.lastLegalMouse = (x,y)
                    else:
                        circle[1]=data.lastLegalMouse
                        circle[5] = False

def pyth(a,b):
    return ((a**2)+(b**2))**.5

def circleSelector(data,x,y):
    n = 0
    for circle in data.circles:
        if pyth(x-circle[1][0],y-circle[1][1])<=circle[2]:
            if not circle[5]:
                resetCircles(data)
                circle[5] = True
            else:
                resetCircles(data)
        else:
            n+=1
        if n == 4:
            resetCircles(data)

def circleColor(data):
    for circle in data.circles:
        if circle[5]: 
            circle[4] = -1
        elif not circle[5]: 
            circle[4] = 5

def resetCircles(data):
    for circle in data.circles:
        circle[4] = 5
        circle[5] = False

################# RUN FUNCTION - created by 112 faculty

def runtk(width=512, height=512):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAllTK(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressedata(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressedata(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFiredata(data)
        cvMouse(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.tkWidth = width
    data.tkHeight = height
    data.timerDelay = 1 # milliseconds
    root = Tk()
    # create the root and the canvas
    canvas = Canvas(root, width=data.tkWidth, height=data.tkHeight)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    initTK(data)
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    cv.destroyAllWindows()
    cv.waitKey(1)
    print("bye!")


runtk(720,50)


def cvToTkinter(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(image))
    print('transformed to tkinter')
    return (img)
    
    
    