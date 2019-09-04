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
dataGlobal = None
clicked = False

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
    
    data.warping = False
    data.corners = [0]*4
    data.splashScreen = cv.imread('images/system/splashScreen.png')
    data.testPattern = cv.imread('images/system/testPattern.png')
    data.mainScreen = True
    
    background = getFrame(data)
    background = fitToWindow(data,background)
    data.background = background
    
    data.buttonR = data.height//30
    data.recButtonC = (data.width//2,data.height-3*data.buttonR)
    data.takingPhotos = False
    data.frameWaitCount = 0
    data.photoDelay = 3
    data.framesPerBooth = 4
    
    global dataGlobal
    dataGlobal = data
    
def mainScreen(data):
    cv.imshow(data.winName,data.splashScreen)
    
    
def timerFired(data, key):
    if data.mainScreen == True:
        mainScreen(data)
    return keyPressedMain(data,key)

def keyPressedMain(data, key):
    if key == ord('c'):
        photo1 = cvImage(data)
        photo1.faceGrab()
        data.cvImages.append(photo1)
    elif key == ord('m'):
        for img in data.cvImages:
            cv.destroyAllWindows()
            img = img.img - data.background
            cv.imshow(data.winName, img)
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
            cv.imshow(data.winName,newImg)
    elif key == ord('p'):
        data.warping = not data.warping
        if data.warping:
            warping(data)
    elif key == 27:
        return False
        

def mousePressedAll(event, x,y,flags,param):
    global clicked
    global dataGlobal
    data = dataGlobal
    if event == cv.EVENT_LBUTTONDOWN:
        clicked = True
        if data.mainScreen == True:
            data.boothOn = not data.boothOn
            data.mainScreen = False
            photoBooth(data)
        elif data.boothOn == True:
            dx = data.recButtonC[0]-x
            dy = data.recButtonC[1]-y
            if (dx**2+dy**2)**.5<=data.buttonR:
                data.boothOn = False
                data.takingPhotos = True
                showSet(data,photoBoothCapture(data))
        
    elif event == cv.EVENT_LBUTTONUP:
        clicked = False
    
    elif event == cv.EVENT_MOUSEMOVE:
        if clicked and data.warping:
            cornerPin(data,x,y)

def cornerPin(data,x,y):
    pass

def warping(data):
    cv.imshow(data.winName,data.testPattern)


def getFrame(data):
    if (data.cap.isOpened()):
        ret, frame = data.cap.read()
        if ret:
            #frame = cv.flip(frame, 1)
            pass
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
        if ret and not data.takingPhotos:
            
            frame = fitToWindow(data,frame)
            frame = drawPhotoBooth(data,frame)
            if type(data.background) == np.ndarray:
                cv.imshow(data.winName, frame)
            else:
                cv.imshow(data.winName, frame)
            keyPressedMain(data, cv.waitKey(data.fps))
        else: break

def photoBoothCapture(data):
    boothSet = photoBoothSet(data)
    while data.takingPhotos:
        ret, frame = data.cap.read()
        if ret and not data.boothOn and data.takingPhotos:
            frame = fitToWindow(data,frame)
            
            data.frameWaitCount += 1
            secsRemain=data.photoDelay-(data.frameWaitCount//(500/data.fps))
            if secsRemain>0:
                frame = drawCountdown(data,frame,secsRemain)
            else:
                boothSet.addImage(frame)
                whitescreen = np.zeros((720,720,3),np.uint8)
                whitescreen[:] = (255,255,255)
                photo = boothSet.images[boothSet.frameCount-1]
                opposite = whitescreen-photo
                for i in range (1,10):
                    div = np.zeros((720,720,3),np.uint8)
                    div[:] = (i,i,i)
                    cv.imshow(data.winName,(photo+opposite//div))
                    cv.waitKey(100)
                data.frameWaitCount = 0
                if boothSet.frameCount==data.framesPerBooth: 
                    data.takingPhotos=False
                    break
            
            cv.imshow(data.winName, frame)
            keyPressedMain(data, cv.waitKey(data.fps))
    return boothSet

def showSet(data,boothSet):
    img = boothSet.composite(data)
    cv.imshow(data.winName,img)
    

def drawPhotoBooth(data,frame):
    if not data.takingPhotos:
        cv.circle(frame,data.recButtonC,data.buttonR+2,(255,255,255),-1)
        cv.circle(frame,data.recButtonC,data.buttonR,(0,0,255),-1)
    return frame


def drawCountdown(data,frame,time):
    font = cv.FONT_HERSHEY_SCRIPT_COMPLEX
    cv.putText(frame,str(time),(0,data.height),font,int(data.height/21),
            (0,0,255),2,cv.LINE_AA)
    return frame


def outline(data,frame):
    blurFrame = gaussianBlur(data,frame)
    #print('blur complete')
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
            mx = max(G,B,R)
            mn = min(G,B,R)
            if mx != mn:
                r = 255/(mx-mn)
                G = (G-mn) * r
                R = (R-mn) * r
                B = (B-mn) * r
            a= np.array([G,B,R], dtype='uint8')
            editFrame[i,j] = a
    print(h,w,'normalized')
    return editFrame
    
def cautiousNormalize(data,frame):
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
    print('normalized')
    return editFrame
    



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
            
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)

class photoBoothSet(cvImage):
    def __init__(self,data):
        data.boothSession += 1
        self.name = 'boothSession_%d' %data.boothSession
        self.images = []
        self.faces = []
        self.frameCount = 0
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
        return canvas

def runcv(width=512, height=512):
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    delay = 10
    init(data)
    
    background = np.zeros((width,height,3), np.uint8)
    cv.setMouseCallback(data.winName,mousePressedAll)
    
    cv.imshow(data.winName,data.splashScreen)
    while(1):
        
        if timerFired(data, cv.waitKey(delay)) == False:
            break
    data.cap.release()
    cv.destroyAllWindows()
    cv.waitKey(1)
    print('bye!')

def initTK(dtk):
    dtk.mainScreen = False
    pass

def redrawAllTK(canvas, dtk):
    pass

def mousePressedTK(event, dtk):
    if dtk.mainScreen == False:
        runcv(720,720)
        dtk.mainScreen == True
    elif dtk.mainScreen == True:
        print('ok')
    pass
    
def keyPressedTK(event, dtk):
    pass
    
def timerFiredTK(dtk):
    pass

def runtk(width=512, height=512):
    def redrawAllWrapper(canvas, dtk):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, dtk.width, dtk.height,
                                fill='white', width=0)
        redrawAllTK(canvas, dtk)
        canvas.update()

    def mousePressedWrapper(event, canvas, dtk):
        mousePressedTK(event, dtk)
        redrawAllWrapper(canvas, dtk)

    def keyPressedWrapper(event, canvas, dtk):
        keyPressedTK(event, dtk)
        redrawAllWrapper(canvas, dtk)

    def timerFiredWrapper(canvas, dtk):
        timerFiredTK(dtk)
        redrawAllWrapper(canvas, dtk)
        # pause, then call timerFired again
        canvas.after(dtk.timerDelay, timerFiredWrapper, canvas, dtk)
    # Set up data and call init
    class Struct(object): pass
    dtk = Struct()
    dtk.width = width
    dtk.height = height
    dtk.timerDelay = 1 # milliseconds
    root = Tk()
    # create the root and the canvas
    canvas = Canvas(root, width=dtk.width, height=dtk.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, dtk))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, dtk))
    initTK(dtk)
    timerFiredWrapper(canvas, dtk)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")


runcv(720,720)


def cvToTkinter(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(image))
    print('transformed to tkinter')
    return (img)