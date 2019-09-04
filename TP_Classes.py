#Eben Hoffer
#15-112, Section I
# ehoffer

##### Classes used for image capture into batches, includes writing to disk


import numpy as np
import cv2 as cv
import os
from TP_Tools import *
#face recognition algorithms, created by intel
faceCheck=cv.CascadeClassifier\
    ('haarCascades/haarcascade_frontalface_default.xml')
sideCheck=cv.CascadeClassifier\
    ('haarCascades/haarcascade_profileface.xml')

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
        #snag faces from incoming image
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
            if max(w,h) == h:
                x=(x+w//2)-(h//2)
                faceColor = color[y:y+h, x:x+h]
            else:
                y=(y+h//2)-(w//2)
                faceColor = color[y:y+w, x:x+w]
            self.faces.append(faceColor)
        
class photoBoothSet(cvImage):
    #class for aggregating photo booth pictures
    def __init__(self,data,session):
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
        #add an image (up to a limit) to class
        self.images.append(frame)
        self.frameCount+=1
        bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.faceGrab(bw,frame)
        
    def composite(self,data):
        #create 'Let It Be' style composite image from stored photos
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
        #save faces and composite to disk, self-destruct. 
        #email composite to user along the way
        filename = data.entry
        filename = self.dir+'/'+filename+'.png'
        cv.imwrite(filename,self.comp)
        sendMail(data,data.entry,filename)
        self.images = []
        faceNum=0
        for face in self.faces:
            filename =self.faceDir+'/face'+\
                str(data.boothSession)+'_'+str(faceNum)+'.png'
            cv.imwrite(filename,face)
            faceNum+=1
        self.faces=[]
        data.entry = ''