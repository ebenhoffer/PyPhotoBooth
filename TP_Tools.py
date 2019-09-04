#Eben Hoffer
#15-112, Section I
# ehoffer

##### various tools for photo booth unassociated with a particular mode
##### resizing to variable window, frame capture, email, init-called functions


import numpy as np
import cv2 as cv
import os
import smtplib
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import PIL
from PIL import Image
from PIL import ImageTk

def getFrame(data):
    if (data.cap.isOpened()):
        ret, frame = data.cap.read()
        if ret:
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
    #determine number of sessions already stored in directory
    highest = 0
    for filename in os.listdir(data.boothDir):
        if filename ==".DS_Store":
            continue
        n = filename.index('_')
        curr = int(filename[n+1:])
        if curr >= highest:
            highest = curr+1
    return highest
    
    
############## EXTRAS

def cvToTkinter(image):
    #turns a cv image into a tkinter image
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(image))
    print('transformed to tkinter')
    return (img)
    
def sendMail(data, toAddress, imgPath=None):
    #adapted from Arjun Krishna Babu, found online at freeCodeCamp
    print('sending')
    # set up the SMTP server
    selfAddress = 'eben112TP@gmail.com'
    password = '15-112Kills'
    
    s = smtplib.SMTP_SSL(host='smtp.gmail.com', port=465)
    s.ehlo()
    s.login(selfAddress, password)
    msg = MIMEMultipart()
    message = "You used a photo booth!"

    # setup the parameters of the message
    msg['From']=selfAddress
    msg['To']=toAddress
    msg['Subject']="Enjoy Your Fine Image"
    fp = open(imgPath, 'rb')
    image = MIMEImage(fp.read())
    
    msg.attach(image)
    msg.attach(MIMEText(message, 'plain'))
    
    # send the message via the server set up earlier.
    try:
        s.send_message(msg)
        print('sent')
    except:
        print('not a real email')
    del msg
    s.quit()
    