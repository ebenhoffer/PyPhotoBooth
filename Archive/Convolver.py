import numpy as np
import cv2 as cv

import numpy as np
import cv2 as cv

def this():
    # photo = PhotoImage(file="gamescreen.gif")
    canvas.bind("<Button-1>", buttonclick_gamescreen)
    canvas.pack(expand = YES, fill = BOTH)
    canvas.create_image(1, 1, image = photo, anchor = NW)
    e1 = Entry(canvas)
    e2 = Entry(canvas)
    # game1 = PhotoImage(file="images/system/splashScreen.png")
    # canvas.create_image(30, 65, image = game1, anchor = NW)
    canvas.create_window(window = e1, x=10, y=10)
    canvas.create_window(window = e2 , x=400, y=10)    
    canvas.update()
    window.mainloop()
this()


# cap = cv.VideoCapture(0)
# fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
# while(1):
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
#     g = cv.cvtColor(fgmask, cv.COLOR_GRAY2BGR)
#     frame = frame*(g//255)
#     cv.imshow('frame',frame)
#     k = cv.waitKey(10) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv.destroyAllWindows()
# 
# 
# def warpTimerFired(data):
#     warpPoints = []
#     for circle in data.circles:
#         coord = list(circle[1])
#         warpPoints.append(coord)
#         x,y = [],[]
#         x.append(circle[1][0])
#         y.append(circle[1][1])
# 
#     w=max(x)-min(x)
#     h=max(y)-min(y)
#     warpPoints = np.array(warpPoints)
#     corners = np.array(data.corners)
#     h,status=cv.findHomography(corners,warpPoints)
# 
#     
#     data.testPattern=cv.warpPerspective(data.testPattern,h,(w,h))
# 

# cap = cv.VideoCapture(0) 
# cv.xfeatures2d.SIFT_create()
# 
# fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
# while(1):
#     ret, frame = cap.read()
#     i=2.
#     whitescreen = np.zeros((720,1280,3),np.uint8)
#     whitescreen[:] = (255,255,255)
#     opposite = whitescreen-frame
#     div = np.zeros((720,1280,3),np.float)
#     div[:] = (i,i,i)
#     cv.imshow('yes',((frame + opposite//div)))
#     
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#         
# cap.release()
# cv.destroyAllWindows()
# cv.waitKey(1)



# while(1):
#     ret, frame = cap.read()
#     i=.9
#     whitescreen = np.zeros((720,1280,3),np.uint8)
#     whitescreen[:] = (255,255,255)
#                 #cv.imshow(data.winName,whitescreen)
#     opposite = whitescreen-frame
#     
#     div = np.zeros((720,1280,3),np.uint8)
#     div[:] = (i,i,i)
#     cv.imshow('yes',((np.multiply(frame,opposite))))