import numpy as np
import cv2 as cv

#img= cv.imread('currentspace.jpg', 0)
def takePicture(name):
    cap = cv.VideoCapture(0)

    on, frame = cap.read()
    if on:
        frame = cv.flip(frame, 1)
        out.write(frame)
        cv.imshow('frame',frame)
        cv.imwrite('%s.png'%name, frame)
    cap.release()
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)

def showVideo():
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv.flip(frame, 1)
            out.write(frame)
            cv.imshow('frame',frame)
        k = cv.waitKey(33)
        if k == ord('q'):
            break
            

# cv.namedWindow('yes', cv.WINDOW_NORMAL)
# cv.imshow('yes',img)
# k = cv.waitKey(0)

# if k == ord('c'):
#     cv.destroyAllWindows()
#     cv. waitKey(1)
# elif k == ord('s'):
#     cv.imwrite('tryit.png', img)
