import cv2
import numpy as np
import os

name = 'Jose_Mourinho'
cap = cv2.VideoCapture(name+'.mp4')
folder=name+'\\'
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path,'..\lfw\origen',folder)
extension = '.jpg'
name =name +"_"
index = 1
maxFileNumber = 1000
startInFrame = 100
indexFrame = 0 
while True:
    ret,img = cap.read()

    # show image
    cv2.imshow('Video', img)
    k=cv2.waitKey(10)& 0xff
    if k==27:
        break
    
    indexFrame=indexFrame+1
    if (indexFrame>startInFrame):
        if (index < maxFileNumber):
            fileName = path+name+str(index)+extension
            print('full name ',fileName)
            # save image
            cv2.imwrite(fileName, img)
            index = index+1

cap.release()
cv2.destroyAllWindows()