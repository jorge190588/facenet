import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
name = 'Jorge_Santos'
folder=name+'\\'
directorioDeImagenes = '..\lfw\origen'
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path,directorioDeImagenes,folder)
name = name+'_'
extension = '.jpg'
index = 1
maxFileNumber = 1000

while True:
	ret,img = cap.read()
	
	# show image
	cv2.imshow('Video', img)
	k=cv2.waitKey(10)& 0xff
	if k==27:
		break

	if (index < maxFileNumber):
		# fileName = fullPath+name+str(index)+extension
		fileName = path+name+str(index)+extension
		print('full name ',fileName)

		# save image
		cv2.imwrite(fileName, img)
		index = index+1

cap.release()
cv2.destroyAllWindows()