from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf, cv2, os, matplotlib.pyplot as plt, numpy as np, argparse, sys, time, copy, math, pickle
from scipy import misc
import facenet
import align.detect_face
from os.path import join as pjoin
from sklearn.svm import SVC
from sklearn.externals import joblib

class TestFacenet:
    def __init__(self, _alignDirectory=None, _facesDirectory=None,  _modelDirectory=None,
                 _modelFile=None, _pairFileDirectory=None, _pairFile=None, _cameraNumber=None):
        self.alignDirectory  = _alignDirectory
        self.facesDirectory = _facesDirectory
        self.modelDirectory = _modelDirectory
        self.modelFile = _modelFile
        self.modelFilePath = _pairFileDirectory
        self.pairFile= _pairFile
        self.pairFileDirectory= _cameraNumber

    def init(self):
        path =os.path.dirname(os.path.realpath(__file__))
        self.alignDirectory  = os.path.join(path,self.alignDirectory)
        self.facesDirectory  = os.path.join(path,self.facesDirectory)
        self.modelDirectory = os.path.join(path,self.modelDirectory)
        self.modelFilePath = os.path.join(path,self.modelDirectory,self.modelFile)
        print("pairFileDirectory",self.pairFileDirectory)
        self.pairFileDirectory = os.path.join(path,self.pairFileDirectory)
        self.pairFilePath = os.path.join(path,self.pairFileDirectory, self.pairFile)
        if (self.checkIfDirectoryExists("Align directory", self.alignDirectory)) is False: return False
        if (self.checkIfDirectoryExists("Faces directory", self.facesDirectory)) is False: return False
        if (self.checkIfDirectoryExists("Model path", self.modelFilePath)) is False: return False
        if (self.checkIfDirectoryExists("Pair file path", self.pairFilePath)) is False: return False
        self.runTest()
        
    def checkIfDirectoryExists(self, name, path):
        if not (os.path.exists(path)):
            print("The",name,"doesn't exists in the path ",path)
            return False
        return True
    
    def getFacesList(self):
        facesList = [name for name in os.listdir(self.facesDirectory) if name.find(".txt") == -1]
        facesList.sort()
        return facesList

    def getFaceNameFromFacesListByIndex(self, facesList, findIndex ):
        for indexOfFacesList in facesList:
            if facesList[findIndex] == indexOfFacesList:
                faceName=facesList[findIndex]
                return faceName
        return "Not detected"

    def getBGRcodeFromColor(self, colorName):
        if (colorName is "black"):
            return 0,0,0
        if (colorName is "red"):
            return 0, 0, 255
        if (colorName is "green"):
            return 0, 255,0
        if (colorName is "blue"):
            return 175, 76, 9
        if (colorName is "white"):
            return 255, 255, 255

    def printTextToImage(self, image, text, position_x, position_y, colorName):
        colorB, colorG, colorR = self.getBGRcodeFromColor(colorName)
        #print("text ",colorB, colorG, colorR)
        cv2.putText(image, text, (position_x, position_y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 
                    (colorB, colorG, colorR), thickness=1, lineType=2)        

    def printRectangleToImage(self, image,position_x1,position_y1,position_x2,position_y2, colorName):
        colorB, colorG, colorR  = self.getBGRcodeFromColor(colorName)
        #print("rectangle ",colorB, colorG, colorR)
        cv2.rectangle(  image, (position_x1, position_y1), (position_x2, position_y2), 
                        (colorB, colorG, colorR), 4)

    def printRectangleToImageBackground(self, image,position_x1,position_y1,position_x2,position_y2, 
                                        colorName, text):
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN                               
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        colorB, colorG, colorR  = self.getBGRcodeFromColor(colorName)
        
        if (position_x1+text_width)>position_x2:
            position_x2 = position_x1 + text_width
        
        cv2.rectangle(image, (position_x1-5, position_y2-1), (position_x2 , position_y2+25), 
                    (colorB, colorG, colorR), 
                    cv2.FILLED)

    def getModel(self):
        classifierFilePath = os.path.expanduser(self.pairFilePath)
        with open(classifierFilePath, 'rb') as infile:
            u = pickle._Unpickler(infile)
            u.encoding = 'latin1'
            (model, class_names) = u.load()
            print('load classifier file-> %s' % classifierFilePath)
        return model

    def runTest(self):    
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, self.alignDirectory)
                minimunSizeOfFace = 20
                scaleFactor = 0.709
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                # margin = 44
                frame_interval = 3
                image_size = 182
                input_image_size = 160                
                facesList = self.getFacesList()
                print('Listado de rostros',facesList)
                facenet.load_model(self.modelFilePath)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                try:
                    model =self.getModel()

                    video_capture = cv2.VideoCapture(0) #'./test.mp4'
                    #video_capture.set(3,4920)
                    #video_capture.set(4,3080)
                    c = 0

                    # #video writer
                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(4920,3080))

                    print('Start Recognition!')
                    prevTime = 0
                    while True:
                        ret, frame = video_capture.read()
                        #if (frame != None):
                        frame = cv2.resize(frame, (0,0), fx=2, fy=2)    #resize frame (optional)
                        curTime = time.time()+1    # calc fps
                        timeF = frame_interval

                        if (c % timeF == 0):
                            find_results = []
                            if frame.ndim == 2:
                                frame = facenet.to_rgb(frame)
                            frame = frame[:, :, 0:3]
                            boundingBoxesOfAllDetectedFaces, _ = align.detect_face.detect_face(frame, minimunSizeOfFace, pnet, rnet, onet, threshold, scaleFactor)
                            numberOfFacesDeteted = boundingBoxesOfAllDetectedFaces.shape[0]
                            self.printTextToImage(frame,"No. Faces "+str(numberOfFacesDeteted),20,20,"black" )
                            if numberOfFacesDeteted > 0:
                                boundingBoxesOfDetectedFacesWith4Positions = boundingBoxesOfAllDetectedFaces[:, 0:4]
                                # img_size = np.asarray(frame.shape)[0:2]
                                cropped = []
                                scaled = []
                                scaled_reshape = []
                                boundingBoxesOfDetectedFace = np.zeros((numberOfFacesDeteted,4), dtype=np.int32)
                                
                                for indexOfFaceDetected in range(numberOfFacesDeteted):
                                    emb_array = np.zeros((1, embedding_size))
                                    boundingBoxesOfDetectedFace[indexOfFaceDetected][0] = boundingBoxesOfDetectedFacesWith4Positions[indexOfFaceDetected][0]
                                    boundingBoxesOfDetectedFace[indexOfFaceDetected][1] = boundingBoxesOfDetectedFacesWith4Positions[indexOfFaceDetected][1]
                                    boundingBoxesOfDetectedFace[indexOfFaceDetected][2] = boundingBoxesOfDetectedFacesWith4Positions[indexOfFaceDetected][2]
                                    boundingBoxesOfDetectedFace[indexOfFaceDetected][3] = boundingBoxesOfDetectedFacesWith4Positions[indexOfFaceDetected][3]

                                    # inner exception
                                    if boundingBoxesOfDetectedFace[indexOfFaceDetected][0] <= 0 or boundingBoxesOfDetectedFace[indexOfFaceDetected][1] <= 0 or boundingBoxesOfDetectedFace[indexOfFaceDetected][2] >= len(frame[0]) or boundingBoxesOfDetectedFace[indexOfFaceDetected][3] >= len(frame):
                                        #print('face is inner of range!')
                                        continue

                                    cropped.append(frame[boundingBoxesOfDetectedFace[indexOfFaceDetected][1]:boundingBoxesOfDetectedFace[indexOfFaceDetected][3], boundingBoxesOfDetectedFace[indexOfFaceDetected][0]:boundingBoxesOfDetectedFace[indexOfFaceDetected][2], :])
                                    cropped[indexOfFaceDetected] = facenet.flip(cropped[indexOfFaceDetected], False)
                                    scaled.append(misc.imresize(cropped[indexOfFaceDetected], (image_size, image_size), interp='bilinear'))
                                    scaled[indexOfFaceDetected] = cv2.resize(scaled[indexOfFaceDetected], 
                                                                    (input_image_size,input_image_size),
                                                                    interpolation=cv2.INTER_CUBIC)
                                    scaled[indexOfFaceDetected] = facenet.prewhiten(scaled[indexOfFaceDetected])
                                    scaled_reshape.append(scaled[indexOfFaceDetected].reshape(-1,input_image_size,input_image_size,3))
                                    feed_dict = {   images_placeholder: scaled_reshape[indexOfFaceDetected], 
                                                    phase_train_placeholder: False}
                                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                    
                                    predictions = model.predict_proba(emb_array)
                                    best_class_indices = np.argmax(predictions, axis=1)
                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                    faceName = self.getFaceNameFromFacesListByIndex(facesList, best_class_indices[0])

                                    self.printRectangleToImage(frame,boundingBoxesOfDetectedFace[indexOfFaceDetected][0], 
                                                                boundingBoxesOfDetectedFace[indexOfFaceDetected][1],
                                                                boundingBoxesOfDetectedFace[indexOfFaceDetected][2], 
                                                                boundingBoxesOfDetectedFace[indexOfFaceDetected][3], "blue")

                                    self.printRectangleToImageBackground(frame,boundingBoxesOfDetectedFace[indexOfFaceDetected][0], 
                                                                boundingBoxesOfDetectedFace[indexOfFaceDetected][1],
                                                                boundingBoxesOfDetectedFace[indexOfFaceDetected][2], 
                                                                boundingBoxesOfDetectedFace[indexOfFaceDetected][3], "blue",
                                                                faceName)

                                    self.printTextToImage(  frame,faceName,
                                                            boundingBoxesOfDetectedFace[indexOfFaceDetected][0],
                                                            boundingBoxesOfDetectedFace[indexOfFaceDetected][3] + 20,
                                                            "white" )
                                    
                                    print('Predicción: ',predictions)
                                    print('best class indices: ',best_class_indices)
                                    print("best class probabilities ",best_class_probabilities[0])
                                    print("face ", faceName)
                            #else:
                                #print('Unable to align, no faces')
                    
                        sec = curTime - prevTime
                        prevTime = curTime
                        fps = 1 / (sec)
                        strFPS = 'FPS: %2.3f' % fps
                        self.printTextToImage(frame,strFPS,20,50,"black" )
                        cv2.imshow('Video', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        #else:
                        #   print('No video')
                    video_capture.release()
                    # #video writer
                    out.release()
                    cv2.destroyAllWindows()
                except Exception as inst:
                    print('exception ',inst)

    def initParams(self, args):
        self.alignDirectory  = args.alignDirectory
        self.facesDirectory = args.facesDirectory
        self.modelDirectory = args.modelDirectory
        self.modelFile = args.modelFile
        self.pairFileDirectory = args.pairFileDirectory
        self.pairFile= args.pairFile
        self.cameraNumber= args.cameraNumber

    def parse_arguments(self,argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--alignDirectory', type=str, help='Align directory', default="align")
        parser.add_argument('--facesDirectory', type=str, help='Faces directory', default="..\\lfw\imagenesRostros")
        parser.add_argument('--modelDirectory', type=str, help='Model directory', default="..\\models\\20180402-114759")
        parser.add_argument('--modelFile', type=str, help='Model file', default="20180402-114759.pb")
        parser.add_argument('--pairFileDirectory', type=str, help='Pair file directory', default="..\\models\\20180402-114759")
        parser.add_argument('--pairFile', type=str, help='Pair file directory', default="lfw_classifier1000x35.pkl")    
        parser.add_argument('--cameraNumber', type=int, help='cameraNumber', default=0)
        return self.initParams(parser.parse_args(argv))

if __name__ == '__main__':
    testFacenet = TestFacenet()
    args=parse_arguments(sys.argv[1:])
    testFacenet.init()
