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
    def __init__(self):
        self.alignDirectory  = ""
        self.facesDirectory = ""
        self.modelDirectory = ""
        self.modelFile = ""
        self.modelFilePath = ""
        self.pairFile=""
        self.pairFileDirectory= ""

    def init(self, args):
        path =os.path.dirname(os.path.realpath(__file__))
        self.alignDirectory  = os.path.join(path,args.alignDirectory)
        self.facesDirectory  = os.path.join(path,args.facesDirectory)
        self.modelFile = args.modelFile
        self.modelDirectory = os.path.join(path,args.modelDirectory)
        self.modelFilePath = os.path.join(path,self.modelDirectory,self.modelFile)
        self.pairFileDirectory = os.path.join(path,args.pairFileDirectory)
        self.pairFile = args.pairFile
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

    def printTextToImage(self, image, text, position_x, position_y, colorName):
        if (colorName is "black"):
            colorR=0
            colorG=0
            colorB=0
        if (colorName is "red"):
            colorR=0
            colorG=0
            colorB=255
        cv2.putText(image, text, (position_x, position_y),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (colorR, colorG, colorB), thickness=1, lineType=2)        

    def printRectangleToImage(self, image,position_x1,position_y1,position_x2,position_y2, colorName):
        if (colorName is "green"):
            colorR=0
            colorG=255
            colorB=0
        cv2.rectangle(image, (position_x1, position_y1), (position_x2, position_y2), (colorR, colorG, colorB), 2)

    def runTest(self):    
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, self.alignDirectory)
                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                margin = 44
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
                classifier_filename_exp = os.path.expanduser(self.pairFilePath)
                print("classifier is load ")
                try:
                    with open(classifier_filename_exp, 'rb') as infile:
                        u = pickle._Unpickler(infile)
                        u.encoding = 'latin1'
                        (model, class_names) = u.load()
                        #(model, class_names) = infile.load()
                        print('load classifier file-> %s' % classifier_filename_exp)

                    video_capture = cv2.VideoCapture(0) #'./test.mp4'
                    video_capture.set(3,4920)
                    video_capture.set(4,3080)
                    c = 0

                    # #video writer
                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(4920,3080))

                    print('Start Recognition!')
                    prevTime = 0
                    while True:
                        ret, frame = video_capture.read()
                        #if (frame != None):
                        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                        curTime = time.time()+1    # calc fps
                        timeF = frame_interval

                        if (c % timeF == 0):
                            find_results = []
                            if frame.ndim == 2:
                                frame = facenet.to_rgb(frame)
                            frame = frame[:, :, 0:3]
                            boundingBoxesOfDetectedFaces, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                            numberOfFacesDeteted = boundingBoxesOfDetectedFaces.shape[0]

                            if numberOfFacesDeteted > 0:
                                print('Detected Face Number: %d' % numberOfFacesDeteted)
                                det = boundingBoxesOfDetectedFaces[:, 0:4]
                                img_size = np.asarray(frame.shape)[0:2]
                                cropped = []
                                scaled = []
                                scaled_reshape = []
                                bb = np.zeros((numberOfFacesDeteted,4), dtype=np.int32)
                                
                                for i in range(numberOfFacesDeteted):
                                    emb_array = np.zeros((1, embedding_size))
                                    bb[i][0] = det[i][0]
                                    bb[i][1] = det[i][1]
                                    bb[i][2] = det[i][2]
                                    bb[i][3] = det[i][3]

                                    # inner exception
                                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                        #print('face is inner of range!')
                                        continue

                                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                    cropped[i] = facenet.flip(cropped[i], False)
                                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                            interpolation=cv2.INTER_CUBIC)
                                    scaled[i] = facenet.prewhiten(scaled[i])
                                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                    predictions = model.predict_proba(emb_array)
                                    print('Predicción: ',predictions)
                                    best_class_indices = np.argmax(predictions, axis=1)
                                    #print('best class indices: ',best_class_indices)
                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                    self.printRectangleToImage(frame,bb[i][0], bb[i][1],bb[i][2], bb[i][3], "green")
                                    if (best_class_probabilities[0] >= 0.75):
                                        print('Mejor probabilidad >= 0.75: ',best_class_probabilities[0])
                                    else:
                                        print('Mejor probabilidad < 0.75: ',best_class_probabilities[0])
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    
                                    faceName = self.getFaceNameFromFacesListByIndex(facesList, best_class_indices[0])
                                    self.printTextToImage(frame,faceName,text_x,text_y,"red" )
                                    print("face ", faceName)
                            #else:
                                #print('Unable to align, no faces')
                    
                        sec = curTime - prevTime
                        prevTime = curTime
                        fps = 1 / (sec)
                        strFPS = 'FPS: %2.3f' % fps
                        text_fps_x = len(frame[0]) - 150
                        text_fps_y = 20
                        self.printTextToImage(frame,strFPS,text_fps_x,text_fps_y,"black" )
                        # c+=1
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


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignDirectory', type=str, help='Align directory', default="align")
    parser.add_argument('--facesDirectory', type=str, help='Faces directory', default="..\lfw\imagenesRostros")
    parser.add_argument('--modelDirectory', type=str, help='Model directory', default="..\models\\20180402-114759")
    parser.add_argument('--modelFile', type=str, help='Model file', default="20180402-114759.pb")
    parser.add_argument('--pairFileDirectory', type=str, help='Pair file directory', default="..\models\\20180402-114759")
    parser.add_argument('--pairFile', type=str, help='Pair file directory', default="lfw_classifier1000x35.pkl")    
    parser.add_argument('--cameraNumber', type=int, help='cameraNumber', default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    testFacenet = TestFacenet()
    args=parse_arguments(sys.argv[1:])
    testFacenet.init(args)
