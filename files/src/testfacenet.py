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
    
    def getHumanNames(self):
        HumanNames = [name for name in os.listdir(self.facesDirectory) if name.find(".txt") == -1]
        return HumanNames.sort()

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
                HumanNames = self.getHumanNames()
                print('Listado de rostros',HumanNames)
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
                            bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]

                            if nrof_faces > 0:
                                print('Detected Face Number: %d' % nrof_faces)
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(frame.shape)[0:2]
                                cropped = []
                                scaled = []
                                scaled_reshape = []
                                bb = np.zeros((nrof_faces,4), dtype=np.int32)
                                for i in range(nrof_faces):
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
                                    
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                    if (best_class_probabilities[0] >= 0.75):
                                        print('Mejor probabilidad >= 0.75: ',best_class_probabilities)
                                    else:
                                        print('Mejor probabilidad < 0.75: ',best_class_probabilities)
                                    #plot result idx under box
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    #print('result: ', best_class_indices[0])
                                    #print(best_class_indices)
                                    #print(HumanNames)
                                    for H_i in HumanNames:
                                        #print(H_i)
                                        if HumanNames[best_class_indices[0]] == H_i:
                                            result_names = HumanNames[best_class_indices[0]]
                                            print('name: ',result_names)
                                            cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 255), thickness=1, lineType=2)
                            #else:
                                #print('Unable to align, no faces')
                    
                        sec = curTime - prevTime
                        prevTime = curTime
                        fps = 1 / (sec)
                        str = 'FPS: %2.3f' % fps
                        text_fps_x = len(frame[0]) - 150
                        text_fps_y = 20
                        cv2.putText(frame, str, (text_fps_x, text_fps_y),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
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
