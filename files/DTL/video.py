import  argparse, sys, cv2, numpy as np, os

class Video:
    def __init__(self):
        self.videoName  = ""
        self.videoExtension=""
        self.videoFolder=""
        self.outDirectory = ""
        self.index= 1
        self.maxFileNumber=1
        self.cameraNumber=0
        self.imageExtension = ""
        self.startInFrame=1
        self.imageName=""
        self.saveWithAkey=0
        self.indexFrame=0

    def jpg_extension(self):
        self.imageExtension=".jpg"

    def initPath(self):
        if not (os.path.exists(self.fullOutDirectory)):
            print("outDirectory created: ",self.fullOutDirectory)
            os.makedirs(self.fullOutDirectory)

    def init(self, args):
        if (args.imageName is ""):
            self.imageName=args.videoName
        else:
            self.imageName=args.imageName
        self.videoName  = args.videoName
        self.videoExtension= args.videoExtension
        self.videoFolder=args.videoFolder
        self.outDirectory  = args.outDirectory
        self.index = args.index
        self.imageExtension= args.imageExtension
        self.index = args.index
        self.maxFileNumber = args.maxFileNumber
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.videoFullPath =os.path.join(self.path,self.videoFolder+"/"+self.videoName+self.videoExtension)
        self.fullOutDirectory = os.path.join(self.path,self.outDirectory,self.imageName+"\\")
        self.cameraNumber = args.cameraNumber
        self.startInFrame=args.startInFrame
        self.saveWithAkey= args.saveWithAkey
        self.initPath()
        self.capture()

    def capture(self):
        print(self.videoFullPath)
        if (os.path.isfile(self.videoFullPath))==False:
            print("File does not exist")
            return False
        
        cap = cv2.VideoCapture(self.videoFullPath)
        indexFrame = 0 
        print("press a to init and esc to quit")
        while True:
            ret,img = cap.read()
            cv2.imshow('Video', img)
            k=cv2.waitKey(10)& 0xff
            if k==27:
                break

            if (self.saveWithAkey== 1):
                if cv2.waitKey(33) == ord('a'):
                    print("init to save frames")
                    self.saveImage(img)
            else: 
                if (self.indexFrame>self.startInFrame):
                    if (self.index < self.maxFileNumber):
                        self.saveImage(img)

        cap.release()
        cv2.destroyAllWindows()

    def saveImage(self,img):
        self.indexFrame=self.indexFrame+1    
        fullName = self.fullOutDirectory+self.imageName+"_"+str(self.index)+self.imageExtension
        print('full name ',fullName)
        cv2.imwrite(fullName, img)
        self.index = self.index+1

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--videoFolder', type=str, help='Video folder', default="videos")
    parser.add_argument('--videoName', type=str, help='Video name', default="videoName")
    parser.add_argument('--videoExtension', type=str, help='Video extension', default=".mp4")
    parser.add_argument('--outDirectory', type=str, help='directory', default='..\lfw\imagenesDeEntrada')
    parser.add_argument('--index', type=int, help='Index', default=1)
    parser.add_argument('--maxFileNumber', type=int, help='maxFileNumber', default=1)
    parser.add_argument('--imageExtension', type=str, help='Image extension', default=".jpg")
    parser.add_argument('--cameraNumber', type=int, help='cameraNumber', default=0)
    parser.add_argument('--startInFrame', type=int, help='startInFrame', default=1)
    parser.add_argument('--saveWithAkey', type=int, help='save image when the user pressed a key', default=0)
    parser.add_argument('--imageName', type=str, help='image name', default="")
    return parser.parse_args(argv)

if __name__ == '__main__':
    video = Video()
    args=parse_arguments(sys.argv[1:])
    video.init(args)

# py DTL/video.py --videoName Alejandro_Giammattei --saveWithAkey 1
