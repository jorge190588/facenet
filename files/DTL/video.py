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
        self.videoFullPath =self.videoFolder+"/"+self.videoName+self.videoExtension
        self.outDirectory  = args.outDirectory
        self.index = args.index
        self.imageExtension= args.imageExtension
        self.index = args.index
        self.maxFileNumber = args.maxFileNumber
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.fullOutDirectory = os.path.join(self.path,self.outDirectory,self.imageName+"\\")
        self.cameraNumber = args.cameraNumber
        self.startInFrame=args.startInFrame
        self.initPath()
        self.capture()

    def capture(self):
        if (os.path.isfile(self.videoFullPath))==False:
            print("File does not exist")
            return False
        
        cap = cv2.VideoCapture(self.videoFullPath)
        indexFrame = 0 
        while True:
            ret,img = cap.read()
            cv2.imshow('Video', img)
            k=cv2.waitKey(10)& 0xff
            if k==27:
                break
            
            indexFrame=indexFrame+1
            if (indexFrame>self.startInFrame):
                if (self.index < self.maxFileNumber):
                    fullName = self.fullOutDirectory+self.imageName+"_"+str(self.index)+self.imageExtension
                    print('full name ',fullName)
                    # save image
                    cv2.imwrite(fullName, img)
                    self.index = self.index+1

        cap.release()
        cv2.destroyAllWindows()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--videoFolder', type=str, help='Video folder', default="video")
    parser.add_argument('videoName', type=str, help='Video name', default="videoName")
    parser.add_argument('--videoExtension', type=str, help='Video extension', default=".mp4")
    parser.add_argument('--outDirectory', type=str, help='directory', default='..\lfw\imagenesDeEntrada')
    parser.add_argument('--index', type=int, help='Index', default=1)
    parser.add_argument('--maxFileNumber', type=int, help='maxFileNumber', default=1)
    parser.add_argument('--imageExtension', type=str, help='Image extension', default=".jpg")
    parser.add_argument('--cameraNumber', type=int, help='cameraNumber', default=0)
    parser.add_argument('--startInFrame', type=int, help='startInFrame', default=1)
    parser.add_argument('--imageName', type=str, help='image name', default="")
    return parser.parse_args(argv)

if __name__ == '__main__':
    video = Video()
    args=parse_arguments(sys.argv[1:])
    video.init(args)
