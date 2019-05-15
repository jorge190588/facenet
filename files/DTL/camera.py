import  argparse, sys, cv2, numpy as np, os

class Camera:
    def __init__(self):
        self.first_name  = ""
        self.last_name  = ""
        self.folder=""
        self.extension=""
        self.name=""
        self.index=0
        self.maxFileNumber=0
        self.directory = ""
        self.path=""
        self.extension=""
        self.cameraNumber=0
    
    def jpg_extension(self):
        self.extension=self.extension

    def capture(self):
        self.cap = cv2.VideoCapture(args.cameraNumber)  
        while True:
            ret,imageCaptured = self.cap.read()
            cv2.imshow('Video', imageCaptured)
            k=cv2.waitKey(10)& 0xff
            if k==27:
                break

            if (self.index <= self.maxFileNumber):
                fileName = self.path+self.name+"_"+str(self.index)+self.extension
                print('Full Name: ',fileName)
                cv2.imwrite(fileName, imageCaptured)
                self.index = self.index+1
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()
    
    def initPath(self):
        if not (os.path.exists(self.path)):
            print("path created: ",self.path)
            os.makedirs(self.path)

    def init(self, args):
        self.first_name  = args.first_name
        self.last_name  = args.last_name
        self.directory = args.directory
        self.extension= args.extension
        self.index = args.index
        self.maxFileNumber = args.maxFileNumber
        self.folder=args.first_name+"_"+args.last_name
        self.name=args.first_name+"_"+args.last_name
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.path = os.path.join(self.path,self.directory,self.folder+"\\")
        self.cameraNumber = args.cameraNumber
        self.initPath()
        self.capture()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_name', type=str, help='First name', default="first")
    parser.add_argument('--last_name', type=str, help='Last name', default="last")
    parser.add_argument('--directory', type=str, help='directory', default='..\lfw\imagenesDeEntrada')
    parser.add_argument('--index', type=int, help='Index', default=1)
    parser.add_argument('--maxFileNumber', type=int, help='maxFileNumber', default=1)
    parser.add_argument('--extension', type=str, help='extension', default=".jpg")
    parser.add_argument('--cameraNumber', type=int, help='cameraNumber', default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    camera = Camera()
    args=parse_arguments(sys.argv[1:])
    camera.init(args)

# example: py camera.py --first_name jorge --last_name santos --maxFileNumber 10 
# example: py camera.py --first_name jorge --last_name santos --maxFileNumber 10 --directory ..\imgs\imagenesDeEntrada
   