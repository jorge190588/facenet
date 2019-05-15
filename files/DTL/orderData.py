import  os,random, sys, argparse,shutil

class OrderData:
    def __init__(self):
        self.inputDirectory  = ""
        self.outputDirectory  = ""
        self.path =  os.getcwd()
    
    def order(self):  
        for name in os.listdir(self.inputDirectory):
            if name.find(".txt") == -1:
                self.createDirectory(name)
                self.renameImages(name)
    
    def createDirectory(self, folderName):
        outputDirectoryByname = self.outputDirectory+'\\'+folderName
        if (os.path.exists(outputDirectoryByname)):
            shutil.rmtree(outputDirectoryByname)
        os.makedirs(outputDirectoryByname)

    def renameImages(self, folderName):
        indice = 1
        for file in os.listdir(self.inputDirectory + folderName):
            oldFile = self.path+'\\'+self.inputDirectory+folderName+'\\'+ file
            newFile= self.outputDirectory+'\\'+folderName+'\\'+folderName+'_'+ '%04d' % int(indice) +'.jpg'
            indice =indice+1
            os.rename(oldFile, newFile)
            print('imagen original ',oldFile,', newFile',newFile)
    
    def init(self, args):
        self.inputDirectory  = args.inputDirectory
        self.outputDirectory  = args.outputDirectory
        self.order()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDirectory', type=str, help='Input directory', default="..\\lfw\\imagenesDeEntrada\\")
    parser.add_argument('--outputDirectory', type=str, help='Last name', default="..\\lfw\\imagenesOrdenadas")
    return parser.parse_args(argv)

if __name__ == '__main__':
    orderData = OrderData()
    args=parse_arguments(sys.argv[1:])
    orderData.init(args)

# example: py camera.py --first_name jorge --last_name santos --maxFileNumber 10 
# example: py camera.py --first_name jorge --last_name santos --maxFileNumber 10 --directory ..\imgs\imagenesDeEntrada
   