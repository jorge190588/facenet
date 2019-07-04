import  os,random, sys, argparse,shutil

class OrderData:
    def __init__(self,_inputDirectory=None, _outputDirectory=None):
        self.inputDirectory  = _inputDirectory
        self.outputDirectory  = _outputDirectory
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
            shutil.copyfile(oldFile, newFile)  
            print('imagen original ',oldFile,', newFile',newFile)
    
    def initParams(self, args):
        self.inputDirectory  = args.inputDirectory
        self.outputDirectory  = args.outputDirectory

    def parse_arguments(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--inputDirectory', type=str, help='Input directory', default="lfw\\imagenesDeEntrada\\")
        parser.add_argument('--outputDirectory', type=str, help='Last name', default="lfw\\imagenesOrdenadas")
        return self.initParams(parser.parse_args(argv))

if __name__ == '__main__':
    orderData = OrderData()
    print(sys.argv[1:])
    orderData.parse_arguments(sys.argv[1:])
    orderData.order()

# example: py camera.py --first_name jorge --last_name santos --maxFileNumber 10 
# example: py camera.py --first_name jorge --last_name santos --maxFileNumber 10 --directory ..\imgs\imagenesDeEntrada
   