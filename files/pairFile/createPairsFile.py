import  argparse, sys, random, numpy as np, os

class CreatePairsFile:
    def __init__(self):
        self.baseDir  = ""
        self.separate  = ""
        self.filePath  = ""
        self.folders  =[]
    
    def initParams(self, args):
        self.baseDir= args.baseDir
        self.separate= args.separate
        self.filePath = args.filePath
        self.folders  =  self.getFolders(self.baseDir)
    
    def init(self):
        self.removeFileIfExist()
        self.pairs()
        self.shuffle()

    def removeFileIfExist(self):
        if os.path.isfile(self.filePath):
            print("file exists, removed")

    def createPairs(self,totalPairs):
        result = []
        for i in range(0, totalPairs-1):
            for y in range(i+1, totalPairs):
                #print str(i)+'-'+str(y)
                result.append([i,y])
        
        return result

    def getFolders(self,baseDir):
        folders = [name for name in os.listdir(baseDir) if name.find(".txt") == -1]  
        return sorted(folders, reverse=False)

    def pairs(self):
        for name in self.folders:    
            path, dirs, files = next(os.walk(self.baseDir+name))
            file_count = len(files)
            if file_count>2:
                file_count=2
                
            if(file_count>0):
                a = []
                for file in os.listdir(self.baseDir + name):
                    a.append(file)

                with open(self.filePath,"a") as f:
                    pair_two_list = self.createPairs(file_count)
                    randomChoice = random.choice(a)
                    temp = random.choice(a).split(self.separate)[0] # This line may vary depending on how your images are named.
                    print('name',name,'count',file_count,'randomChoice',randomChoice,'temp',temp)
                    for l1 in pair_two_list:
                        n0 = a[l1[0]].split(self.separate)[1].lstrip("0").rstrip(".png")
                        n1 = a[l1[1]].split(self.separate)[1].lstrip("0").rstrip(".png")
                        content = temp + "\t" + n0 + "\t" + n1 + "\n"
                        f.write(content)

    def shuffle(self): 
        for i,name in enumerate(self.folders):
            remaining = self.folders

            del remaining[i] # deletes the file from the list, so that it is not chosen again
            other_dir = random.choice(remaining)

            with open(self.filePath,"a") as f:
                flag=1
                for i in range(1):
                    file1 = random.choice(os.listdir(self.baseDir + name))
                    file2 = random.choice(os.listdir(self.baseDir+ other_dir))
                    content = name + "\t" + file1.split(self.separate)[1].lstrip("0").rstrip(".png") + "\t" + other_dir + "\t" + file2.split(self.separate)[1].lstrip("0").rstrip(".png") + "\n"
                    f.write(content)    

    def parse_arguments(self, argv):
        parser = argparse.ArgumentParser()
        currentPath = os.getcwd()
        baseDir     = os.path.join(currentPath,"lfw\\imagenesRostros\\")
        pairFileDir = os.path.join(currentPath,"pairFile\\pairs.txt")
        parser.add_argument('--baseDir', type=str, help='Folder path', default=baseDir)
        parser.add_argument('--filePath', type=str, help='file name', default=pairFileDir)
        parser.add_argument('--separate', type=str, help='Separate string', default="_0")
        return self.initParams(parser.parse_args(argv))

if __name__ == '__main__':
    createPairsFile = CreatePairsFile()
    createPairsFile.parse_arguments(sys.argv[1:])
    createPairsFile.init()
