import os
import random

baseDir = "../lfw/imagenesRostros/"
separate = "_"
filePath = "../data/pairs.txt"

def createPairs(totalPairs):
    result = []
    for index in range(0, totalPairs-1):
        for index2 in range(index+1, totalPairs):
            # print(str(index)+'-'+str(index2))
            result.append([index,index2])
    
    return result

def getFolders(baseDir):
    folders = [name for name in os.listdir(baseDir) if name.find(".txt") == -1]  
    return sorted(folders, reverse=False)

def pairs():
    for folderName in folders:    
        
        folderPath, dirs, files = next(os.walk(baseDir+folderName))
        
        print('dirs',dirs)
        filesNumberInFolder = len(files)
        if filesNumberInFolder>2:
            filesNumberInFolder=2
        
        print('Folder Path',folderPath)
        print('Folder name: ',folderName)
        print('Files number in folder ',filesNumberInFolder)
        
        if(filesNumberInFolder>0):
            filenamesInFolder= []
            
            for file in os.listdir(folderPath):
                filenamesInFolder.append(file)
            
            with open(filePath,"a") as f:
                pair_two_list = createPairs(filesNumberInFolder)
                randomFilename = random.choice(filenamesInFolder)
                randomFolder = randomFilename.split(separate)[0]+'_'+randomFilename.split(separate)[1]
                print('folderName',folderName,'count',filesNumberInFolder,'randomFilename',randomFilename,'randomFolder',randomFolder)
                
                for pair in pair_two_list:                     
                    print('filenamesInFolder[pair[0]]',filenamesInFolder[pair[0]])  
                    number0 = filenamesInFolder[pair[0]].split(separate)[2].strip(".png")
                    number1 = filenamesInFolder[pair[1]].split(separate)[2].strip(".png")
                    content = randomFolder + "\t" + number0 + "\t" + number1 + "\n"
                    f.write(content)

def shuffle(): 
    for i,name in enumerate(folders):
        remaining = folders

        del remaining[i] # deletes the file from the list, so that it is not chosen again
        other_dir = random.choice(remaining)

        with open(filePath,"a") as f:
            for i in range(1):
                file1 = random.choice(os.listdir(baseDir + name))
                file2 = random.choice(os.listdir(baseDir+ other_dir))
                content = name + "\t" + file1.split(separate)[2].strip(".png") + "\t" + other_dir + "\t" + file2.split(separate)[2].strip(".png") + "\n"
                f.write(content)    

 
folders = getFolders(baseDir)
pairs()
shuffle()