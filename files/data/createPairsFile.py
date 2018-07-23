import os
import random


baseDir = "../lfw/imageneesRostros/"
separate = "_0"
folders = getFolders(baseDir)
filePath = "pairs.txt"

def createPairs(totalPairs):
    result = []
    for i in range(0, totalPairs-1):
        for y in range(i+1, totalPairs):
            #print str(i)+'-'+str(y)
            result.append([i,y])
    
    return result

def getFolders(baseDir):
    folders = [name for name in os.listdir(baseDir) if name.find(".txt") == -1]  
    return sorted(folders, reverse=False)



def pairs():
    for name in folders:    
        path, dirs, files = next(os.walk(baseDir+name))
        file_count = len(files)
        if file_count>2:
            file_count=2
            
        if(file_count>0):
            a = []
            for file in os.listdir(baseDir + name):
                a.append(file)

            with open(filePath,"a") as f:
                pair_two_list = createPairs(file_count)
                randomChoice = random.choice(a)
                temp = random.choice(a).split(separate)[0] # This line may vary depending on how your images are named.
                print('name',name,'count',file_count,'randomChoice',randomChoice,'temp',temp)
                for l1 in pair_two_list:
                    n0 = a[l1[0]].split(separate)[1].lstrip("0").rstrip(".png")
                    n1 = a[l1[1]].split(separate)[1].lstrip("0").rstrip(".png")
                    content = temp + "\t" + n0 + "\t" + n1 + "\n"
                    f.write(content)

def shuffle(): 
    for i,name in enumerate(folders):
        remaining = folders

        del remaining[i] # deletes the file from the list, so that it is not chosen again
        other_dir = random.choice(remaining)

        with open(filePath,"a") as f:
            flag=1
            for i in range(1):
                file1 = random.choice(os.listdir(baseDir + name))
                file2 = random.choice(os.listdir(baseDir+ other_dir))
                content = name + "\t" + file1.split(separate)[1].lstrip("0").rstrip(".png") + "\t" + other_dir + "\t" + file2.split(separate)[1].lstrip("0").rstrip(".png") + "\n"
                f.write(content)    

pairs()
shuffle()