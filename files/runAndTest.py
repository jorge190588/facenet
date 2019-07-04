import os
import subprocess
import sys
import argparse
from DTL.orderData import OrderData
from pairFile.createPairsFile import CreatePairsFile

class RunAndTest:
    def __init__(self):
        self.order()
        self.setEnvironmentVariable()
        self.align()
        self.createPairs()
        self.validate()
        self.clasify()
        self.test()

    def order(self):
        print("step 1. order data")
        orderData = OrderData()
        orderData.parse_arguments(None)
        orderData.order()
    
    def setEnvironmentVariable(self):
        facenetPath= os.environ["PYTHONPATH"] = os.getcwd()+'\\src'
        os.environ["PYTHONPATH"]= facenetPath
        print("step 2. Set PYTHONPATH=",os.environ["PYTHONPATH"])
        if facenetPath not in sys.path:
            sys.path.append(facenetPath)

    def align(self):
        print("step 3. Align data")
        from src.align.align_dataset_mtcnn import AlignDatasetMtcnn
        alignDatasetMtcnn= AlignDatasetMtcnn()
        alignDatasetMtcnn.parse_arguments(None)
        alignDatasetMtcnn.main()
    
    def createPairs(self):
        print("step 4. create pairs file")
        createPairsFile = CreatePairsFile()
        createPairsFile.parse_arguments(None)
        createPairsFile.init()

    def validate(self):
        print("step 5. validate on lfw")
        from src.validate_on_lfw import Validate_on_lfw
        validate_on_lfw = Validate_on_lfw()
        validate_on_lfw.parse_arguments(None)
        validate_on_lfw.init()
    
    def clasify(self):
        print("step 6. clasifier")
        from src.classifier import Classifier
        classifier=Classifier()
        classifier.parse_arguments(None)
        classifier.main()

    def test(self):
        print("step 7. test")
        from src.testfacenet import TestFacenet
        test=TestFacenet()
        test.parse_arguments(None)
        test.init()

runAndTest= RunAndTest()