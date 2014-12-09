import numpy as np
from datetime import datetime

class Tester(object):
    
    def __init__(self,dpath,cpath):
        td = np.load(dpath).item()
        self.samples = td['samples']
        self.labels = td['labels']
        
        self.classifier = np.load(cpath).item()
    
    
    def test_MNB_classifier(self):
        print self.classifier.score(self.samples,self.labels)
        
        

