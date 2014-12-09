import numpy as np
from datetime import datetime

class Tester(object):
    
    def __init__(self,dpath,cpath):
        td = np.load(dpath).item()
        self.samples = td['samples']
        self.labels = td['labels']
        
        self.classifier = np.load(cpath).item()
    
    
    def test_classifier(self):
        print self.classifier.score(self.samples,self.labels)
    
    
    def predict_label(self,sample):
        return self.classifier.predict(sample)
        
        
    def predict_labels_for_custom_data(self,data):
        results = {}
        index = 0
        
        for fv in self.samples:
            msg = data[index]['msg']
            
            #print msg,fv
            
            results[msg] = self.predict_label(fv)
            index += 1
        
        return results
        
        
        

