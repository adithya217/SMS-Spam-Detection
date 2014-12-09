import numpy as np
from datetime import datetime

class Tester(object):
    
    def __init__(self):
        td = np.load('bin_data/testing_fv.npy').item()
        self.samples = td['samples']
        self.labels = td['labels']
    
    
    def test_MNB_classifier(self):
        classifier = np.load('bin_data/mnb-classifier.npy').item()
        print classifier.score(self.samples,self.labels)
        
        

def main():
    started = datetime.now()
    
    tester = Tester()
    tester.test_MNB_classifier()
    
    finished = datetime.now()
    
    print 'Started at: ',started
    print 'Finished at: ',finished
    print 'Time taken: ',(finished-started)
    

main()
