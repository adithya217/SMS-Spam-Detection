from Tester import Tester
from datetime import datetime

def main():
    started = datetime.now()
    
    tdpath = 'bin_data/testing_fv.npy'
    cpath = 'bin_data/mnb-classifier.npy'
    
    tester = Tester(tdpath,cpath)
    tester.test_classifier()
    
    finished = datetime.now()
    
    print 'Started at: ',started
    print 'Finished at: ',finished
    print 'Time taken: ',(finished-started)
    

main()
