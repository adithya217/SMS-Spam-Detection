from Preprocessor import Preprocessor
from FeatureModel import FeatureModel
from Tester import Tester
import numpy as np

def main():
    pp = Preprocessor()
    print 'processing custom data, computing bows...'
    tdpath = 'dataset/test/sms-data'
    pp.process_custom_data(tdpath)
    
    fm = FeatureModel()
    print 'converting custom data to fvs...'
    fm.compute_custom_fv_matrix('custom')
    
    tdpath = 'bin_data/custom_fv.npy'
    cpath = 'bin_data/mnb-classifier.npy'
    data = np.load('bin_data/custom-data.npy').item()
    
    tester = Tester(tdpath,cpath)
    print 'predicting labels for custom data...'
    results = tester.predict_labels_for_custom_data(data)
    
    with open('output/results.txt','w') as textfile:
        for msg in results:
            line = '%s -> %s\n' % (msg,results[msg])
            textfile.write(line)
        
        textfile.close()
    
    print 'Results written to results.txt'
    

main()





