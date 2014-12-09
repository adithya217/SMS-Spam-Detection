import numpy as np
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime

class Trainer(object):
    
    def __init__(self):
        td = np.load('bin_data/training_fv.npy').item()
        self.samples = td['samples']
        self.labels = td['labels']
        
    
    def train_MNB_classifier(self):
        classifier = MultinomialNB()
        classifier.fit(self.samples,self.labels)
        
        np.save('bin_data/mnb-classifier',classifier)




