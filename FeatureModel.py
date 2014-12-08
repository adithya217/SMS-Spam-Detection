import numpy as np

class FeatureModel(object):
    
    def __init__(self):
        self.data = np.load('bin_data/data.npy').item()
        self.tokens = np.load('bin_data/tokens.npy').item()
        self.features = set([])
        self.min_threshold = 5
        self.max_threshold = 500
    
    
    def save_to_disk(self):
        np.save('bin_data/features',self.features)
    
    
    def extract_features(self):
        for token in self.tokens:
            tfc = self.tokens[token] # term frequency in entire corpus
            
            if (tfc < self.min_threshold) or (tfc > self.max_threshold):
                continue
            
            self.features.add(token)
        
    


def main():
    fm = FeatureModel()
    fm.extract_features()
    fm.save_to_disk()
    

main()
