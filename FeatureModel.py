import numpy as np

class FeatureModel(object):
    
    def __init__(self):
        pass
    
    
    def extract_features(self):
        tokens = np.load('bin_data/training-tokens.npy').item()
        features = set([])
        
        min_threshold = 5
        max_threshold = 500
        
        for token in tokens:
            tfc = tokens[token] # term frequency in entire corpus
            
            if (tfc < min_threshold) or (tfc > max_threshold):
                continue
            
            features.add(token)
        
        np.save('bin_data/features',features)
    
    
    def get_fv_from_msg_item(self,msg_item,features,feature_count):
        bow = msg_item['bow']
            
        ef1 = msg_item['$_count']
        ef2 = msg_item['nt_count']
        ef3 = msg_item['msg_length']
        
        fv = np.zeros((1,feature_count))
        
        feature_index = 0
        for feature in features:
            if feature in bow:
                occurence = bow[feature]
                fv[0][feature_index] = occurence
            
            feature_index += 1
        
        fv[0][feature_index] = ef1
        feature_index += 1
        fv[0][feature_index] = ef2
        feature_index += 1
        fv[0][feature_index] = ef3
        
        return fv
    
    
    def compute_fv_matrix(self,purpose):
        data = np.load('bin_data/'+purpose+'-data.npy').item()
        
        features = np.load('bin_data/features.npy').item()
        feature_count = len(features) + 3
        
        samples = []
        labels = []
        
        for msg_index in data:
            msg_item = data[msg_index]
            
            fv = self.get_fv_from_msg_item(msg_item,features,feature_count)
            
            samples.append(fv.flatten())
            labels.append(msg_item['label'])
        
        samples = np.array(samples)
        labels = np.array(labels)
        
        computed_data = { 'samples' : samples, 'labels' : labels }
        np.save('bin_data/'+purpose+'_fv',computed_data)
    
    
    def compute_custom_fv_matrix(self,purpose):
        data = np.load('bin_data/'+purpose+'-data.npy').item()
        
        features = np.load('bin_data/features.npy').item()
        feature_count = len(features) + 3
        
        samples = []
        
        for msg_index in data:
            msg_item = data[msg_index]
            
            fv = self.get_fv_from_msg_item(msg_item,features,feature_count)
            
            samples.append(fv.flatten())
        
        samples = np.array(samples)
        
        computed_data = { 'samples' : samples, 'labels' : np.array([]) }
        np.save('bin_data/'+purpose+'_fv',computed_data)
            
            
            
        

