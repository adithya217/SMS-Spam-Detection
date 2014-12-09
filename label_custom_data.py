from Preprocessor import Preprocessor
from FeatureModel import FeatureModel
from Trainer import Trainer
import numpy as np

def main():
    pp = preprocessor()
    
    tdpath = 'dataset/train/training-data-full'
    pp.process_training_data(tdpath)
    
    tdpath = 'dataset/test/sms-data'
    pp.process_custom_data(tdpath)
    
    fm = FeatureModel()
    
    fm.extract_features()
    fm.compute_fv_matrix('training')
    fm.compute_custom_fv_matrix('custom')
    
    trainer = Trainer()
    trainer.train_MNB_classifier()
    
    
