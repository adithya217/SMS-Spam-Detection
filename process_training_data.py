from Preprocessor import Preprocessor

def main():
    pp = Preprocessor()
    #tdpath = 'dataset/train/train-data-1'
    tdpath = 'dataset/train/training-data-full'
    pp.process_training_data(tdpath)



main()
