from Preprocessor import Preprocessor
from sys import argv

def main():
    pp = Preprocessor()
    tdpath = argv[1]
    pp.process_training_data(tdpath)



main()
