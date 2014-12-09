from Trainer import Trainer
from datetime import datetime

def main():
    started = datetime.now()
    
    trainer = Trainer()
    trainer.train_MNB_classifier()
    
    finished = datetime.now()
    
    print 'Started at: ',started
    print 'Finished at: ',finished
    print 'Time taken: ',(finished-started)
    

main()
