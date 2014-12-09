import re
import numpy as np
import nltk

class Preprocessor(object):
    
    def __init__(self):
        self.tokens = {}
        self.token_filter = re.compile('^[a-zA-Z]+$',re.UNICODE) # match only alphabet strings
        self.dollar = re.compile('\$') # matches $ symbol in message -> currency in spam
        self.numeric_token = re.compile('^(\d+|\d+[.]\d+)$',re.UNICODE) # match only numeric token
        self.data = {}
        
        
    def process_tokens(self,tokens,counter):
        numerics_count = 0
        bow = {}
        
        for token in tokens:
            if re.match(self.numeric_token,token):
                numerics_count += 1
                
            if not re.match(self.token_filter,token):
                continue
            
            if token in self.tokens:
                self.tokens[token] += 1
            else:
                self.tokens[token] = 1
            
            if token in bow:
                bow[token] += 1
            else:
                bow[token] = 1
        
        self.data[counter]['nt_count'] = numerics_count
        
        return bow
    
    
    def process_extra_features(self,line,counter):
        dollar_count = len(re.findall(self.dollar,line))
        msg_length = len(line)
        
        self.data[counter]['$_count'] = dollar_count
        self.data[counter]['msg_length'] = msg_length
        
    
    def process_message(self,line,counter):
        try:
            tokens = nltk.word_tokenize(line.decode('utf-8'))
            bow = self.process_tokens(tokens,counter)
            
            self.data[counter]['bow'] = bow
            self.process_extra_features(line,counter)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print 'Error in tokenizing: %s' % line
            
    
    def process_training_data(self,path):
        with open(path) as textfile:
            counter = 0
            for line in textfile:
                line = line.rstrip('\n')
                data = line.split('\t')
                
                self.data[counter] = {'label':data[0]}
                self.process_message(data[1],counter)
                
                counter += 1
            
            np.save('bin_data/training-tokens',self.tokens)
            np.save('bin_data/training-data',self.data)
    
    
    def process_test_data(self,path):
        with open(path) as textfile:
            counter = 0
            for line in textfile:
                line = line.rstrip('\n')
                data = line.split('\t')
                
                self.data[counter] = {'label':data[0]}
                self.process_message(data[1],counter)
                
                counter += 1
            
            np.save('bin_data/testing-data',self.data)
    
    
    def process_custom_data(self,path):
        with open(path) as textfile:
            counter = 0
            for line in textfile:
                line = line.rstrip('\n')
                
                self.data[counter] = {'msg':line}
                self.process_message(line,counter)
                
                counter += 1
            
            np.save('bin_data/custom-data',self.data)
    
    
    
    
            



