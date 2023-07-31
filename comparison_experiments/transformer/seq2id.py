# -*- coding: utf-8 -*-

"""
Created on Sat Dec 24 14:55:00 2022
@author: Jiani Ma
tokenizer: one token method 
word2Sequence() class: encode the str sentences 
"""
import re 

def tokenizer(fasta): 
    seq = re.findall(r'.{1}', fasta)
    return seq 

class Word2Sequence():     
    UNK_TAG = "UNK"
    PAD_TAG = "PAD" 
    UNK = 0 
    PAD = 1     
    
    def __init__(self):
        self.dic = {self.UNK_TAG:self.UNK, 
                     self.PAD_TAG:self.PAD} 
        self.count = {} 
        self.inverse_dict = {}

    # add the word from the sequence to the count dictionary, initially calculate the word frequency 
    def fit(self,sequence):
        for word in sequence: 
            self.count[word] = self.count.get(word,0) + 1  # if the word does not exist, then return 0, else +1 

    # build the vocabulary according to the count dictionary 
    def build_vocab(self,max_features):         
        self.count = {word:value for word,value in self.count.items()}
        #if max_num is not None: 
            #self.count = {word:value for word,value in self.count.items() if value < max_num}            
        #sort the dictionary according to the reverse order 
        tmp_count = sorted(self.count.items(),key=lambda x:x[-1],reverse=True)[0:max_features]
        self.count = dict(tmp_count)
        # fill the dctionary with the aid of the count dictionary 
        for word, _ in self.count.items():
            self.dic[word] = len(self.dic)                     
        self.inverse_dict = dict(zip(self.dic.values(),self.dic.keys()))   

    #transform the text sequence into the encoding sequencing    
    def transform(self,sentence,max_len): 
        # PAD the remaining sequence if it is not long enough 
        if len(sentence) < max_len: 
            sentence = sentence + [self.PAD_TAG] * (max_len-len(sentence))            
        # slice the original sequence if it is longer than the max features 
        if len(sentence) > max_len:
            sentence = sentence[0:max_len]            
        return [self.dic.get(word,self.UNK) for word in sentence]    

    # inverse transform the encoding sequencing into the text sequence 
    def inverse_transform(self,indice):
        return [self.inverse_dict.get(idx) for idx in indice] 

    def __len__(self):
        return len(self.dic)







    



