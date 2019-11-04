import os
from io import open
import torch

from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(1,'/home/ascalercio/nlp/repo/tensor2tensor/')
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators import text_encoder

class LM_Dataset(Dataset):
    
    def __init__(self, data_dict):
        self.encoder = text_encoder.SubwordTextEncoder()
        self.encoder._load_from_file('./lm_subword_text_encoder')        
        #self.dictionary = SequenceVocabulary()
        self.train = torch.tensor(self.encoder.encode(data_dict['train'])).type(torch.int64)
        self.valid = torch.tensor(self.encoder.encode(data_dict['val'])).type(torch.int64)
        self.test = torch.tensor(self.encoder.encode(data_dict['test'])).type(torch.int64)