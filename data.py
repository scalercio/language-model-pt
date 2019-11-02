import os
from io import open
import torch
import tokenizer
import text_encoder

class LM_Dataset(Dataset):
    
    def __init__(self, data_dict):
        self.encoder = text_encoder.SubwordTextEncoder()
        self.encoder._load_from_file(os.environ['DATA_LM']+'/lm_subword_text_encoder')        
        #self.dictionary = SequenceVocabulary()
        self.train = torch.tensor(self.encoder.encode(data_dict['train'])).type(torch.int64)
        self.valid = torch.tensor(self.encoder.encode(data_dict['val'])).type(torch.int64)
        self.test = torch.tensor(self.encoder.encode(data_dict['test'])).type(torch.int64)