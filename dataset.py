import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def create_dataset():
    texts = load_dataset("opus_books", "en-fr", split='train')
    
    src_lang = 'en'
    tgt_lang = 'fr'
    
    texts = texts.map(
        lambda x: {
            'en': x['translation']['en'],
            'fr': x['translation']['fr']
        }
    )
    
    # Total number of samples
    max_size = len(texts)
    # max_size = 10000
    
    # Percentage of samples from the total number that will be used for training
    train_frac = 0.8 # Percen
    
    # Calculation of exact number of training and testing samples
    train_size = int(train_frac * max_size)
    test_size = max_size - train_size
    
    # Selection of the data for training and testing according to the number
    # of training and testing samples
    train_texts = texts.select(range(train_size))
    test_texts = texts.select(range(train_size, train_size+test_size))
    
    class TranslationDataset(torch.utils.data.Dataset):
        def __init__(self, data, src_lang='en', tgt_lang='fr'):
            self.data = data
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            return {
                "src": self.data[idx][src_lang],
                "tgt": self.data[idx][tgt_lang]
            }
    
    train_dataset = TranslationDataset(train_texts)
    test_dataset = TranslationDataset(test_texts)
    
    return train_dataset, test_dataset

