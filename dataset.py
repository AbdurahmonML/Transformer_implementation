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

def preprocess_and_dataloader(train_dataset, test_dataset, tokenizer = "Helsinki-NLP/opus-mt-en-fr", max_length = 64, batch_size = 64):
  
  tokenizer = AutoTokenizer.from_pretrained()
  
  # Add Begining of Sentence (BOS) token since it is absent in the tokenizer
  tokenizer.add_special_tokens({'bos_token': '<BOS>'})
  
  # Custom function to process and batch data
  def collate_fn(batch):
      # Prepend "<BOS>" token to each source text in the batch and tokenize it
      tokenized_src_texts = tokenizer.batch_encode_plus(
          ["<BOS> " + item['src'] for item in batch],  # Add "<BOS>" to the source text before tokenization
          padding="max_length",                        # Pad sequences to the max_length
          max_length=max_length,                       # Define the maximum length for padding/truncation
          truncation=True,                             # Truncate sequences longer than max_length
          return_tensors="pt"                          # Return a PyTorch tensor
      )['input_ids']                                   # Extract the 'input_ids' (token IDs)
  
      # Prepend "<BOS>" token to each target text in the batch and tokenize it
      tokenized_tgt_texts = tokenizer.batch_encode_plus(
          ["<BOS> " + item['tgt'] for item in batch],  # Add "<BOS>" to the target text before tokenization
          padding="max_length",                        # Pad sequences to the max_length
          max_length=max_length,                       # Define the maximum length for padding/truncation
          truncation=True,                             # Truncate sequences longer than max_length
          return_tensors="pt"                          # Return a PyTorch tensor
      )['input_ids']                                   # Extract the 'input_ids' (token IDs)
  
      # Return the tokenized and padded source and target texts as tensors
      return {
          'src': tokenized_src_texts,                  # Tokenized source sequences tensor
          'tgt': tokenized_tgt_texts                   # Tokenized target sequences tensor
      }
  
  # Create DataLoader for training data
  train_dataloader = DataLoader(
      train_dataset,                                   # Training dataset
      batch_size=batch_size,                           # Batch size for training
      collate_fn=collate_fn,                           # Custom collate function for tokenizing and batching
      shuffle=True                                     # Shuffle the data each epoch
  )
  
  # Create DataLoader for test data
  test_dataloader = DataLoader(
      test_dataset,                                    # Test dataset
      batch_size=batch_size,                           # Batch size for testing
      collate_fn=collate_fn,                           # Custom collate function for tokenizing and batching
      shuffle=False                                    # Do not shuffle the test data
  )

  return train_dataloader, test_dataloader
