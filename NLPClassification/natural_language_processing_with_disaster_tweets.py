"""
Natural Language Processing with Disaster Tweets.ipynb

Natural Language Processing with Disaster Tweets
* Predict which Tweets are about real disasters and which ones are not

This script fine tunes a LM for classification.
"""

base_dir = "./"

import pandas as pd
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# #############################################################################
## Basic EDA
# #############################################################################
tweet= pd.read_csv(base_dir+'train.csv', keep_default_na=False)
test=pd.read_csv(base_dir+'test.csv', keep_default_na=False)
print(tweet.head(5))
tweet.info()

# #############################################################################
## Create Train, Dev, and Test Datasets
# #############################################################################

tweet["fulltext"] = tweet["keyword"] + " " + tweet["location"] + " " + tweet["text"]
test["fulltext"] = test["keyword"] + " " + test["location"] + " " + test["text"]

data = tweet["fulltext"]
labels = tweet["target"]

train_texts, val_texts, train_labels, val_labels = train_test_split(data, labels, test_size=.1)
test_texts = test["fulltext"]
train_texts.head(5)

# #############################################################################
## Encodings 
# #############################################################################
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# #############################################################################
## PyTorch Datasets
# #############################################################################

class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = DisasterDataset(train_encodings, train_labels.tolist())
val_dataset = DisasterDataset(val_encodings, val_labels.tolist())


# #############################################################################
## Setup Training Env
# #############################################################################
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=7,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=10,                # number of warmup steps for learning rate scheduler
    weight_decay=0.015,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

print()
print("Loading distilbert-base-uncased...")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
print("Complete")
print()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
)

print()
print("Start training ")
trainer.train()
print()
print(trainer.evaluate())
print()


# #############################################################################
## Save Model
# #############################################################################
model.save_pretrained(base_dir + "v1model")
model.eval()

# #############################################################################
# Leak data testing
# #############################################################################
leak = pd.read_csv(base_dir+'socialmedia-disaster-tweets-DFE.csv', encoding='latin_1', keep_default_na=False)
leak['target'] = (leak['choose_one']=='Relevant').astype(int)
leak['id'] = leak.index
leak = leak[['id', 'keyword','location','text', 'target']]

### test with leaked data ###
leak["fulltext"] = leak["keyword"] + " " + leak["location"] + " " + leak["text"]
final_test_encodings = tokenizer(leak["fulltext"].tolist(), truncation=True, padding=True)
final_test_dataset = DisasterDataset(final_test_encodings, leak['target'].tolist())
print()
print("Start Leak Test ")
print(trainer.evaluate(final_test_dataset))
print()