# -*- coding: utf-8 -*-
"""Natural Language Processing with Disaster Tweets.ipynb

# Natural Language Processing with Disaster Tweets
## Predict which Tweets are about real disasters and which ones are not
"""

### Run Once to setup data ###

# !pip install -q kaggle
### Uploade the kaggle.json API key
#from google.colab import files
#files.upload() 
#!mkdir ~/.kaggle
#!cp kaggle.json ~/.kaggle/
#!chmod 600 ~/.kaggle/kaggle.json
#!kaggle competitions download -c nlp-getting-started # -f "/content/drive/MyDrive/Colab Notebooks/DisasterTweets/"
#!mv *.csv "/content/drive/MyDrive/Colab Notebooks/DisasterTweets/"

####################################################
# Load the Drive helper and mount
####################################################
#from google.colab import drive
#drive.mount('/content/drive')
base_dir = "./"
#!ls "/content/drive/MyDrive/Colab Notebooks/DisasterTweets"

import pandas as pd
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

"""## Basic EDA"""

tweet= pd.read_csv(base_dir+'train.csv', keep_default_na=False)
test=pd.read_csv(base_dir+'test.csv', keep_default_na=False)
tweet.head(15)

tweet.info()


"""## Create Train, Dev, and Test Datasets"""

from sklearn.model_selection import train_test_split
tweet["fulltext"] = tweet["keyword"] + " " + tweet["location"] + " " + tweet["text"]
test["fulltext"] = test["keyword"] + " " + test["location"] + " " + test["text"]

data = tweet["fulltext"]
labels = tweet["target"]

train_texts, val_texts, train_labels, val_labels = train_test_split(data, labels, test_size=.2)
test_texts = test["fulltext"]
train_texts.head(5)

"""## Encodings """

#!pip install transformers
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

"""## PyTorch Datasets"""

import torch

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
#test_dataset = DisasterDataset(test_encodings, test_labels)

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

trainer.train()

print(trainer.evaluate())

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs

model.save_pretrained(base_dir + "v1model")
#!ls -alsh "/content/drive/MyDrive/Colab Notebooks/DisasterTweets"
