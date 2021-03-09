'''
disaster_tweets_preds.py is used for evaling the latest and greatet model. The script
loads the prettained model, evaluates against the training set and the leaked dataset.
Finally the script predicts agains the kaggle hold out file and creates a submission 
file submissions.csv.
'''

import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

base_dir = "./"

### use GPU if avialable ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# #############################################################################
# Load the tokenizer and the fine tuned model
# #############################################################################
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
print()
print("Loading Fine Tuned Model...")
model = DistilBertForSequenceClassification.from_pretrained(base_dir + "v1model")
# set to evaluation rather than training mode
model.eval() 
print("Complete")
print()

# #############################################################################
# sanity check on the training set
# #############################################################################
print()
print("Starting Sanity Checks...")

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
    
val_dataset=pd.read_csv(base_dir+'train.csv', keep_default_na=False)
val_dataset["fulltext"] = val_dataset["keyword"] + " " + val_dataset["location"] + " " + val_dataset["text"]
val_dataset_t = val_dataset["fulltext"]
val_encodings = tokenizer(val_dataset_t.tolist(), truncation=True, padding=True)
val_dataset = DisasterDataset(val_encodings, val_dataset['target'].tolist())

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=10,                # number of warmup steps for learning rate scheduler
    weight_decay=0.015,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

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
    #train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
)

print(trainer.evaluate())
print("Complete")
print()

# #############################################################################
# END sanity check on the training set
# #############################################################################

# #############################################################################
# Leak data testing
# #############################################################################
print()
print("Leak Testing")
leak = pd.read_csv(base_dir+'socialmedia-disaster-tweets-DFE.csv', encoding='latin_1', keep_default_na=False)
leak['target'] = (leak['choose_one']=='Relevant').astype(int)
leak['id'] = leak.index
leak = leak[['id', 'keyword','location','text', 'target']]
# print(leak.head(5))

### test with leaked data ###
leak["fulltext"] = leak["keyword"] + " " + leak["location"] + " " + leak["text"]
final_test_encodings = tokenizer(leak["fulltext"].tolist(), truncation=True, padding=True)
final_test_dataset = DisasterDataset(final_test_encodings, leak['target'].tolist())


print(trainer.evaluate(final_test_dataset))
print("Complete")
print()
# #############################################################################
# END Leak data testing
# #############################################################################


# #############################################################################
# Create predictions
# #############################################################################
print()
print("Starting Predictions")
test=pd.read_csv(base_dir+'test.csv', keep_default_na=False)
test["fulltext"] = test["keyword"] + " " + test["location"] + " " + test["text"]
test_texts = test["fulltext"]

test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings.to(device)


with torch.no_grad():
    preds = model(**test_encodings)
print("Complete")


logits = preds.logits.detach().cpu().numpy()
pred_flat = np.argmax(logits, axis=1).flatten()

print()
print("Writing Prediction and Submissions Files")
frame = { 'id': test['id'], 'target': pd.Series(pred_flat), 'fulltext': test["fulltext"] } 
preds_out = pd.DataFrame(frame)
preds_out.to_csv('predictions.csv')
submission = preds_out[['id', 'target']]
submission.to_csv('submission.csv',index=False)
print("Complete")
print()