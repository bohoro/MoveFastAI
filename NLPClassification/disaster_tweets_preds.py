base_dir = "./"
#!ls "/content/drive/MyDrive/Colab Notebooks/DisasterTweets"

import pandas as pd
import numpy as np
import torch

test=pd.read_csv(base_dir+'test.csv', keep_default_na=False)
print(test.head(5))
test["fulltext"] = test["keyword"] + " " + test["location"] + " " + test["text"]
test_texts = test["fulltext"]


from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

model = RobertaForSequenceClassification.from_pretrained("roberta-large")
model.from_pretrained(base_dir + "v1model")

print()
print()

predictions = []
i = 1
for obs in test_texts:
    #print(obs)
    inputs = tokenizer(obs, return_tensors="pt")
    outputs = model(**inputs)
    pred = outputs.logits.cpu().detach().numpy()[0]
    #print(pred, np.argmax(pred))
    predictions.append(np.argmax(pred))

frame = { 'id': test['id'], 'target': pd.Series(predictions), 'fulltext': test["fulltext"] } 
preds = pd.DataFrame(frame)
print(preds.head(5))
preds.to_csv('predictions.csv')


submission = preds[['id', 'target']]
print(submission.head(5))
submission.to_csv('submission.csv',index=False)

