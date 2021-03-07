base_dir = "./"
#!ls "/content/drive/MyDrive/Colab Notebooks/DisasterTweets"

import pandas as pd
#import numpy as np
import torch

test=pd.read_csv(base_dir+'test.csv', keep_default_na=False)
test["fulltext"] = test["keyword"] + " " + test["location"] + " " + test["text"]
test_texts = test["fulltext"]


from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

model = RobertaForSequenceClassification.from_pretrained("roberta-large")
model.from_pretrained(base_dir + "v1model")

print()
print()
print(test_texts[0])
inputs = tokenizer(test_texts[0], return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
print()
print(test_texts[1])
inputs = tokenizer(test_texts[1], return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
print()
print(test_texts[2])
inputs = tokenizer(test_texts[2], return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
print()
print(test_texts[3])
inputs = tokenizer(test_texts[2], return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
print()
print(test_texts[4])
inputs = tokenizer(test_texts[2], return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
print()
print(test_texts[5])
inputs = tokenizer(test_texts[2], return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)