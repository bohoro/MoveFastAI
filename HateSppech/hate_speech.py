# install transformers from https://github.com/huggingface/transformers

''' 
model example is from - https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english?text=I+like+you.+I+love+you 
@article{aluru2020deep,
  title={Deep Learning Models for Multilingual Hate Speech Detection},
  author={Aluru, Sai Saket and Mathew, Binny and Saha, Punyajoy and Mukherjee, Animesh},
  journal={arXiv preprint arXiv:2004.06465},
  year={2020}
}
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")

model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")

# example of non-hate speech - see the paper above (Table 1) for valid hate speech examples, I refuse to put them in code.
inputs = tokenizer("We are having fun and learning a lot!", return_tensors="pt")
outputs = model(**inputs)
print(outputs)
