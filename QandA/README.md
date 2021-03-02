# Extractive Question Answering 
## Huggingface Inference Example

Straight forward inference example using the huggingface "question-answering" transformer pipeline.

```
python QandADemo.py

# What war did Lionel Colin Matthews fight in?
Answer: 'World War II', score: 0.9704, start: 96, end: 108

# What medals did he win?
Answer: 'George Cross', score: 0.3807, start: 1823, end: 1835
```

## installation 

```
conda create -n huggingface python=3.7
conda activate huggingface
conda install -c huggingface transformers
pip install importlib_metadata
conda install pytorch torchvision -c pytorch
```
