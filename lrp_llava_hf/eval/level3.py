import json
import re

with open('/path/merge.jsonl', 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]


def clean(labels):
    labels = labels.split('\n', 1)[0]
    labels = labels.strip()
    if labels.endswith('.'):
        labels = labels[:-1]
    labels = labels.lower()
    return labels


sum = len(dataset)
cor = 0

for sample in dataset:
    response = clean(sample["text"])
    labels = clean(sample["answer"])
    if response == labels:
        cor +=1
    
print(cor/sum)