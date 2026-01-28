      
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder='./model')

with open('/path/merge.jsonl', 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

def compute_similarities(response, labels):
    emb = model.encode([response, labels])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return sim



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
    sim = compute_similarities(response, labels)
    if sim >= 0.8:
        cor += 1
    
print(cor/sum)


