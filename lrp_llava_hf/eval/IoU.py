import json
import re

with open('/path/merge.jsonl', 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

sum = len(dataset)
cor = 0
def extract_bbox(text):
    text = re.sub(r'\s+', '', text)
    pattern = r'(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+)'
    match = re.search(pattern, text)

    if match:
        return list(map(float, match.groups()))
    else:
        raise ValueError(f"{text}")

def clamp_bbox(box, min_val=0.0, max_val=1.0):
    return [
        max(min_val, min(max_val, box[0])),
        max(min_val, min(max_val, box[1])),
        max(min_val, min(max_val, box[2])),
        max(min_val, min(max_val, box[3])),
    ]

def iou(box1, box2):
    box1 = clamp_bbox(box1)
    box2 = clamp_bbox(box2)
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

for sample in dataset:
    response = sample["text"]
    labels = sample["answer"]
    try:
        response_box = extract_bbox(response)
        label_box = extract_bbox(labels)
        s = iou(response_box, label_box)
        if s >= 0.5:
            cor +=1
    except:
        print(response)
        pass
    
print(cor/sum)