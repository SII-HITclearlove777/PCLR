import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from lrp_qwen.model.load_model import load_model

from PIL import Image
import math

import re


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, query_cls = load_model(model_path)
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    im_start = tokenizer.decode(tokenizer.im_start_id, skip_special_tokens=True)
    im_end = tokenizer.decode(tokenizer.im_end_id, skip_special_tokens=True)
    sys = im_start + "system\nYou are a helpful assistant." + im_end + "\n"
    for i, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        question = line['text']
        qs = question.replace('<image>', '').strip()
        instruction = line['text'].replace('<image>', '')
        # im_start = "<|im_start|>"
        # im_end = "<|im_end|>"
        if 'image' in line.keys():
            image_file = line["image"]
            if '.jpg.jpg' in image_file:
                image_file = image_file.replace('.jpg.jpg', '.jpg')
            image_file = os.path.join(args.image_folder, image_file.replace('./',''))
            prompt = sys + '{}user\n<img>{}</img>{}{}\n{}assistant\n'.format(im_start, image_file, qs, im_end, im_start)
            # prompt = 'user: <img>{}</img>{}\nassistant:'.format(image_file, qs)
            images = Image.open(image_file).convert('RGB')
            images = model.base_model.transformer.visual.image_transform(images).unsqueeze(0)
            iamge_path = image_file
        else:
            prompt = sys + '{}user\n{}{}\n{}assistant\n'.format(im_start, qs, im_end, im_start)
            # prompt = 'user: {}\nassistant:'.format(qs)
            images = None
            iamge_path = ''
        inputs = tokenizer(prompt, return_tensors='pt')
        instruction_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(model.ori_device)
        input_ids_size = inputs.data['input_ids'].size(1)
        input_ids = inputs.input_ids.to(model.ori_device)
        llm_query, vit_query = query_cls.get_query(instruction, iamge_path)
        with torch.inference_mode():
            pred = model.generate(input_ids=input_ids,
                                instruction_ids=instruction_ids,
                                llm_query=llm_query,
                                vit_query=vit_query,
                                images=images,
                                do_sample=False,
                                num_beams=1,
                                max_new_tokens=32,
                                use_cache=True)
        outputs = [tokenizer.decode(_[input_ids_size:].cpu(),skip_special_tokens=True) for _ in pred]
        text = outputs[0].replace(im_start, "").replace(im_end, "").replace("user", "").replace("assistant", "").strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": text,
                                   "answer_id": ans_id}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
