import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
from myllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from myllava.conversation import conv_templates, SeparatorStyle
from myllava import conversation as conversation_lib
from lrp_llava_hf.model.load_model import load_model
from myllava.utils import disable_torch_init
from myllava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math



def preprocess_v1(
    source,
    conv,
    tokenizer: transformers.PreTrainedTokenizer,
):
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    instructions = []
    if roles[source[0]["from"]] != conv.roles[0]:
        source = source[1:]
    instruction = source[0]["value"].replace("<image>\n","")
    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        conv.append_message(role, sentence["value"])
    conv.append_message(conv.roles[1], None)
    conversations.append(conv.get_prompt())
    instructions.append(instruction)
    input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    instruction_ids = tokenizer(
            instructions,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    return input_ids, instruction_ids


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, image_processor = load_model(model_path)
    conv = conversation_lib.default_conversation.copy()
    conv.system = 'You are a helpful assistant.'
    
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    count = 0 
    for line in tqdm(questions):
        count += 1
        idx = line["id"]
        answer = line["conversations"][-1]["value"]
        source = line["conversations"][:-1]
        input_ids, instruction_ids = preprocess_v1(source, conv, tokenizer)

        if 'image' in line.keys():
            image_file = line["image"]
            image = Image.open(os.path.join(image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor = image_tensor.unsqueeze(0).half().cuda()
        else:
            image_tensor = None
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.to(device='cuda'),
                instruction_ids=instruction_ids.to(device='cuda'),
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)
        input_token_len = input_ids.shape[1]
        output_ids = output_ids[:, input_token_len:]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs,
                                   "answer": answer,
                                   "answer_id": ans_id}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    args = parser.parse_args()

    eval_model(args)
