import torch
import numpy as np
import random
import os
from datasets import load_dataset
import json

def write_pretty_json(file_path, data):
    import json
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)


# batch, left pad (for inference), and tokenize
def prepare_prompts(prompts, tokenizer, batch_size=16):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_longbench_prompt():
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",  \
                    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("LB_config/dataset_prompt.json", "r"))
    dataset2maxlen = json.load(open("LB_config/dataset_maxlen.json", "r"))
    # predict on each dataset

    prompt_all = []

    for dataset in datasets:
        data = load_dataset('json', dataset, data_files =f"/mnt/disk1/w84373270/datasets/LongBench/data/{dataset}.jsonl", split = "train")

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        prompt_dataset = [prompt_format.format(**data_sample) for data_sample in data_all]

        prompt_all.extend(prompt_dataset)

    return prompt_all