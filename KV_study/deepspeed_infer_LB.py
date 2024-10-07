import os
import sys

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
from collections import Counter
import numpy as np
from statistics import mean
import torch, time, json


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import load_prompts_from_json, prepare_prompts

import deepspeed



local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
model_path = "/mnt/disk1/w84373270/models/llama3.1-8b/llama3.1-8b"


def run_zero_inference():
    ds_config = {
        "fp16": {"enabled": True},
        "bf16": {"enabled": False},
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
            },
        },
        "train_micro_batch_size_per_gpu": 1,
    }
    # Share the DeepSpeed config with HuggingFace so we can properly load the
    # large model with zero stage 3
    hfdsc = HfDeepSpeedConfig(ds_config)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    )

    # Initialize DeepSpeed
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    # Run inference
    start_time = time.time()
    inputs = tokenizer.encode("DeepSpeed is", return_tensors="pt").to(
        f"cuda:{local_rank}"
    )
    outputs = model.generate(inputs, max_new_tokens=20)
    output_str = tokenizer.decode(outputs[0])
    end_time = time.time()
    print("ZeRO-inference time:", end_time - start_time)
    print(f"ZeRO output: {output_str}")


def run_deepspeed_inference():

    json_file_path = '/mnt/disk1/w84373270/distributed_infer_ch/data/samples/bin_10000_12000_part2.json'

    # Load prompts
    prompts_all = load_prompts_from_json(json_file_path)
    print(f"Jay test: {len(prompts_all)}")
    print(f"Jay test input len in words: {len(prompts_all[0].split())}")
    #print(f"Jay test: {prompts_all[0]}")

    # this var pick one prompt and name the folder to save
    infer_id = 22

    prompts_all = [prompts_all[infer_id]]


    # Load the model on meta tensors
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with deepspeed.OnDevice(dtype=torch.float16, device="meta", enabled=True):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)


    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    )


    # Initialize DeepSpeed
    model = deepspeed.init_inference(
        model,
        replace_with_kernel_inject=False,
        mp_size=world_size,
        dtype=torch.float16
    )

    # Run inference
    for prompt in prompts_all:

        os.environ['infer_idx'] = str(infer_id)
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(
            f"cuda:{local_rank}"
        )
        print(f"Jay test  len of token: {len(inputs)}")

        start_time = time.time()
        outputs = model.generate(inputs, 
                                max_new_tokens=max_new_tokens, 
                                min_new_tokens=min_new_tokens,
                                use_cache=True, 
                                temperature=0.15,
                                top_k=50,
                                top_p=1,
                                repetition_penalty=1.03,
                                num_return_sequences=1)

        print(f"Jay test input/ output: {inputs.shape} and {outputs.shape}")
        #exit()
        # remove prompt from gen. tokens
        outputs_tokenized= outputs[:, inputs.shape[1]:] 
        print(f"Jay test: {outputs_tokenized.shape}")


        print(f"Jay test outputs len: {len(outputs_tokenized[0])}")
        output_str = tokenizer.decode(outputs_tokenized[0], skip_special_tokens=True)
        end_time = time.time()
        print("DeepSpeed-inference time:", end_time - start_time)
        if local_rank==0:
            print(f"ds infer output from device cuda: {local_rank}: {output_str}")

        print(f"Jay test: this is the {os.getenv('infer_idx')}th infer")
        torch.save(outputs,f'/dev/shm/infer_{os.getenv('infer_idx')}/tokenized_output.pth')

        current_infer = int(os.environ['infer_idx'])
        current_infer += 1
        os.environ['infer_idx'] = str(current_infer)

if __name__ == "__main__":
    #run_zero_inference() # zero is pretty slow for now

    max_new_tokens=150
    min_new_tokens=2
    run_deepspeed_inference()