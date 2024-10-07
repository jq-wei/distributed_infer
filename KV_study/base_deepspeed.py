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
    # Load the model on meta tensors
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with deepspeed.OnDevice(dtype=torch.float16, device="meta", enabled=True):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    # Define the checkpoint dict. You may need to convert *.safetensors to
    # *.bin for this work. Make sure you get all the *.bin and *.pt files in
    # the checkpoint_files list.
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
    start_time = time.time()
    inputs = tokenizer.encode("DeepSpeed is", return_tensors="pt").to(
        f"cuda:{local_rank}"
    )
    outputs = model.generate(inputs, max_new_tokens=200000)
    output_str = tokenizer.decode(outputs[0])
    end_time = time.time()
    print("DeepSpeed-inference time:", end_time - start_time)
    print(f"ds infer output: {output_str}")

if __name__ == "__main__":
    #run_zero_inference() # zero is pretty slow for now
    run_deepspeed_inference()