from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import numpy as np
from statistics import mean
import torch, time, json
from datasets import load_dataset
import os
import argparse
import sys

from utils.utils import prepare_prompts, seed_everything, get_longbench_prompt


### some parameters set by user
batch_size = 128
max_new_tokens = 100
copy_of_prompt = 100
model_path = "../models/Phi-3.5-mini/Phi-3.5-mini" # "../models/llama3.1-8b/llama3.1-8b"


use_longbench_prompt = False
if use_longbench_prompt:
    # Here we load the selected prompts from LongBench
    prompts_all = get_longbench_prompt()

    # We show the distribution of the length of all the LongBench prompt
    lengths = [len(s) for s in prompts_all]
    bins = [500,1000, 2000, 5000, 10000, 50000, 2000000]  # Adjust bin ranges as needed
    binned_lengths = np.digitize(lengths, bins)
    bin_distribution = Counter(binned_lengths)

    # Display the distribution by bins
    print("Length distribution by bins:")
    for b in sorted(bin_distribution.keys()):
        print(f"Bin {bins[b-1]}-{bins[b]}: {bin_distribution[b]}")
else:
    # 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books
    prompts_all=[
        "The King is dead. Long live the Queen.",
        "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
        "The story so far: in the beginning, the universe was created.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "The sweat wis lashing oafay Sick Boy; he wis trembling.",
        "124 was spiteful. Full of Baby's venom.",
        "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",
        "I write this sitting in the kitchen sink.",
        "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",
    ] * copy_of_prompt


#sys.exit()

accelerator = Accelerator()

# load a base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index}
)
print(f"accelerator test: {accelerator.process_index}")
tokenizer = AutoTokenizer.from_pretrained(model_path)   
tokenizer.pad_token = tokenizer.eos_token

print(model)
sys.exit()

# sync GPUs and start the timer
accelerator.wait_for_everyone()    
start=time.time()

# divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(prompts_all) as prompts:
    results=dict(outputs=[], num_tokens=0)

    # have each GPU do inference in batches
    prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=batch_size)

    for prompts_tokenized in prompt_batches:
        outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=max_new_tokens)

        # remove prompt from gen. tokens
        outputs_tokenized=[ tok_out[len(tok_in):] 
            for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

        # count and decode gen. tokens 
        num_tokens=sum([ len(t) for t in outputs_tokenized ])
        outputs=tokenizer.batch_decode(outputs_tokenized)

        # store in results{} to be gathered by accelerate
        results["outputs"].extend(outputs)
        results["num_tokens"] += num_tokens

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs
results_gathered=gather_object(results)

if accelerator.is_main_process:
    timediff=time.time()-start
    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
