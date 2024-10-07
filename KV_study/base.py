import json
import os
import sys

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import numpy as np
from statistics import mean
import torch, time, json


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import load_prompts_from_json, prepare_prompts


def main():
    # Path to the JSON file containing the prompts
    json_file_path = '/mnt/disk1/w84373270/distributed_infer_ch/data/samples/bin_2000_4000.json'

    # Load prompts
    prompts_all = load_prompts_from_json(json_file_path)
    print(f"Jay test: {len(prompts_all)}")
    print(f"Jay test: {len(prompts_all[0])}")
    #print(f"Jay test: {prompts_all[0]}")


    prompts_all = [prompts_all[0]]
    print(f"Jay test: {len(prompts_all[0])}")

    #exit()
    # Check if prompts were loaded successfully
    '''
    if prompts_all is not None:
        for i, prompt in enumerate(prompts_all):
            if i<= 4:
                print(f"Prompt {i+1}: {prompt}")

    '''


    #exit()

    ### some parameters set by user
    batch_size = 1
    max_new_tokens = 500
    min_new_tokens = 10
    model_path = "/mnt/disk1/w84373270/models/llama3.1-8b/llama3.1-8b" #"../models/Phi-3.5-mini/Phi-3.5-mini" 

    accelerator = Accelerator()

    print(f"proc: {accelerator.process_index}")

    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index}, 
        torch_dtype=torch.float16
    )
    print(f"accelerator test: {accelerator.process_index}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    #print(model)
    #sys.exit()

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        results=dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        print(f"Jay test prompts len: {len(prompts[0])}")
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=batch_size)
        print(f"Jay test num of prompts: {len(prompt_batches)}")
        print(f"Jay test len of first input_ids: {prompt_batches[0]['input_ids'].shape}")
        os.environ['infer_idx'] = '0'

        print(f"Jay test with prompt_batches: {len(prompt_batches)}")

        for prompts_tokenized in prompt_batches:
            #print(f"tokenized prompts: {prompts_tokenized}")
            outputs_tokenized=model.generate(**prompts_tokenized, 
                                             max_new_tokens=max_new_tokens, 
                                             min_new_tokens=min_new_tokens,
                                             use_cache=True, 
                                             temperature=0.45,
                                             top_k=50,
                                             top_p=1,
                                             repetition_penalty=1.03,
                                             num_return_sequences=1)

            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

            # count and decode gen. tokens 
            num_tokens=sum([ len(t) for t in outputs_tokenized ])
            outputs=tokenizer.batch_decode(outputs_tokenized)
            print(f"Jay test: {outputs}")
            print(f"Jay test: this is the {os.getenv('infer_idx')}th infer")
            torch.save(outputs_tokenized,f'/mnt/disk1/w84373270/distributed_infer_ch/KV_study/cache_data/infer_{os.getenv('infer_idx')}/tokenized_output.pth')


            current_infer = int(os.environ['infer_idx'])
            current_infer += 1
            os.environ['infer_idx'] = str(current_infer)

            #print(f"Jay test output_token shape: {len(outputs_tokenized)}")
            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens

        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered=gather_object(results)

    if accelerator.is_main_process:
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])

        # Print the decoded outputs
        for i, output in enumerate(results_gathered[0]["outputs"]):
            print(f"Generated text {i+1}: {output}")

        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
        


if __name__ == "__main__":
     main()