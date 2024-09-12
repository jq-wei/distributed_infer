# distributed_infer

This repo we build a distributed infer platform based on Huggingface [accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/distributed_inference). 

The original tut is using `diffusers` which is for image generation. Here we go for LLMs.

To use multiple GPUs, make sure to set `export CUDA_VISIBLE_DEVICES="0,1,2,3"`

To launch `accelerate` (which uses spwan), use `CUDA_VISIBLE_DEVICES="0" accelerate launch {script_name.py} --arg1 --arg2 ...` (CUDA part is optional if env var is set). Or `accelerate launch --multi_gpu {script_name.py} {--arg1} {--arg2} ...`

In this repo, `infer_with_batch_mp.py` and `infer_with_model_parallel.py` are based online sample code. 

# How to use

The main code for my use case is `infer_LB_batch.py`. Before running the code, update the `batch_size, max_new_tokens, copy_of_prompt, model_path` in the script. And then run `accelerate launch --multi_gpu infer_LB_batch.py`.


# Test with LongBench

Now the pipeline to fetch the prompts in specific format of each class within LongBench is ready, but since most of the prompt is very long, it can not run on V100 + llama3.1-8b.