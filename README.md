# distributed_infer

This repo we build a distributed infer platform based on Huggingface [accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/distributed_inference). 

The original tut is using `diffusers` which is for image generation. Here we go for LLMs.

To use multiple GPUs, make sure to set `export CUDA_VISIBLE_DEVICES="0,1,2,3"`

To launch `accelerate` (which uses spwan), use `CUDA_VISIBLE_DEVICES="0" accelerate launch {script_name.py} --arg1 --arg2 ...` (CUDA part is optional if env var is set). Or `accelerate launch --multi_gpu {script_name.py} {--arg1} {--arg2} ...`

In this repo, `infer_with_batch_mp.py` and `infer_with_model_parallel.py` are based online sample code. 


# Test with LongBench

