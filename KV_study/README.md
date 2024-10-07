In this folder, we do some detailed analysis on KV data, layers, heads, etc. 

# we first do single gpu inference with simple_prompts

For single GPU, run `accelerate launch --num_processes=1 base.py`. Don't know why CUDA_VISIBLE_DEVISE seems not working. 


# data saving format

We now save the hidden state for each inference, around 4k seq_len in total.

Llama 3.1 has 32 layers, each layer has 32 heads.

We save the K,V (before ROPE), attn_weights of each layer，for each token. 

For each token, the shapes are:
1. K (V) is: bsz (1), self.num_key_value_heads (8), q_len (1), self.head_dim (128). Here q_len = 1 since it is for each token
2. attn_weights is: bsz (1), self.num_heads (32), 1 (current_token), q_len.  For example, the Q*K^T = (1,32,1,128) * (1,32,128,seq_len)  -> (1,32,1,seq_len). This attn_weight is, for current token, the attention for all the previous tokens, for all the head.


# run the basic model parallelism

In this one, we want to use basic model parallelism to equally partition the model to 4 GPUs. In this case each GPU will have more vram, so we can infer with longer context. For now, we don't care about the bubbles. 

To launch the script, use `deepspeed --num_gpu 4 deepspeed_infer_LB.py`.