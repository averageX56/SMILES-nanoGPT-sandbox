# small model — transformer core ~0.8M params
out_dir = 'out-small'
wandb_run_name = 'small'

# model
n_layer = 4
n_head = 4
n_embd = 128      
block_size = 128   
dropout = 0.0
bias = False
dataset = 'kv'   

batch_size = 128         
gradient_accumulation_steps = 1  
max_iters = 4000
lr_decay_iters = 1000
warmup_iters = 100
learning_rate = 1e-3
min_lr = 1e-4
eval_interval = 500
eval_iters = 100

