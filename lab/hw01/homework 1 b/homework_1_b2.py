import os
from sys import argv

import torch
import torch.distributed as dist
from simplellm.dataloaders import TinyStories  # dataset
from simplellm.llama import LLamaLastStage, LLamaFirstStage, LLamaStage  # get our models
from simplellm.losses import causalLLMLoss  # loss
from simplellm.tokenizers import SPTokenizer  # tokenizer
from torch.optim import Adam

rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"
world_size = 6  # 3 ranks x 2 pipelines
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 6
device = "cpu"
microbatch_size = batch_size // world_size  # initializing microbatch size
pipeline_size = 6 // 2  # initializing pipeline batch size

# two separate pipeline groups
pipeline_1 = dist.new_group([0, 1, 2])
pipeline_2 = dist.new_group([3, 4, 5])

# parallel data group for first stage of both pipelines
parallel_data_group = dist.new_group([0, 3])

# get group for pipeline communication
if rank in [0, 1, 2]:
    pipeline_group = pipeline_1
elif rank in [3, 4, 5]:
    pipeline_group = pipeline_2
else:
    pipeline_group = None

# handle tokenizer (rank 0 and rank 2 handle tokenization)
if rank in [0, 2, 3, 5]:
    tokenizer = SPTokenizer()

# model pipeline
if rank in [0, 3]:  # first stage
    net = LLamaFirstStage(
        tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
        device=device, n_layers=n_layers, ctx_size=seq_l)

    # each pipeline should process different data
    dataset_offset = 0 if rank == 0 else 5000  # rank 0 starts at 0, rank 3 starts at 5000
    ds = TinyStories(tokenizer, batch_size=pipeline_size, seq_l=seq_l, skip=dataset_offset)
    iter_ds = iter(ds)
elif rank in [1, 4]:  # middle stage
    net = LLamaStage(
        dmodel=dmodel, num_heads=num_heads,
        device=device, n_layers=n_layers, ctx_size=seq_l)
else:  # last stage
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                         device=device, n_layers=n_layers, ctx_size=seq_l)
    # each pipeline should process different data
    dataset_offset = 0 if rank == 2 else 5000
    ds = TinyStories(tokenizer, batch_size=pipeline_size, seq_l=seq_l, skip=dataset_offset)
    iter_ds = iter(ds)

optim = Adam(net.parameters(), lr=8e-4)

# allocate tensors for communication
input_batch = torch.empty((microbatch_size, seq_l, dmodel), device=device)
input_grad = torch.empty((microbatch_size, seq_l, dmodel), device=device)

# training model
for itr in range(5000):
    optim.zero_grad()

    # forward pass
    if rank in [0, 3]:
        out_full = next(iter_ds).to(device)  # get next full batch from dataset
        out_full = net.embed(out_full)  # embed full batch

        # splits the full batch into microbatches and passes it to rank 1, 4
        for i in range(0, out_full.size(0), microbatch_size):
            # get current microbatch
            out = out_full[i:i + microbatch_size]
            # pass microbatch either to rank 1 or 4
            req = dist.isend(out.to("cpu"), 1 if rank == 0 else 4, tag=itr, group=pipeline_group)
            req.wait()

    elif rank in [1, 4]:
        for i in range(0, pipeline_size, microbatch_size):
            req = dist.irecv(input_batch, 0 if rank == 1 else 3, tag=itr, group=pipeline_group)
            req.wait()

            input_batch = input_batch.to(device)
            input_batch.requires_grad_()
            input_batch.retain_grad()

            out = net(input_batch)

            req = dist.isend(out.to("cpu"), 2 if rank == 1 else 5, tag=itr, group=pipeline_group)
            req.wait()

    elif rank in [2, 5]:  # final stage
        target = next(iter_ds).to(device)
        for i in range(0, pipeline_size, microbatch_size):
            req = dist.irecv(input_batch, 1 if rank == 2 else 4, tag=itr, group=pipeline_group)
            req.wait()

            input_batch = input_batch.to(device)
            input_batch.requires_grad_()

            target_microbatch = target[i:i + microbatch_size]
            logits = net(input_batch)
            loss = causalLLMLoss(logits, target_microbatch, tokenizer.vocab_size)

            if i == 0:
                print(f"Iteration {itr}, Loss: {loss.item()}")

            loss.backward()

    # backward pass
    if rank in [2, 5]:
        req = dist.isend(input_batch.grad.to("cpu"), 1 if rank == 2 else 4, tag=itr, group=pipeline_group)
        req.wait()

    elif rank in [1, 4]:
        req = dist.irecv(input_grad, 2 if rank == 1 else 5, tag=itr, group=pipeline_group)
        req.wait()
        out.backward(input_grad.to(device))

        req = dist.isend(input_batch.grad.to("cpu"), 0 if rank == 1 else 3, tag=itr, group=pipeline_group)
        req.wait()

    elif rank in [0, 3]:
        req = dist.irecv(input_grad, 1 if rank == 0 else 4, tag=itr, group=pipeline_group)
        req.wait()
        out.backward(input_grad.to(device))

    # all ranks synchronized before optimization
    if rank in [0, 3]:
        dist.barrier(group=parallel_data_group)

    # average gradient from all pipelines after backward passing is finished
    if rank in [0, 3]:
        for param in net.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=parallel_data_group)
                param.grad /= 2  # averaging gradients across pipelines

    optim.step()
    torch.cuda.empty_cache() if device == "cuda" else None
