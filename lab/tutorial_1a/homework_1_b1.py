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
world_size = 3
os.environ["MASTER_PORT"] = "29502"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 3
device = "cpu"
microbatch_size = batch_size // world_size  # initializing microbatch size

# initialize tokenizer (rank 1 and rank 2 deal with the tokenizer)
if rank == 0 or rank == 2:
    tokenizer = SPTokenizer()  # sentence piece tokenizer

# make the tokenizer
if rank == 0 or rank == 2:
    tokenizer = SPTokenizer()

if rank == 0:
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)  # No skip
    iter_ds = iter(ds)
elif rank == 1:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads,
                     device=device, n_layers=n_layers, ctx_size=seq_l)
else:
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads, device=device, n_layers=n_layers,
                         ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
    iter_ds = iter(ds)

optim = Adam(net.parameters(), lr=8e-4)

# allocate tensors for communication
input_batch = torch.empty((microbatch_size, seq_l, dmodel), device=device)
input_grad = torch.empty((microbatch_size, seq_l, dmodel), device=device)

# training the model using pipeline parallelism with microbatches asynchronously
for itr in range(5000):
    optim.zero_grad()

    if itr % 100 == 0:  # orint every 100 iterations
        print(f"Rank {rank} | Iteration {itr} | Training in Progress...")

    # forward pass
    if rank == 0:
        out_full = next(iter_ds).to(device)  # get entire batch from dataset
        out_full = net.embed(out_full)  # embed the aforementioned batch

        # splits the full batch into microbatches and sends it to rank 1
        for i in range(0, out_full.size(0), microbatch_size):
            # gets the current microbatch (i)
            out = out_full[i:i + microbatch_size]
            # passes the retrieved microbatch to rank 1
            req = dist.isend(out.to("cpu"), 1, tag=itr)
            # waits until the pass is done
            req.wait()

    elif rank == 1:
        # deals with microbatches and passes their output to rank 2
        for i in range(0, batch_size, microbatch_size):
            # gets embedded microbatch from rank 0
            req = dist.irecv(input_batch, 0, tag=itr)
            # waits on hold until the microbatch is fully received
            req.wait()

            inp_batch = input_batch.to(device)
            # track gradients
            inp_batch.requires_grad_()

            out = net(inp_batch)  # forward pass

            # pass output to rank 2
            req = dist.isend(out.to("cpu"), 2, tag=itr)
            req.wait()

    elif rank == 2:
        target = next(iter_ds).to(device)
        for i in range(0, batch_size, microbatch_size):
            req = dist.irecv(input_batch, 1, tag=itr)
            req.wait()

            inp_batch = input_batch.to(device)
            inp_batch.requires_grad_()  # Track gradients

            target_microbatch = target[i:i + microbatch_size]
            logits = net(input_batch)  # Forward pass
            loss = causalLLMLoss(logits, target_microbatch, tokenizer.vocab_size)

            if i == 0:
                print(f"Iteration {itr}, Loss: {loss.item()}")

            loss.backward()

    # backward pass
    if rank == 2:
        # pass gradients asynchronously to rank 1
        req = dist.isend(input_batch.grad.to("cpu"), 1, tag=itr)
        # wait untilk finished
        req.wait()

    elif rank == 1:
        # receive gradients that have been passed from rank 2
        req = dist.irecv(input_grad, 2, tag=itr)
        # wait until finished
        req.wait()

        # gradients backpropagation
        out.backward(input_grad.to(device))

        # pass gradietns to rank 0
        req = dist.isend(input_batch.grad.to("cpu"), 0, tag=itr)
        # wait until finished
        req.wait()

    elif rank == 0:
        # receive output gradients passed from rank 1
        req = dist.irecv(input_grad, 1, tag=itr)
        # wait until finished
        req.wait()

        # final backpropagation
        out.backward(input_grad.to(device))

        # synchronisation before updating parameters
    dist.barrier()
    optim.step()
    torch.cuda.empty_cache() if device == "cuda" else None
