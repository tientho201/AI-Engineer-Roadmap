"""train_fsdp.py — chạy: torchrun --nproc_per_node=4 train_fsdp.py"""
import torch, torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.data.distributed import DistributedSampler
import os

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main():
    setup()
    rank = dist.get_rank()

    model = build_model().cuda()
    model = FSDP(
        model,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32),   # giữ fp32 khi all-reduce -> ổn định hơn
        use_orig_params=True,              # cần cho torch.compile
    )

    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        sampler.set_epoch(epoch)          # BẮT BUỘC, nếu quên mọi epoch shuffle giống nhau
        for xb, yb in loader:
            loss = model(xb.cuda(), yb.cuda())
            loss.backward()
            model.clip_grad_norm_(1.0)
            opt.step(); opt.zero_grad(set_to_none=True)

        if rank == 0:                     # CHỈ rank 0 log và lưu checkpoint
            print(f"epoch {epoch} loss {loss.item():.4f}")

    dist.destroy_process_group()