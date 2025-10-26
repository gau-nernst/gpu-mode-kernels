# AMD all2all worklog

- `reference.py`: 93540μs
- `submission_v0.py`: mainly re-write the reference implementation to understand things better. Some minor optimizations, such as `.bincount()` and re-use counts for `combine()`. Identify that we must eliminate the loop over tokens. 27548μs.
- `submission_v1.py`: data rearrangement is to group tokens together -> this is a sorting problem. This eliminates the for loop. 1361μs.
- `submission_v2.py`: `combine()` and `dispatch()` are symmetric. The sorting indices used in `dispatch()` act as gather-op, which becomes scatter-op in dispatch -> Don't need to sort again in `combine()`. 1311μs.
- `submission_v3.py`: replace `dist.all_to_all_single()` with custom a2a using symmetric memory. Slower than v2. 1365μs.
- `submission_v4.py`: fused dispatch and combine kernels, following [`pplx-kernels`](pplx-kernels). 116ms.
- `submission_v5.py`: more fine-grained lock for dispatch: per-token flag instead of per `src_rank` flag. 517μs.
- `submission_v6.py`: perform fake-MoE on active tokens only (ignore pad tokens). 464μs. Overlap combine-send and combine-recv using threadblock-specialization. 440μs. Tuned combine kernel params - 379μs.
- `submission_v7.py`: fuse fake-MoE with combine. 421μs. Tuned kernel params - 345μs
- `submission_v7b.py`: reduce various overheads: malloc and zero_ in hot loop, Python->C++ - 303μs

## Submission v3 - P2P

Profile of v2 (unit is us)

|   hidden_dim |   num_experts |   max_num_tokens |   topk |   v2 |   dispatch a2a + fake MoE |
|-------------:|--------------:|-----------------:|-------:|-----:|----------------------------:|
|         6144 |             8 |               16 |      2 | 1059 |                         586 |
|         2048 |            64 |               32 |      6 | 1228 |                         709 |
|         2880 |           128 |              128 |      4 | 1073 |                         700 |
|         4096 |           128 |              256 |      8 | 1054 |                         640 |
|         7168 |           256 |              256 |      8 | 1118 |                         761 |

- Latency doesn't change much across problem size -> there is a huge fixed overhead.
- Doing only dispatch all2all roughly halves the latency -> all2all is the biggest bottleneck, optimizing for local op won't yield much improvements.
- We are handling intra-node 8xMI300X. According to [AMD datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf), they have "fully-meshed 128 GB/s bidirectional Infinity Fabric Connectivity" -> we can send/recv data from each node to every node directly, don't need to bother with fancy algorithms. We should focus on reducing the overhead, which might be introduced by PyTorch.
- [xGMI blogpost](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html) "Each AMD Instinct MI300X system GPU is connected to its seven peer GPUs via xGMI links", "64 GB/s for each point-to-point link", "usable bandwidth to approximately 48 GB/s per link"

P2P
- General flow for multi-process multi-GPU P2P: `cudaSetDevice()` to set current GPU, `cudaDeviceEnablePeerAccess()`, each GPU allocate its own memory in their respective processes, `cudaIpcGetMemHandle()`, all-gather IPC handles, `cudaIpcOpenMemHandle()` -> maps memory from another process to the **current device** address space.
- In the competition eval code, each test case re-uses the same set of processes, but they can get re-assigned different GPUs across tests. Hence, we patch `dist.init_process_group()` and `dist.destroy_process_group()` to do allocation and tear down.
- In v3, we try to replace `dist.all_to_all()` with our own version using symmetric memory.

-> Worse than v2

## Submission v4

Resources:
- https://github.com/deepseek-ai/DeepEP
- https://github.com/perplexityai/pplx-kernels
- https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication
