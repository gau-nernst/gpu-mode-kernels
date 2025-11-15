#%%
import os
import pickle
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
import tg4perfetto
import sys
#%%
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
import argparse
import csv
import json
from collections import namedtuple
from enum import Enum
from typing import List

import torch
from tg4perfetto import TraceGenerator
#%%
file_name = "time_tensor_rank5_4.pkl"
if 'pykernel' not in sys.argv[0]:
    file_name = sys.argv[1]
with open(file_name, "rb") as f:
    data = pickle.load(f)
#%%
profiler_buffer_host = data.cpu().numpy().tolist()
num_blocks = profiler_buffer_host[0]
print(data[:100])
tgen = TraceGenerator(f"{file_name}.trace")
pid_global = tgen.create_group(f"all_blocks")
xcd_global=[]
for i in range(8):
    xcd_global.append(tgen.create_group(f"xcd_{i}"))
for i in range(num_blocks):
    global_track = pid_global.create_track(f'{i:04d}')
    xcd_track = xcd_global[i % 8].create_track(f'{i:04d}')
    cnt = 0
    pre_timestamp = 0
    for j in range(i + 1, len(profiler_buffer_host), num_blocks):
        timestamp = profiler_buffer_host[j]
        if not timestamp or timestamp < pre_timestamp:
            break
        else:
            print(j, timestamp)
            if j > 1: 
                global_track.close(timestamp-1)
                xcd_track.close(timestamp-1)
            s = "abcdefghlighmnopqrstuvwxyz"
            global_track.open(timestamp, f"{s[cnt % len(s)]}")
            xcd_track.open(timestamp, f"{s[cnt % len(s)]}")
            pre_timestamp = timestamp
            cnt += 1
    global_track.close(pre_timestamp + 1)
    xcd_track.close(pre_timestamp + 1)
        
tgen.flush()
1/0
#%%

class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    block_group_tag = tag >> 12
    event_idx = (tag >> 2) & 0x3FF
    event_type = tag & 0x3
    return (
        block_group_tag // num_groups,
        block_group_tag % num_groups,
        event_idx,
        event_type,
    )


#%%
profiler_buffer_host = data.cpu().numpy().tolist()
num_blocks = profiler_buffer_host[0]
num_blocks
#%%

#%%

def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    event_names: List[str],
    file_name: str,
) -> None:

    profiler_buffer_host = profiler_buffer
    num_blocks, num_groups = profiler_buffer_host[:1]
    num_blocks = int(num_blocks)
    num_groups = 1

    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    print(f"num_blocks: {num_blocks}, num_groups: {num_groups}")
    pid_global = tgen.create_group(f"all_blocks")
    global_track_map = {}
    
    xcd_global=[]
    xcd_global_map = {}
    for i in range(8):
        xcd_global.append(tgen.create_group(f"xcd_{i}"))


    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx:04d}")
        # for group_idx in range(num_groups):
        #     tid = pid.create_group(f"group_{group_idx:04d}")
        tid_map[block_idx] = pid

    for i in range(1, min(len(profiler_buffer_host), 1000000)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )
        # continue
        event = event_names[event_idx]
        tid = tid_map[block_idx]

        if (block_idx, event_idx) in track_map:
            track = track_map[(block_idx, event_idx)]
        else:
            track = tid.create_track('time')
            track_map[(block_idx, event_idx)] = track
        if block_idx in global_track_map:
            global_track = global_track_map[block_idx]
        else:
            global_track = pid_global.create_track(f'{block_idx:04d}')
            global_track_map[block_idx] = global_track
            

        if block_idx in xcd_global_map:
            global_xcd_track = xcd_global_map[block_idx]
        else:
            if block_idx < 304:
                global_xcd_track = xcd_global[block_idx % 8].create_track(f'{block_idx:04d}')
            else:
                global_xcd_track = xcd_global[block_idx % 8].create_track(f'{block_idx:04d}_>304_may_not_accurate') 
            xcd_global_map[block_idx] = global_xcd_track

        assert event_type >= 0 and event_type <= 1, f"event_type: {event_type}"
        if event_type == EventType.kBegin.value:
            # print("start", timestamp, event)
            track.open(timestamp, event)
            global_track.open(timestamp, event)
            global_xcd_track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            # print("end", timestamp, event)
            track.close(timestamp)
            global_track.close(timestamp)
            global_xcd_track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            # print("instant", timestamp, event)
            track.instant(timestamp, event)
            global_track.instant(timestamp, event)
            global_xcd_track.instant(timestamp, event)

    tgen.flush()
names = [
    "START",
    "MAIN",
    "MMA",
    "WB", 
]
for i in range(10):
    print("start export", i)
    export_to_perfetto_trace(torch.tensor(data[i]), names, f"{sys.argv[1]}_profiler{i}.trace")

#%%
# import pickle
# import torch
# with open("profiler.pkl", "rb") as f:
#     data = pickle.load(f)
# for i in range(1, 1000000, 1024):
#     if data[1][i + 1] != 0:
#         print(i, data[1][i + 1] - data[1][i + 1 - 1024], '\t', f"{data[1][i + 1]:x}")
    
# #%
# data[1][1], data[1][1025]
