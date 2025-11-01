BM, BN = 256, 128
M, N = 8192, 29696//8


def get_xcd_id(pid, my_rank):
    num_pid_m = M // BM
    num_pid_n = N // BN
    if pid < (num_pid_m * num_pid_n) // 8: 
        result = (pid // num_pid_n) + my_rank * (num_pid_m // 8), pid % num_pid_n
    else:
        pid_left = pid - (num_pid_m * num_pid_n) // 8
        dest_rank = pid_left % 7
        if dest_rank >= my_rank:
            dest_rank += 1
        dest_pid = pid_left // 7
        # print(f"pid_left: {pid_left}, dest_rank: {dest_rank}, dest_pid: {dest_pid}")
        result = dest_rank * (num_pid_m // 8) + (dest_pid // num_pid_n), dest_pid % num_pid_n
    # print(f"pid: {pid}, my_rank: {my_rank}, result: {result}")
    assert result[0] < M // BN, f"result[0] = {result[0]} is out of range"
    assert result[1] < N // BN, f"result[1] = {result[1]} is out of range"
    return result
def remap_xcd(pid, GRID_MN, NUM_XCDS = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    return pid

def remap_xcd_simple(pid, GRID_MN, NUM_XCDS = 8):
    pids_per_xcd = GRID_MN // NUM_XCDS
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # pid = GRID_MN + (xcd - NUM_XCDS) * (pids_per_xcd - 1) + local_pid 
    pid = xcd * pids_per_xcd + local_pid
    return pid


def compute_pid(pid, 
                grid_m,
                grid_n,
                GROUP_M,
                my_rank,
                REMAP_XCD=True):
    GROUP_M = 1 
    if pid < (grid_m * grid_n // 8): 
        if REMAP_XCD:
            pid = remap_xcd(pid, grid_m // 8 * grid_n)

        if GROUP_M == 1:
            pid_m = pid // grid_n
            pid_n = pid % grid_n
        else:
            width = GROUP_M * grid_n
            group_id = pid // width
            group_size = min(grid_m//8 - group_id * GROUP_M, GROUP_M)
            pid_m = group_id * GROUP_M + (pid % group_size)
            pid_n = (pid % width) // (group_size)
        pid_m = (pid_m + grid_m // 8 * my_rank) % grid_m
        return pid_m, pid_n
    else:
        pid -= (grid_m * grid_n) // 8
        if REMAP_XCD:
            which_xcd = pid % 8
            xcd_local_index = pid // 8
            local_xcd_row, local_xcd_col = xcd_local_index // grid_n, xcd_local_index % grid_n
            
            id = local_xcd_row * 8 + which_xcd
            which_group = id % 7
            group_pos = id // 7
            # if pid == 367:
            #     print(f"{pid=} {local_xcd_row=} {local_xcd_col=} {which_xcd=} {xcd_local_index=} {id=} {which_group=} {group_pos=}")
            # if pid == 419:
            #     print(f"{pid=} {local_xcd_row=} {local_xcd_col=} {which_xcd=} {xcd_local_index=} {id=} {which_group=} {group_pos=}")
            # if group_pos == grid_m//8:
            #     which_group += 3
            #     group_pos -=1
            #     local_xcd_col += grid_n // 2
            # if pid >= 416:
            #     local_xcd_col += grid_n // 2
            # assert group_pos < grid_m//8, f"which_group = {which_group} is out of range {id} {group_pos}"
            final_pos_row = which_group * (grid_m//8) + group_pos 
            # print(id, which_group, group_pos, final_pos_row)
            pid_m = final_pos_row
            pid_n = local_xcd_col
        pid_m = (pid_m + (grid_m // 8) * (my_rank + 1)) % grid_m
    # print(pid, pid_m, pid_n)
    return pid_m, pid_n

# Create a rectangular grid (M//BM, N//BN) and fill it with get_xcd_id results
# grid = []
online_config = {
(64, 2304, 7168): {}, ########## tmp not doing M=64 calculation
(512, 1536, 4096): {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 4, 'num_ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 200}, # try test it's time.
(2048, 360, 2880): {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num8192*2ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 500},
(4096, 512, 4096): {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 500}, # may try send 6 ?
(8192, 1792, 4096): {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 500},
(8192, 3696, 8192): {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2, "SLEEP_CYCLE": 2000}, #'NUM_KSPLIT': 1,  'SPLITK_BLOCK_SIZE': 8192},
}

which_one = 2
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type x: Block
    :param div: the divisor
    :type div: Block
    """
    return (x + div - 1) // div
num_pid_m, num_pid_n = [
    (512//64, cdiv(1536, 64)),
    (2048//64, cdiv(360, 32)),
    (4096//64, cdiv(512, 64)),
    (8192//256, cdiv(1792, 128)),
    (8192//256, cdiv(3696, 256)),
][which_one]
# num_pid_m, num_pid_n = 8192//256, 3696//256
total_pids = num_pid_m * num_pid_n

# # Initialize the grid
# for i in range(num_pid_m):
#     row = []
#     for j in range(num_pid_n):
#         row.append(None)
#     grid.append(row)

# Fill the grid with get_xcd_id results for each rank
for my_rank in range(8): # only frist now...  # Assuming 8 ranks based on the context
    grid = []
    for i in range(num_pid_m):
        row = []
        for j in range(num_pid_n):
            row.append(None)
        grid.append(row)
    print(f"\nRank {my_rank}: {num_pid_m=} {num_pid_n=}")
    for pid in range(total_pids):
        # 1. origin method
        # m_idx, n_idx = get_xcd_id(pid, my_rank) 
        # 2. now method
        m_idx, n_idx = compute_pid(pid, num_pid_m, num_pid_n, 4, my_rank)
        assert m_idx < num_pid_m, f"m_idx = {m_idx} is out of range"
        assert n_idx < num_pid_n, f"n_idx = {n_idx} is out of range"
        # print(f"{pid=}, {m_idx} {n_idx}")
        grid_m = m_idx
        grid_n = n_idx
        if grid[grid_m][grid_n] is None:
            grid[grid_m][grid_n] = []
        grid[grid_m][grid_n].append((my_rank, pid))
    
    # Print the grid for this rank
    for i in range(num_pid_m):
        if (i % (num_pid_m // 8)) == 0: 
            print("-----------------------------------"*4)
        print(f"line {i}:", end = " ")
        for j in range(num_pid_n):
            if grid[i][j] and any(rank == my_rank for rank, _ in grid[i][j]):
                pids = [pid for rank, pid in grid[i][j] if rank == my_rank]
                # assert len(pids) == 1, f"len(pids) = {len(pids)} is not 1, {pids}"
                for idx, ppp in enumerate(pids):
                    end = "  " if idx != len(pids) - 1 else ", "
                    print(f"{ppp:4d}:{ppp%8}", end="  ")
                print(end = ", ")
                # print(f"({i},{j}): {pids}", end="  ")
            else:
                # assert False, f"{i=} {j=}"
                print(f"({i},{j}): []", end="  ")
        print()
    
    # Clear grid for next rank
    for i in range(num_pid_m):
        for j in range(num_pid_n):
            grid[i][j] = None


