import subprocess
import time
import os

# Get current timestamp and commit hash

timestamp = time.strftime("%m%d_%H%M%S")
try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short=6', 'HEAD'], 
                                        stderr=subprocess.DEVNULL).decode().strip()
except:
    commit_hash = "unknown"
filename = f"__{commit_hash}.py"

with open(filename, "w") as f_write:
    f_write.write("import os\n")
    # f_write.write("os.environ['FAST_RUN'] = '10'\n")
    with open("ref10_first.py", "r") as f_read:
        lines = f_read.readlines()
    f_write.write(f"# {timestamp} gen by commit: {commit_hash}\n")
    for line in lines:
        if line.startswith("CPP_BARRIER"):
            with open("ref10_first.cpp", "r") as f_cpp:
                f_write.write('CPP_BARRIER = r"""' + f_cpp.read() + '"""' + "\n")
        elif line.startswith("CUDA_BARRIER"):
            with open("ref10_first.hip", "r") as f_cu:
                f_write.write('CUDA_BARRIER = r"""' + f_cu.read() + '"""' + "\n")
        elif line.startswith("CUDA_MAIN_SRC"):
            with open("make_share.hip", "r") as f_main:
                f_write.write('CUDA_MAIN_SRC = r"""' + f_main.read() + '"""' + "\n")
        elif line.startswith("CREATE_SHEMEM_CODE"):
            with open("create_shemem.py", "r") as f_create:
                f_write.write('CREATE_SHEMEM_CODE = r"""' + f_create.read() + '"""' + "\n")
        else:
            f_write.write(line)


print()
print()
print(f"rm -rf __*.py && scp amddocker2:/root/a2a/ref10/{filename} .")
print()
print()
print(f"rm -rf __*.py && scp spondocker:/root/a2a/ref10/{filename} .")
os.system(f"cp {filename} /root/amd-competition-operator/problems/amd_distributed/gemm-rs/submission.py")