import modal

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("ninja", "nvidia-cutlass-dsl")
    .add_local_python_source("cutedsl_test")
)

app = modal.App("sm100-nvfp4", image=image)


@app.function(gpu="B200")
def f():
    import torch
    from cutedsl_test import Kernel, custom_kernel

    M, N, K = 256, 4096, 7168
    A = torch.randint(255, size=(M, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    B = torch.randint(255, size=(N, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    SFA = torch.randn(M, K // 16, device="cuda").to(torch.float8_e4m3fn)
    SFB = torch.randn(N, K // 16, device="cuda").to(torch.float8_e4m3fn)

    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    custom_kernel(A, B, SFA, SFB, C, Kernel.Config((128, 128), (1, 1)))

    C_ref = torch._scaled_mm(A, B.T, SFA, SFB, out_dtype=torch.float16)
    torch.testing.assert_close(C, C_ref)


@app.local_entrypoint()
def main():
    f.remote()
