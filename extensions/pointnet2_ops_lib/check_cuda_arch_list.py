import torch

if torch.cuda.is_available():
    arch_list = torch.cuda.get_arch_list()
    print(f"CUDA architectures compiled for this PyTorch build: {arch_list}")
else:
    print("CUDA is not available on this system.")