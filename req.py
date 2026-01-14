import sys
import torch
import numpy

print("="*30)
print("     Environment Check")
print("="*30)
print(f"Python Version:    {sys.version.split()[0]}")
print(f"PyTorch Version:   {torch.__version__}")
print(f"PyTorch CUDA:      {torch.version.cuda}")
print(f"NumPy Version:     {numpy.__version__}")

# 检查 CUDA 是否真的可用
if torch.cuda.is_available():
    print(f"CUDA Available:    Yes")
    print(f"GPU Name:          {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA Available:    NO (Reviewers will fail to reproduce!)")
print("="*30)