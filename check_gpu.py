import torch

print("PyTorch Version:", torch.__version__)

if torch.cuda.is_available():
    print("✅ NVIDIA GPU is enabled!")
    print("Device Name:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print("✅ Apple M-Series GPU is enabled!")
else:
    print("❌ Still using CPU.")