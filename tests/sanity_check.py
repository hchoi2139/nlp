import torch
from transformers import AutoTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("WARNING: PyTorch cannot see the GPU!")

try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    print("DeBERTa-v3 Tokenizer loaded successfully!")
except Exception as e:
    print(f"Tokenizer Error: {e}")