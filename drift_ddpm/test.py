import torch
print(torch.__version__)  # Should output 1.13.1+cu116
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Should return 11.6
