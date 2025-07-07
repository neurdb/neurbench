import torch
print(torch.__version__)  # 应该输出 1.13.1+cu116
print(torch.cuda.is_available())  # 应该返回 True
print(torch.version.cuda)  # 应该返回 11.6
