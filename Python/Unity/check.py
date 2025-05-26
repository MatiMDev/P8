import torch
print(torch.cuda.is_available())  # Should return True
print(torch.__version__)
print(torch.version.cuda)  # Should show something like '11.8'
