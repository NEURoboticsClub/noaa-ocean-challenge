import torch
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
print(DEVICE)
print(torch.zeros(1).cuda())