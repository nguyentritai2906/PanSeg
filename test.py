import torch
from torchinfo import summary
from vargnet import FPSNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FPSNet(3, 20)
model.to(device)
input = torch.rand((1, 3, 512, 512))
output = model(input)
