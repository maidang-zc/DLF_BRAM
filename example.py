import torch
import torch.nn as nn
from models.DLF_BRAM import DLF_BRAM_NET

model = DLF_BRAM_NET(num_classes=2)
inputs = torch.randn((2,3,8,5,224,224))
output1, output2, output3 = model(inputs)
print(output1.shape)
