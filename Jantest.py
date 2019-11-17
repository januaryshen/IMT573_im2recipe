import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchwordemb
from args import get_parser


parser = get_parser()
opts = parser.parse_args()

irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)
_, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
print(vec)
