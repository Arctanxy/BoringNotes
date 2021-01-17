import time
import torch
from torch import nn
import torch.nn.functional as F

FAKE_QUANT = False
REAL_QUANT = False

# 训练时用fake quant
def fake_quant(x, qbit = 8, clamp_val = None):
    if clamp_val is None:
        clamp_val = torch.max(x)
    scale = clamp_val / (2**(qbit - 1))
    q_value = (x / scale).floor().clamp(-2**(qbit - 1), 2**(qbit - 1) - 1)
    data = q_value * scale
    return data

# 推理时用real quant处理过的参数
def real_quant(x, qbit = 8, clamp_val = None):
    if clamp_val is None:
        clamp_val = torch.max(x)
    scale = clamp_val / (2**(qbit - 1))
    q_value = (x / scale).floor().clamp(-2**(qbit - 1), 2**(qbit - 1) - 1)
    # return q_value.short() # 按理说int8 应该返回char， int16应该返回short
    return q_value.int()

class QuantLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(QuantLinear, self).__init__()
        self.weight = nn.Parameter(torch.rand(in_channels, out_channels))
        self.bias = nn.Parameter(torch.rand(out_channels))

    def forward(self,x):
        if FAKE_QUANT and not REAL_QUANT:
            weight = fake_quant(self.weight, 8, 2)
            bias = fake_quant(self.bias, 16, 16)
            x = fake_quant(x, 8, 4)
            a = time.time()
            out = F.linear(x, weight.t(), bias)
            print("fake quant cost time : {}".format(time.time() - a))
            return out
        elif REAL_QUANT:
            weight = real_quant(self.weight, 8, 2)
            bias = real_quant(self.bias, 16, 16) # scale_bias = scale_weight * scale_x
            x = real_quant(x, 8, 4)
            # return F.linear(x, weight.t(), bias) / 2048  # 2048 为bias的scale的倒数
            a = time.time()
            # out = F.linear(x, weight.t(), bias) >> 11
            out = x.mm(weight) + bias # admm_cuda not implemented for "Int"
            print("real quant cost time : {}".format(time.time() - a))
            return out
        else:
            print(x.device, self.weight.device, self.bias.device)
            return F.linear(x, self.weight.t(), self.bias)

if __name__ == "__main__":
    fc = QuantLinear(500,800).cuda()
    x = torch.rand(400, 500).cuda()
    out1 = fc(x)
    FAKE_QUANT = True
    out2 = fc(x)
    REAL_QUANT = True
    out3 = fc(x)
