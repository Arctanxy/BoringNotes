import torch

def psnr(tensor1, tensor2):
    mse = ((tensor1 - tensor2) ** 2).mean()
    if mse == 0:
        return torch.inf
    return 10 * torch.log10(1*1 / torch.sqrt(mse))

def jacarrd(string1, string2):
    if isinstance(string2, torch.Tensor):
        string2 = string2.cpu().data.numpy().tolist()
    union_length = len(set(string1 + string2))
    assert union_length != 0
    intersection_length = len(set(string1)) + len(set(string2)) - union_length
    return intersection_length / union_length