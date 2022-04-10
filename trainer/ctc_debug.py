import torch

inp = torch.randn((10,1,200))
net = torch.nn.Linear(200, 100)
loss = torch.nn.CTCLoss(blank=99)
target_lengths = torch.Tensor([5]).long()
input_lengths = torch.Tensor([10]).long()
label = torch.Tensor([1,2,3,4,5])
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
for i in range(1000):
    out = torch.nn.functional.log_softmax(net(inp), dim=-1)
    optimizer.zero_grad()
    l = loss(out, label, input_lengths, target_lengths)
    l.backward()
    optimizer.step()
    print(l.item(), torch.argmax(out, dim = -1).flatten())
