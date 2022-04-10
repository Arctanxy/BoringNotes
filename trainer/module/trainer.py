import os
import torch
import importlib
from module.dataloader import load_train_data, load_eval_data
from module.logger import Logger

def get_module(string):
    attrs = string.split(".")
    module_name = attrs[0]
    for attr in attrs[1:-1]:
        module_name += ("." + attr.strip())
    attr_name = attrs[-1].strip()
    m = importlib.import_module(module_name)
    module = getattr(m, attr_name)
    return module

class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.model = get_module(args.model_name)(self.args).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.train_loader = load_train_data(self.args)
        self.val_loader = load_eval_data(self.args)
        self.logger = Logger(self.args)

    def train(self):
        for i in range(self.args.epochs):
            self.train_one_epoch(i)

    def train_one_epoch(self, epoch):
        for i, data in enumerate(self.train_loader):
            for k, v in data.items():
                data[k] = v.cuda()
            self.optimizer.zero_grad()
            loss1, loss2, loss3, loss4, loss5 = self.model(data)
            loss1.backward()
            self.optimizer.step()
            if (i + 1) % self.args.display_freq == 0:
                self.logger.info("Train Loss1 is {} Loss2 is {} ".format(loss1.item(), loss2.item()))
            if (i + 1) % self.args.val_freq == 0:
                self.validation()
        self.scheduler.step()
        torch.save(self.model.state_dict(), os.path.join(self.args.ckpt_folder, "epoch_{}.pth".format(epoch)))
    
    def validation(self):
        self.model.eval()
        acc1s = []
        acc2s = []
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                for k, v in data.items():
                    data[k] = v.cuda()
                acc1, acc2, acc3, acc4, acc5 = self.model(data)
                if isinstance(acc1, torch.Tensor):
                    acc1s.append(acc1.item())
                else:
                    acc1s.append(acc1)
                acc2s.append(acc2)
        self.logger.info("Accuracy sr is {}".format(sum(acc1s) / len(acc1s)))
        self.logger.info("Accuracy sr is {}".format(sum(acc2s) / len(acc2s)))
        self.model.train()
            








