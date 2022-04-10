import os
import random
import numpy as np 
import pandas as pd
import os.path as osp
from glob import glob
from keys import alphabetChinese
from torchvision import transforms
from PIL import ImageFont, ImageDraw, Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tools.image_generator import ImageGenerator

def create_input(batch):
    batch = [item.permute(0,2,1).squeeze(0) for item in batch]
    imgs = pad_sequence(batch, batch_first=True, padding_value=1.0)
    imgs = imgs.permute(0,2,1).unsqueeze(1)
    return imgs

class CHNData(Dataset):
    def __init__(self, args, font_size = 60, transform = None, subset = "train"):
        self.args = args
        if subset == "train":
            # font_folder = osp.join(self.args.font_folder, "fonts", "train_fonts")
            font_folder = self.args.font_folder
        elif subset == "eval":
            # font_folder = osp.join(self.args.font_folder, "fonts", "eval_fonts")
            font_folder = self.args.font_folder
        else:
            print("Could not find subset")

        self.generator = ImageGenerator(font_folder=args.font_folder)
        self.font_num = len(self.fonts)
        self.font_size = font_size
        self.characters = alphabetChinese
        # characters = pd.read_excel("3500常用汉字.xls")
        # self.characters = list(characters["hz"].values)
        # self.characters = self.characters[: self.args.num_chars]  # 调试模型的时候只用一部分
        # if self.args.num_chars < 3500:
        #     print("使用文字:{}".format(self.characters))

        if transform is None:
            self.transform = transforms.Compose(
                [transforms.ToTensor()]
            )
        else:
            self.transform = transform

    def __getitem__(self, index):
        chars_len = random.randint(5, 10) # 5 to 9 chars per image
        chars = "".join(random.choices(self.characters, k=chars_len))
        img = self.generator.run()
        img = self.transform(img)
        return img

    def __len__(self):
        return self.font_num * len(self.characters)

class Zoom(object):
    def __call__(self, image):
        image_= image.resize((image.size[0] // 5, image.size[1]// 5))
        image_ = image_.resize((image.size[0], image.size[1]))
        return image_


def load_generate_data(args):
    transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=3), 
        Zoom()
    ])
    train_data = CHNData(args, 60, transform, "train")
    return train_data
    

def load_train_data(args):
    transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=3), 
        Zoom(),
        transforms.ToTensor()
    ])
    train_data = CHNData(args, 60, transform, "train")
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, collate_fn=create_input,num_workers=8)
    return train_loader

def load_eval_data(args):
    transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=3), 
        Zoom(),
        transforms.ToTensor()
    ])
    eval_data = CHNData(args, 60, transform, "eval")
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size * 2, shuffle=True, collate_fn=create_input, num_workers=8)
    return eval_loader

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Args for BlurOCR")

    parser.add_argument("--image-height", default=80, type=int)
    parser.add_argument("--font-folder", default="./fonts", type = str)
    parser.add_argument("--data-dir", default="D:\\datasets\\blurocr\\src_data",type=str)
    parser.add_argument("--blur-dir", default="D:\\datasets\\blurocr\\blur_data", type=str)
    parser.add_argument("--cut-ratio", default=1, type=float)
    parser.add_argument("--crnn-model", default="./ckpts/ocr-lstm.pth", type = str)
    parser.add_argument("--drrn-model", default="./model_0_1999", type=str)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--train-from-scratch", default=0, type=int)
    parser.add_argument("--ckpt-freq", default=2000, type=int)
    parser.add_argument("--display-freq", default=100, type=int)
    parser.add_argument("--cv-freq", default=500, type=int)

    args = parser.parse_args()

    # train_loader = load_train_data(args)

    # for i in tqdm(train_loader, total=len(train_loader)):
    #     pass

    gen_data = load_generate_data(args)
    for i in tqdm(gen_data, total=len(gen_data)):
        pass