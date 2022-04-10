import torch
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tools.image_generator import ImageGenerator


def create_input(batch):
    blur_imgs = [item[0] for item in batch]
    imgs = [item[1] for item in batch]
    max_width = max(item.shape[2] for item in blur_imgs)
    max_width_hr = max(item.shape[2] for item in imgs)
    num = len(batch)
    padded_blur_img = torch.ones((num, 1, 16, max_width))
    for i, item in enumerate(blur_imgs):
        padded_blur_img[i][:item.shape[0], :item.shape[1], :item.shape[2]] = item
    labels = [item[2] for item in batch]
    lengths = torch.tensor([len(item) for item in labels])
    labels = torch.concat(labels)
    padded_img = torch.ones((num, 1, 64, max_width_hr))
    for i, item in enumerate(imgs):
        padded_img[i][:item.shape[0], :item.shape[1], :item.shape[2]] = item
    mask = padded_img != 0
    data = {}
    data["lr"] = padded_blur_img
    data["hr"] = padded_img
    data["label"] = labels
    data["target_lengths"] = lengths
    data["mask"] = mask
    return data

class TextData(Dataset):
    def __init__(self, args, data_type) -> None:
        super().__init__()
        self.word2index = {}
        with open(args.vocab_list, 'r', encoding="utf-8") as f:
            vocab = f.read().replace("\n", "")
            for i, item in enumerate(vocab):
                self.word2index[item] = i
        self.word2index[" "] = len(self.word2index)
        self.generator = ImageGenerator(args)
        self.args = args
        self.data_type = data_type

    def __getitem__(self, index):
        blur_img, img, label = self.generator.run()
        # print(img.size, label)
        blur_img = transforms.ToTensor()(blur_img)
        img = transforms.ToTensor()(img)
        label = torch.tensor([self.word2index[item] for item in label if item in self.word2index])
        return blur_img, img, label

    def __len__(self):
        if self.data_type == "train":
            return self.args.train_length
        else:
            return self.args.val_length

def load_train_data(args):
    train_data = TextData(args, "train")
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, collate_fn=create_input,num_workers=0)
    return train_loader

def load_eval_data(args):
    eval_data = TextData(args, "eval")
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size * 2, shuffle=True, collate_fn=create_input, num_workers=8)
    return eval_loader




