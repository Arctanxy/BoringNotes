import os 
import random
import numpy as np 
from glob import glob 
from copy import deepcopy
from PIL import Image, ImageFont, ImageDraw, ImageFilter

class ImageGenerator:
    def __init__(self, args ):
        self.fonts = sorted(
            glob(os.path.join(args.font_folder, "*/*.ttf"))
            + glob(os.path.join(args.font_folder, "*/*.TTF"))
            + glob(os.path.join(args.font_folder, "*/*.ttc"))
            + glob(os.path.join(args.font_folder, "*/*.TTC"))
            + glob(os.path.join(args.font_folder, "*/*.otf"))
        )
        text_files = glob(os.path.join(args.text_folder, "*txt"))
        self.texts = ""
        for file in text_files:
            try:
                self.texts += open(file, 'r', encoding='utf-8').read().replace("\n", "").replace("\t", "")
            except:
                self.texts += open(file, 'r', encoding='gbk').read().replace("\n", "").replace("\t", "")
        
        self.bg = [Image.open(path).convert("RGB").resize((1000,1000)) for path in glob(os.path.join(args.bg_folder, "*"))]
        self.rng = np.random.default_rng(args.seed)

    def generate_one(self, word, font_file, font_size, img):
        font = ImageFont.truetype(font_file, font_size)
        # img = Image.new(mode = "L", size = (width, 32), color=255)
        draw = ImageDraw.Draw(img)
        start_point = (self.rng.integers(0, font_size), 0)
        draw.text(start_point, word, fill=0, font=font)
        return img

    def blur(self, img):
        img = deepcopy(img)
        if self.rng.random() > 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=3))
        if self.rng.random() > 0.3:
            img = img.filter(ImageFilter.MinFilter(3))
        if self.rng.random() > 0.3:
            img = img.filter(ImageFilter.BLUR())
        return img

    def run(self):
        st = self.rng.integers(0, len(self.texts))
        length = self.rng.integers(2, 8)
        font = self.rng.choice(self.fonts)
        font_size = self.rng.choice([20,25,30,35,40,45])
        bg = self.bg[self.rng.choice(list(range(len(self.bg))))]
        word = self.texts[st:st+length]
        height = font_size + 5
        width = len(word) * height + font_size
        left = self.rng.integers(0, bg.size[0] - width)
        up = self.rng.integers(0, bg.size[1] - height)
        right = left + width
        down = up + height
        img = bg.crop((left, up, right, down))
        img = self.generate_one(word, font, font_size, img).convert("L")
        blur_img = self.blur(img).convert("L")
        img = self.resize(img, 64)
        blur_img = self.resize(blur_img, 16)
        # print("img shape {} blur img shape {} ".format(img.size, blur_img.size))
        return blur_img, img, word


    def resize(self, img, height):
        w, h = img.size
        ratio = h / height
        w_resize = int(w / ratio)
        if height == 64:
            w_resize = w_resize // 4 * 4
        img = img.resize((w_resize, height))
        return img
