import enum
import imp
from module.dataloader import load_train_data
if __name__ == "__main__":
    import yaml
    from module.config import Config
    args = Config(yaml.load(open("config.yaml"), Loader=yaml.SafeLoader))
    # train_loader = load_train_data(args)
    # for i, data in enumerate(train_loader):
    #     import pdb;pdb.set_trace()

    from tools.image_generator import ImageGenerator
    from PIL import Image
    ig = ImageGenerator(args)
    while True:
        blur_img, img, word = ig.run()
        image = Image.new(blur_img.mode, (max(blur_img.size[0], img.size[0]), img.size[1] + blur_img.size[1]))
        image.paste(blur_img, (0, 0))
        image.paste(img, (0, blur_img.size[1]))
        image.show()