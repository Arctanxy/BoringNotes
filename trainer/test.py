import imp
import torch
from torchvision import transforms
from module.trainer import get_module, load_eval_data
from module.config import load_args

if __name__ == "__main__":
    args = load_args("config.yaml")
    model = get_module(args.model_name)(args)
    model.load_state_dict(torch.load("out/model/epoch_8.pth"))
    # # test image
    # from PIL import Image
    # img = Image.open(r"F:\scripts\git_repo\BoringNotes\trainer\res\afaecb969241d772153ca3355dc0a66.png").convert("L")
    # img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    # out = model.predict(img_tensor)
    # out_img = transforms.ToPILImage()(out[0])
    # out_img.show()

    # # test val loader
    val_loader = load_eval_data(args)
    for i, data in enumerate(val_loader):
        out = model.predict(data["lr"])
        out_img = [transforms.ToPILImage()(item) for item in out.cpu()]
        in_img = [transforms.ToPILImage()(item) for item in data["lr"].cpu()]
        for j in range(len(out_img)):
            out_img[j].show()




