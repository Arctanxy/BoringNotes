from torch import nn
from layers.acc import psnr, jacarrd
from network.attention_sr import *
from network.ocr import BidirectionalLSTM

class CRNN(nn.Module):
    def __init__(self, args):
        super(CRNN, self).__init__()
        self.args = args
        # down sample for crnn
        self.maxpool = nn.MaxPool2d((2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        SCPA_block_f = functools.partial(SCPA, nf=self.args.nf, reduction=2)
        self.conv = nn.Sequential(
            nn.Conv2d(self.args.in_channels, self.args.nf, 3, 1, 1, bias=True),
            self.maxpool,
            make_layer(SCPA_block_f, 2),
            self.maxpool,
            make_layer(SCPA_block_f, 2),
            self.maxpool,
            make_layer(SCPA_block_f, 2),
            self.avgpool
        )
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(self.args.nf, self.args.nh, self.args.nh),
        #     BidirectionalLSTM(self.args.nh, self.args.nh, self.args.num_classes)
        # )
        self.rnn = nn.Linear(self.args.nf, self.args.num_classes)

    def forward(self, x):
        cnn = self.conv(x)
        cnn = cnn.squeeze(2)
        cnn = cnn.permute(2, 0, 1)
        # rlt = self.rnn(cnn)
        rlt = self.rnn(cnn.contiguous().view(cnn.shape[0] * cnn.shape[1], cnn.shape[2]))
        rlt = rlt.view(cnn.shape[0], cnn.shape[1], -1)
        # print(rlt.shape)
        return rlt


class PAN(nn.Module):
    def __init__(self, args):
        super(PAN, self).__init__()
        self.args = args
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=self.args.nf, reduction=2)
        self.scale = self.args.scale

        ### first convolution
        self.conv_first = nn.Conv2d(self.args.in_channels, self.args.nf, 3, 1, 1, bias=True)

        ### main blocks
        self.SCPA_trunk = make_layer(SCPA_block_f, self.args.nb)
        self.trunk_conv = nn.Conv2d(self.args.nf, self.args.nf, 3, 1, 1, bias=True)

        #### upsampling
        self.upconv1 = nn.Conv2d(self.args.nf, self.args.unf, 3, 1, 1, bias=True)
        self.att1 = PA(self.args.unf)
        self.HRconv1 = nn.Conv2d(self.args.unf, self.args.unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(self.args.unf, self.args.unf, 3, 1, 1, bias=True)
            self.att2 = PA(self.args.unf)
            self.HRconv2 = nn.Conv2d(self.args.unf, self.args.unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(self.args.unf, self.args.out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk
        # print("feature shape ", fea.shape)

        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)

        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out

class PACRNN(nn.Module):
    def __init__(self, args):
        super(PACRNN, self).__init__()
        self.args = args
        self.sr = PAN(self.args)
        self.rec = CRNN(self.args)

    def forward(self, x):
        img = self.sr(x)
        rlt = self.rec(img)
        return img, rlt


class SROCR(nn.Module):
    def __init__(self, args):
        super(SROCR, self).__init__()
        self.args = args
        self.model = PACRNN(self.args)
        self.loss_sr = nn.L1Loss()
        self.loss_ocr = nn.CTCLoss(blank=self.args.num_classes - 1, zero_infinity=True)
        self.acc_sr = psnr
        self.acc_ocr = jacarrd

    def forward(self, meta):
        x = meta["lr"]
        # print("x shape ", x.shape)
        img, rlt = self.model(x)
        hr = meta["hr"]
        label = meta["label"]
        # input_lengths = meta["input_lengths"]
        input_lengths = torch.full(size = (rlt.shape[1],), fill_value=rlt.shape[0],dtype=torch.long)
        target_lengths = meta["target_lengths"]
        if self.training:
            loss1 = self.loss_sr(img, hr)
            loss2 = self.loss_ocr(rlt, label, input_lengths, target_lengths)
            loss = loss1 + loss2
            return loss, loss1, loss2, loss2, loss2
        else:
            acc1 = self.acc_sr(img, hr)
            rlt = torch.argmax(rlt, dim = -1).permute(1,0)
            rlt = self.convert(rlt, self.args.num_classes)
            acc2 = self.acc_ocr(rlt, label)  # label 是拼接起来的，一个batch算一次acc
            return acc1, acc2, acc2, acc2, acc2

    def convert(self, out, blank):
        rlts = []
        for n in range(len(out)):
            rlt = []
            for i in range(len(out[n])):
                if out[n][i] != blank and (not (i > 0 and out[n][i - 1] == out[n][i])):
                    rlt.append(out[n][i].item())
            rlts.extend(rlt)
        return rlts

class OCR(nn.Module):
    def __init__(self, args):
        super(OCR, self).__init__()
        self.args = args
        self.model = CRNN(self.args)
        self.loss_ocr = nn.CTCLoss(blank=self.args.num_classes - 1, reduction='mean', zero_infinity=True)
        self.acc_ocr = jacarrd

    def forward(self, meta):
        x = meta["lr"]
        from torchvision import transforms
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        rlt = self.model(x)
        hr = meta["hr"]
        label = meta["label"]
        # input_lengths = meta["input_lengths"]
        input_lengths = torch.full(size = (rlt.shape[1],), fill_value=rlt.shape[0],dtype=torch.long)
        target_lengths = meta["target_lengths"]
        if self.training:
            rlt = torch.nn.functional.log_softmax(rlt, dim=-1)
            loss2 = self.loss_ocr(rlt, label, input_lengths, target_lengths)
            # print("### \n", torch.argmax(rlt,dim = -1).flatten(), "\n", label)
            return loss2, loss2, loss2, loss2, loss2
        else:
            rlt = torch.argmax(rlt, dim = -1).permute(1,0)
            rlt = self.convert(rlt, self.args.num_classes-1)
            acc2 = self.acc_ocr(rlt, label)  # label 是拼接起来的，一个batch算一次acc
            return acc2, acc2, acc2, acc2, acc2

    def convert(self, out, blank):
        rlts = []
        for n in range(len(out)):
            rlt = []
            for i in range(len(out[n])):
                if out[n][i] != blank and (not (i > 0 and out[n][i - 1] == out[n][i])):
                    rlt.append(out[n][i].item())
            rlts.extend(rlt)
        return rlts


if __name__ == "__main__":
    import yaml
    from module.config import Config
    # yml_file = sys.argv[1]
    yml_file = "../config.yaml"
    f = open(yml_file)
    params = yaml.load(f, Loader=yaml.SafeLoader)
    args = Config(params)
    net = PACRNN(args)
    input = torch.ones((2, 3, 32, 224))
    out, rlt = net(input)
    print(out.shape, rlt.shape)



# todo: 拆解一下，模块化