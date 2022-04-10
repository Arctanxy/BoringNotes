import torch
import functools
from layers.acc import psnr, jacarrd
from torch import nn
import torch.nn.functional as F

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def convert(out, blank):
    rlts = []
    for n in range(len(out)):
        rlt = []
        for i in range(len(out[n])):
            if out[n][i] != blank and (not (i > 0 and out[n][i - 1] == out[n][i])):
                rlt.append(out[n][i].item())
        rlts.extend(rlt)
    return rlts

class PA(nn.Module):
    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(x)
        out = torch.mul(x,y)
        return out

class NOPA(nn.Module):
    def __init__(self, nf):
        super(NOPA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)

    def forward(self, x):
        y = self.conv(x)
        return y

class PAConv(nn.Module):
    def __init__(self, nf, k_size = 3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(x)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out

class NOPAConv(nn.Module):
    def __init__(self, nf, k_size = 3):
        super(NOPAConv, self).__init__()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        out = self.k3(x)
        out = self.k4(out)
        return out

class SCPA(nn.Module):
    def __init__(self, nf, reduction=2, stride = 1, dilation=1):
        super(SCPA, self).__init__()

        group_width = nf // reduction
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=dilation, bias=False)

        self.PAConv = PAConv(group_width)

        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual
        return out

class SCNOPA(nn.Module):
    def __init__(self, nf, reduction=2, stride = 1, dilation=1):
        super(SCNOPA, self).__init__()

        group_width = nf // reduction
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=dilation, bias=False)

        self.PAConv = NOPAConv(group_width)

        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual
        return out


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

class NOPAN(nn.Module):
    def __init__(self, args):
        super(NOPAN, self).__init__()
        self.args = args
        # SCPA
        SCPA_block_f = functools.partial(SCNOPA, nf=self.args.nf, reduction=2)
        self.scale = self.args.scale

        ### first convolution
        self.conv_first = nn.Conv2d(self.args.in_channels, self.args.nf, 3, 1, 1, bias=True)

        ### main blocks
        self.SCPA_trunk = make_layer(SCPA_block_f, self.args.nb)
        self.trunk_conv = nn.Conv2d(self.args.nf, self.args.nf, 3, 1, 1, bias=True)

        #### upsampling
        self.upconv1 = nn.Conv2d(self.args.nf, self.args.unf, 3, 1, 1, bias=True)
        self.att1 = NOPA(self.args.unf)
        self.HRconv1 = nn.Conv2d(self.args.unf, self.args.unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(self.args.unf, self.args.unf, 3, 1, 1, bias=True)
            self.att2 = NOPA(self.args.unf)
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

class CRNNPA(nn.Module):
    def __init__(self, args):
        super(CRNNPA, self).__init__()
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

class CRNNPA_PERCEPTUAL(nn.Module):
    def __init__(self, args):
        super(CRNNPA_PERCEPTUAL, self).__init__()
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
        return rlt, cnn

class CRNNNOPA(nn.Module):
    def __init__(self, args):
        super(CRNNNOPA, self).__init__()
        self.args = args
        # down sample for crnn
        self.maxpool = nn.MaxPool2d((2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        SCPA_block_f = functools.partial(SCNOPA, nf=self.args.nf, reduction=2)
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

class CUBIC_OCRNOPA(nn.Module):
    def __init__(self, args):
        super(CUBIC_OCRNOPA, self).__init__()
        self.args = args
        self.sr = F.interpolate
        self.ocr = CRNNNOPA(self.args)
        self.loss_sr = None
        self.acc_sr = None
        self.loss_ocr = nn.CTCLoss(blank=self.args.num_classes - 1, reduction='mean', zero_infinity=True)
        self.acc_ocr = jacarrd

    def forward(self, data):
        x = data["lr"]
        hr = data["hr"]
        label = data["label"]
        target_lengths = data["target_lengths"]
        x = self.sr(x, scale_factor=2, mode='nearest')
        rlt = self.ocr(x)
        input_lengths = torch.full(size=(rlt.shape[1],), fill_value=rlt.shape[0], dtype=torch.long)
        if self.training:
            rlt = torch.nn.functional.log_softmax(rlt, dim=-1)
            loss = self.loss_ocr(rlt, label, input_lengths, target_lengths)
            return loss, loss, loss, loss, loss
        else:
            rlt = torch.argmax(rlt, dim=-1).permute(1, 0)
            rlt = convert(rlt, self.args.num_classes - 1)
            acc = self.acc_ocr(rlt, label)  # label 是拼接起来的，一个batch算一次acc
            return acc, acc, acc, acc, acc

class CUBIC_OCRPA(nn.Module):
    def __init__(self, args):
        super(CUBIC_OCRPA, self).__init__()
        self.args = args
        self.sr = F.interpolate
        self.ocr = CRNNPA(self.args)
        self.loss_sr = None
        self.acc_sr = None
        self.loss_ocr = nn.CTCLoss(blank=self.args.num_classes - 1, reduction='mean', zero_infinity=True)
        self.acc_ocr = jacarrd

    def forward(self, data):
        x = data["lr"]
        hr = data["hr"]
        label = data["label"]
        target_lengths = data["target_lengths"]
        x = self.sr(x, scale_factor=2, mode='nearest')
        rlt = self.ocr(x)
        input_lengths = torch.full(size=(rlt.shape[1],), fill_value=rlt.shape[0], dtype=torch.long)
        if self.training:
            rlt = torch.nn.functional.log_softmax(rlt, dim=-1)
            loss = self.loss_ocr(rlt, label, input_lengths, target_lengths)
            return loss, loss, loss, loss, loss
        else:
            rlt = torch.argmax(rlt, dim=-1).permute(1, 0)
            rlt = convert(rlt, self.args.num_classes - 1)
            acc = self.acc_ocr(rlt, label)  # label 是拼接起来的，一个batch算一次acc
            return acc, acc, acc, acc, acc

class SR_OCRPA(nn.Module):
    def __init__(self, args):
        super(SR_OCRPA, self).__init__()
        self.args = args
        self.sr = NOPAN(self.args)
        self.ocr = CRNNPA(self.args)
        self.loss_sr = nn.L1Loss()
        self.acc_sr = psnr
        self.loss_ocr = nn.CTCLoss(blank=self.args.num_classes - 1, reduction='mean', zero_infinity=True)
        self.acc_ocr = jacarrd

    def forward(self, data):
        x = data["lr"]
        hr = data["hr"]
        label = data["label"]
        target_lengths = data["target_lengths"]
        x = self.sr(x, scale_factor=2, mode='nearest')
        rlt = self.ocr(x)
        input_lengths = torch.full(size=(rlt.shape[1],), fill_value=rlt.shape[0], dtype=torch.long)
        if self.training:
            rlt = torch.nn.functional.log_softmax(rlt, dim=-1)
            loss1 = self.loss_ocr(rlt, label, input_lengths, target_lengths)
            loss2 = self.loss_sr(x, hr)
            loss = loss1 + loss2
            return loss, loss1, loss2, loss2, loss2
        else:
            rlt = torch.argmax(rlt, dim=-1).permute(1, 0)
            rlt = convert(rlt, self.args.num_classes - 1)
            acc1 = self.acc_ocr(rlt, label)  # label 是拼接起来的，一个batch算一次acc
            acc2 = self.acc_sr(x, hr)
            return acc1, acc2, acc2, acc2, acc2


class SRPA_OCRPA(nn.Module):
    def __init__(self, args):
        super(SRPA_OCRPA, self).__init__()
        self.args = args
        self.sr = PAN(self.args)
        self.ocr = CRNNPA_PERCEPTUAL(self.args)
        self.loss_sr = nn.L1Loss()
        self.acc_sr = psnr
        self.loss_ocr = nn.CTCLoss(blank=self.args.num_classes - 1, reduction='mean', zero_infinity=True)
        self.acc_ocr = jacarrd

    def forward(self, data):
        x = data["lr"]
        hr = data["hr"]
        label = data["label"]
        target_lengths = data["target_lengths"]
        x = self.sr(x, scale_factor=2, mode='nearest')
        rlt = self.ocr(x)
        input_lengths = torch.full(size=(rlt.shape[1],), fill_value=rlt.shape[0], dtype=torch.long)
        if self.training:
            rlt = torch.nn.functional.log_softmax(rlt, dim=-1)
            loss1 = self.loss_ocr(rlt, label, input_lengths, target_lengths)
            loss2 = self.loss_sr(x, hr)
            loss = loss1 + loss2
            return loss, loss1, loss2, loss2, loss2
        else:
            rlt = torch.argmax(rlt, dim=-1).permute(1, 0)
            rlt = convert(rlt, self.args.num_classes - 1)
            acc1 = self.acc_ocr(rlt, label)  # label 是拼接起来的，一个batch算一次acc
            acc2 = self.acc_sr(x, hr)
            return acc1, acc2, acc2, acc2, acc2

class SRPA_OCRPA_PER(nn.Module):
    def __init__(self, args):
        super(SRPA_OCRPA_PER, self).__init__()
        self.args = args
        self.sr = PAN(self.args)
        self.ocr = CRNNPA_PERCEPTUAL(self.args)
        self.loss_sr = nn.L1Loss()
        self.acc_sr = psnr
        self.loss_ocr = nn.CTCLoss(blank=self.args.num_classes - 1, reduction='mean', zero_infinity=True)
        self.acc_ocr = jacarrd
        self.loss_per = nn.L1Loss()

    def forward(self, data):
        x = data["lr"]
        hr = data["hr"]
        label = data["label"]
        target_lengths = data["target_lengths"]
        x = self.sr(x, scale_factor=2, mode='nearest')
        rlt, cnn = self.ocr(x)
        input_lengths = torch.full(size=(rlt.shape[1],), fill_value=rlt.shape[0], dtype=torch.long)
        if self.training:
            rlt = torch.nn.functional.log_softmax(rlt, dim=-1)
            loss1 = self.loss_ocr(rlt, label, input_lengths, target_lengths)
            loss2 = self.loss_sr(x, hr)
            rlt_hr, cnn_hr = self.ocr(hr)
            rlt_sr, cnn_sr = self.ocr(x)
            loss3 = self.loss_per(cnn_hr, cnn_sr)
            loss = loss1 + loss2 + loss3
            return loss, loss1, loss2, loss3, loss3
        else:
            rlt = torch.argmax(rlt, dim=-1).permute(1, 0)
            rlt = convert(rlt, self.args.num_classes - 1)
            acc1 = self.acc_ocr(rlt, label)  # label 是拼接起来的，一个batch算一次acc
            acc2 = self.acc_sr(x, hr)
            return acc1, acc2, acc2, acc2, acc2



