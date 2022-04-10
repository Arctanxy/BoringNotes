import math
import torch
from torch import nn  
import torch.nn.functional as F 

class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None) -> None:
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(mid_channels),
			nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	def __init__(self, in_channels, out_channels) -> None:
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)
	
	def forward(self,x):
		return self.maxpool_conv(x)

class Up(nn.Module):
	def __init__(self, in_channels, out_channels, bilinear=True) -> None:
		super().__init__()
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
		else:
			self.up = nn.ConvTransposed2d(in_channels, in_channels //2, kernel_size = 2, stride = 2)
			self.conv = DoubleConv(in_channels, out_channels)
	
	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim = 1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels) -> None:
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size =1 )

	def forward(self, x):
		return self.conv(x)

class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=True):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.biliear = bilinear

		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64,128)
		self.down2 = Down(128,256)
		self.down3 = Down(256,512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512,1024 // factor)
		self.up1 = Up(1024, 512//factor, bilinear)
		self.up2 = Up(512, 256//factor, bilinear)
		self.up3 = Up(256,128//factor, bilinear)
		self.up4 = Up(128,64,bilinear)
		self.outc = OutConv(64, n_classes)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5,x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return torch.sigmoid(logits)

class DRRN(nn.Module):
	def __init__(self, layers = 5):
		super(DRRN, self).__init__()
		self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		# self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
		self.layers = layers

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))

	def forward(self, x):
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
		for _ in range(self.layers): # 25 is too big to run on rtx3070
			out = self.conv2(self.relu(self.conv1(self.relu(out))))
			out = torch.add(out, inputs)

		out = self.output(self.relu(out))

		out = torch.add(out, residual)
		return torch.sigmoid(out)


class DRNN_Network(nn.Module):
	def __init__(self, args) -> None:
		super(DRNN_Network, self).__init__()
		self.args = args
		self.model = DRRN()
		self.loss = torch.nn.L1Loss()

	def forward(self, data):
		if self.training:
			lr = data["lr"]
			hr = data["hr"]
			out = self.model(lr)
			# from torchvision import transforms
			# out_img = transforms.ToPILImage()(out[0].cpu())
			# in_img = transforms.ToPILImage()(hr[0].cpu())
			# import pdb;pdb.set_trace()
			out = out * data["mask"]
			hr = hr * data["mask"]
			loss = self.loss(out, hr)
			return loss
		else:
			lr = data["lr"]
			hr = data["hr"]
			out = self.model(lr)
			out *= data["mask"]
			hr *= data["mask"]
			loss = self.loss(out, hr)
			return loss

	def predict(self, image_tensor):
		out = self.model(image_tensor)
		return out


class UNet_Network(nn.Module):
	def __init__(self, args) -> None:
		super(UNet_Network, self).__init__()
		self.args = args
		self.model = UNet(1,1)
		self.loss = torch.nn.L1Loss()

	def forward(self, data):
		if self.training:
			lr = data["lr"]
			hr = data["hr"]
			out = self.model(lr)
			# from torchvision import transforms
			# out_img = transforms.ToPILImage()(out[0].cpu())
			# in_img = transforms.ToPILImage()(hr[0].cpu())
			# import pdb;pdb.set_trace()
			out = out * data["mask"]
			hr = hr * data["mask"]
			loss = self.loss(out, hr)
			return loss
		else:
			lr = data["lr"]
			hr = data["hr"]
			out = self.model(lr)
			out *= data["mask"]
			hr *= data["mask"]
			loss = self.loss(out, hr)
			return loss

	def predict(self, image_tensor):
		out = self.model(image_tensor)
		return out
