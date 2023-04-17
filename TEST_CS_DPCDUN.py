import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
from torch.nn import init
import copy
import cv2
from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser(description='DPC-DUN')

parser.add_argument('--layer_num', type=int, default=25, help='stage number of DPCDUN')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--noise', type=float, default=0, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--lambda_index', type=int, default=5, help='from {0, 1, 2, 3, 4, 5}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--channels', type=int, default=32, help='1 for gray, 3 for color')
# parser.add_argument('--gate1_weight', type=float, default=0.001)
# parser.add_argument('--gate2_weight', type=float, default=0.002)
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='./model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='./Dataset', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='./result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--patch_size', type=int, default=99)
parser.add_argument('--algo_name', type=str, default='DPCDUN', help='log directory')

args = parser.parse_args()

layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
channels = args.channels
noise = args.noise
lambda_index = args.lambda_index

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 5:55, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}
lambda_list = {0:0.002, 1:0.001, 2:0.0005, 3:0.0001, 4:0.00005, 5:0.00001}

n_input = ratio_dict[cs_ratio]
n_output = 1089

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']

def gumbel_softmax(x, dim=-1):

#     gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    y_soft = x.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(x).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

# Define CU(Controllable Unit)
class Lambda_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(Lambda_Conv, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.fc2 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)

    def forward(self, x, lam):
        lam = lam.unsqueeze(-1).unsqueeze(-1)
        s = self.fc1(lam)
        s = F.softplus(s)
        b = self.fc2(lam)
        x = s*self.conv(x)+b
        return x

# Define PCS (Path-Controllable Selector)
class Attention_SEblock(nn.Module):
    def __init__(self, channels, reduction):
        super(Attention_SEblock, self).__init__()
        self.conv = Lambda_Conv(6, channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2)
        self.fc3 = nn.Linear(channels // reduction, 2)
        self.fc2.bias.data[0] = 0.1 
        self.fc2.bias.data[1] = 2
        self.fc3.bias.data[0] = 0.1
        self.fc3.bias.data[1] = 2
        self.channels = channels
    def forward(self, x, lam):
        x = self.conv(x, lam)
        x = self.avg_pool(x).view(-1, self.channels)
        x = self.fc1(x)
        x = self.relu(x)
        x_2 = self.fc2(x)  
        x_2 = gumbel_softmax(x_2)
        x_3 = self.fc3(x)   
        x_3 = gumbel_softmax(x_3)
        return x_2, x_3

# Define RB
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):

        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res + input
        return x

# Define one stage in DPC-DUN
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        rb_num = 2
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        modules_body = [ResidualBlock(32, 32, 3, bias=True, res_scale=1) for _ in range(rb_num)]
        self.body = nn.Sequential(*modules_body)
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        
    def forward(self, x, z, PhiWeight, PhiTWeight, PhiTb, gate1, gate2):
        
        if gate1[:,1] == 0:
            x_input = x
        else:
            x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
            x_input = x + self.lambda_step * PhiTb
       
        x_input = torch.cat([x_input, z], 1)
        
        if gate2[:,1] == 0:
            x_pred = x_input
        else:
            x_D = F.conv2d(x_input, self.conv_D, padding=1)
            x_backward = self.body(x_D)
            x_G = F.conv2d(x_backward, self.conv_G, padding=1)
            x_pred = x_input + x_G

        return x_pred

# Define DPC-DUN
class DPCDUN(torch.nn.Module):
    def __init__(self, LayerNo):
        super(DPCDUN, self).__init__()
        onelayer = []
        gates = []
        self.LayerNo = LayerNo
        n_feat = channels - 1

        for i in range(LayerNo):
            onelayer.append(BasicBlock())
        self.fcs = nn.ModuleList(onelayer)
        for i in range(LayerNo):
            gates.append(Attention_SEblock(channels, 4))
        self.gates = nn.ModuleList(gates)
    
        self.fe = nn.Conv2d(1, n_feat, 3, padding=1, bias=True)

    def forward(self, Phix, Phi, lamda):

        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None) # 64*1089*3*3 
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb
        z = self.fe(x)
        lamda = lamda.repeat(x.shape[0], 1).type(torch.FloatTensor).to(device)
        gate1_s = []
        gate2_s = []

        for i in range(self.LayerNo):
            if i==0:
                gate1, gate2 = self.gates[i](torch.cat([x, z], 1), lamda)
            else:
                gate1, gate2 = self.gates[i](x_dual, lamda)
            x_dual = self.fcs[i](x, z, PhiWeight, PhiTWeight, PhiTb, gate1, gate2)
            x = x_dual[:, :1, :, :]
            z = x_dual[:, 1:, :, :]
            gate1_s.append(gate1[:,1])
            gate2_s.append(gate2[:,1])

        x_final = x
        gate1_s = torch.cat(gate1_s, 0)
        gate2_s = torch.cat(gate2_s, 0)

        return x_final, gate1_s, gate2_s

model = DPCDUN(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

num_params = 0
for para in model.parameters():
    num_params += para.numel()
print("total para num: %d\n" %num_params)

model_dir = "%s/CS_%s_layer_%d_ratio_%d" % (args.model_dir, args.algo_name, layer_num, cs_ratio)

  
if cs_ratio==25:
    epoch_num=402
elif cs_ratio==30:
    epoch_num=410
elif cs_ratio == 40:
    epoch_num = 404
elif cs_ratio == 50: 
    epoch_num = 406
else:
    epoch_num=403
    

model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch_num)))
print('\n')
print("CS Reconstruction Start")

def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


test_dir = os.path.join(args.data_dir, test_name)
if test_name=='Set11':
    filepaths = glob.glob(test_dir + '/*.tif')
else:
    filepaths = glob.glob(test_dir + '/*.png')

result_dir = os.path.join(args.result_dir, test_name)
result_dir = os.path.join(result_dir, str(args.cs_ratio))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
gate1_ALL = np.zeros([1, ImgNum], dtype=np.float32)
gate2_ALL = np.zeros([1, ImgNum], dtype=np.float32)

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]

        Img = cv2.imread(imgName, 1)
        ImgN = imgName.split('/')[-1].split('.')[0]

        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()

        Iorg_y = Img_yuv[:,:,0]
        Iorg = Iorg_y.copy()

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
        Img_output = Ipad.reshape(1, 1, Ipad.shape[0], Ipad.shape[1])/255.0

        start = time()

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(batch_x, PhiWeight, padding=0, stride=33, bias=None)
        noise_sigma = noise/255.0 * torch.randn_like(Phix)
        Phix = Phix + noise_sigma

        lamblist_encoding = torch.nn.functional.one_hot(torch.tensor([0,1,2,3,4,5]))
        lambda_encoding = lamblist_encoding[lambda_index]
        lambda_value = lambda_list[lambda_index]

        x_output, gate1_s, gate2_s = model(Phix, Phi, lambda_encoding)

        end = time()

        Prediction_value = x_output.cpu().data.numpy().squeeze()
        row = Iorg.shape[0]
        col = Iorg.shape[1]

        gates1 = gate1_s.cpu().data.numpy().squeeze()
        gates2 = gate2_s.cpu().data.numpy().squeeze()

        X_rec = np.clip(Prediction_value[0:row, 0:col], 0, 1)

        rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

        print("[%02d/%02d] Run time for %s is %.4f, sum_gate1 is %d, sum_gate2 is %d, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), np.sum(gates1), np.sum(gates2), rec_PSNR, rec_SSIM))

        Img_rec_yuv[:,:,0] = X_rec*255

        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

        cv2.imwrite("%s/%s_%s_lambda_%.5f_ratio_%d_PSNR_%.2f_SSIM_%.4f.png" % (result_dir, ImgN, args.algo_name, lambda_value, cs_ratio, rec_PSNR, rec_SSIM), im_rec_rgb)
        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM
        gate1_ALL[0, img_no] = np.sum(gates1)
        gate2_ALL[0, img_no] = np.sum(gates2)

print('\n')
output_data = "CS ratio is %d, lambda is %.5f, Avg PSNR/SSIM for %s is %.2f/%.4f, Avg gate1/gate2 is %d/%d, Epoch number of model is %d \n" % (cs_ratio, lambda_list[lambda_index], args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(gate1_ALL), np.mean(gate2_ALL), epoch_num)
print(output_data)

print("CS Reconstruction End")
