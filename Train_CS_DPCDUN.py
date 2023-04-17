# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
import random
import csdata_fast
import copy

parser = ArgumentParser(description='DPC-DUN')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=400, help='epoch number of end training')
parser.add_argument('--finetune', type=int, default=10, help='epoch number of finetuning')
parser.add_argument('--layer_num', type=int, default=25, help='stage number of DPCDUN')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {10, 25, 30, 40, 50}')

parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--patch_size', type=int, default=33)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--rgb_range', type=int, default=1, help='value range 1 or 255')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument('--channels', type=int, default=32, help='number of feature map')
# parser.add_argument('--gate1_weight', type=float, default=0.001)
# parser.add_argument('--gate2_weight', type=float, default=0.002)

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='./model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='./Dataset', help='training data directory')
parser.add_argument('--train_name', type=str, default='train400', help='name of test set')
parser.add_argument('--ext', type=str, default='.png', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--algo_name', type=str, default='DPCDUN', help='log directory')
parser.add_argument('--data_copy', type=int, default=200)

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
channels = args.channels
finetune = args.finetune
batch_size = args.batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {10:0, 25:1, 30:2, 40:3, 50:4, 5:5}
n_input_dict = {1: 10, 4: 43, 5:55, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}
lambda_list = {0:0.002, 1:0.001, 2:0.0005, 3:0.0001, 4:0.00005, 5:0.00001}

n_input = n_input_dict[cs_ratio]
n_output = 1089

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 5)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input5 = Phi_data['phi']

Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 10)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input10 = Phi_data['phi']

Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 25)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input25 = Phi_data['phi']

Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 30)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input30 = Phi_data['phi']

Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 40)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input40 = Phi_data['phi']

Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 50)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input50 = Phi_data['phi']


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
        x_2 = F.gumbel_softmax(x_2, tau=1, hard=True)
        x_3 = self.fc3(x)        
        x_3 = F.gumbel_softmax(x_3, tau=1, hard=True)
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
        
        x_gate1 = gate1[:,1].view(-1, 1, 1, 1)
        x_gate2 = gate2[:,1].view(-1, 1, 1, 1)

        x = x - self.lambda_step * x_gate1 * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x_input = x + self.lambda_step * x_gate1 * PhiTb
        x_input = torch.cat([x_input, z], 1)

        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x_backward = self.body(x_D)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_gate2 * x_G

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

        return x_final, gate1_s, gate2_s

model = DPCDUN(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print_flag = 1  # print parameter number

if print_flag:
    num_count = 0
    num_params = 0
    for para in model.parameters():
        num_count += 1
        num_params += para.numel()
        print('Layer %d' % num_count)
        print(para.size())
    print("total para num: %d" % num_params)

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

training_data = csdata_fast.SlowDataset(args)

if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=8,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "%s/CS_%s_layer_%d_ratio_%d" % (args.model_dir, args.algo_name, layer_num, cs_ratio)
log_file_name = "%s/Log_CS_%s_layer_%d_ratio_%d.txt" % (model_dir, args.algo_name, layer_num, cs_ratio)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

Phi5 = torch.from_numpy(Phi_input5).type(torch.FloatTensor).to(device)
Phi10 = torch.from_numpy(Phi_input10).type(torch.FloatTensor).to(device)
Phi25 = torch.from_numpy(Phi_input25).type(torch.FloatTensor).to(device)
Phi30 = torch.from_numpy(Phi_input30).type(torch.FloatTensor).to(device)
Phi40 = torch.from_numpy(Phi_input40).type(torch.FloatTensor).to(device)
Phi50 = torch.from_numpy(Phi_input50).type(torch.FloatTensor).to(device)

Phi_matrix = {0: Phi10, 1: Phi25, 2: Phi30, 3: Phi40, 4: Phi50, 5: Phi5}

media_epoch = end_epoch
if finetune > 0:
    end_epoch = end_epoch + finetune
    patch_size1 = 99

# Training loop
for epoch_i in range(start_epoch + 1, end_epoch + 1):

    if epoch_i > media_epoch:
        args.patch_size = patch_size1
    
    for data in rand_loader:

        batch_x = data
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(-1, 1, args.patch_size, args.patch_size)

        Phi = Phi_matrix[ratio_dict[cs_ratio]]
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(batch_x, PhiWeight, padding=0,stride=33, bias=None)
        
        lambda_index = random.randint(0,5)
        lamblist_encoding = torch.nn.functional.one_hot(torch.tensor([0,1,2,3,4,5]))
        lambda_encoding = lamblist_encoding[lambda_index]
        lambda_value = lambda_list[lambda_index]

        x_output, gate1_s, gate2_s = model(Phix, Phi, lambda_encoding)

        # Compute and print loss
        loss_gate1 = 0
        loss_gate2 = 0
        for i in range(layer_num):
            loss_gate1 = loss_gate1 + gate1_s[i]
            loss_gate2 = loss_gate2 + gate2_s[i]
        loss_gate1 = torch.mean((loss_gate1+1e-6)/layer_num) 
        loss_gate2 = torch.mean((loss_gate2+1e-6)/layer_num)

        loss_rec = nn.L1Loss()(x_output, batch_x)
        loss_all = lambda_value * (loss_gate1 + loss_gate2) + loss_rec

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        output_data = "[%02d/%02d] Loss_gate1: %.4f, Loss_gate2: %.4f, Loss_rec: %.4f, Total Loss: %.4f\n" % (epoch_i, end_epoch, loss_gate1.item(), loss_gate2.item(), loss_rec.item(), loss_all.item())
        print(output_data)

    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 10 == 0 and epoch_i <= 400:
        torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
    elif epoch_i > 400:
        torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
