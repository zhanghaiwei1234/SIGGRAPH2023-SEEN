# -*- coding: utf-8 -*

import torch.nn as nn
import torch
from opts import parse_opts
import torch.nn.functional as F

opts = parse_opts()
thresh = opts.thresh # neuronal threshold
lens = opts.lens  # hyper-parameters of approximate function
decay = opts.decay  # decay constants
beta = opts.beta
# global thresh

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # print('threshold.............', thresh)
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply


# membrane potential update
def mem_update(x, mem, spike):
    mem = mem * decay * (1. - spike) + x
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike

cfg_cnn = [(3, 64, 2, 1, 3),
           (64, 256, 2, 1, 3),
           (256, 512, 4, 1, 3)]
# kernel size
opts.sample_size = int(opts.sample_size)
cfg_kernel = [int(opts.sample_size/2), int(opts.sample_size/4)+1,  int(opts.sample_size/16)+1]
# cfg_kernel_first = [59, 26, 11, 15, 15]
# fc layer
cfg_fc = [256, 7]
device = 'cuda:0'

class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

'''
cross- channel attention
'''
class CA(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(CA, self).__init__()
        self.filter = FilterLayer(2*in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = mainPath + channel_weight * guidePath
        return out

'''spatial attention'''
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, frame, event):
        x  = torch.cat([frame, event], dim=1)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return event * scale + event

class SNNCNN3(nn.Module):
    r"""
    SNN

    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    """
    def __init__(self):
        super(SNNCNN3, self).__init__()
        
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.dim = out_planes

        self.conv1_f2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
        self.conv1_f3 = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2)
        self.conv1_f4 = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3)

        self.conv1_cat = nn.Conv2d(3*out_planes, out_planes, kernel_size=1, stride=1, padding=0)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2_f2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3_f2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        
        self.conv0 = nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0)

        self.sa = SA()

        self.mlp_f1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Conv2d(self.dim, self.dim // 8, 1, 1, 0),
                                     nn.BatchNorm2d(self.dim // 8),
                                     nn.ReLU(),
                                     nn.Conv2d(self.dim // 8, 1, 1, 1, 0))
        self.mlp_f2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(self.dim, self.dim // 8, 1, 1, 0),
                                       nn.BatchNorm2d(self.dim // 8),
                                       nn.ReLU(),
                                       nn.Conv2d(self.dim // 8, 1, 1, 1, 0))
        self.mlp_f3 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(self.dim, self.dim // 8, 1, 1, 0),
                                       nn.BatchNorm2d(self.dim // 8),
                                       nn.ReLU(),
                                       nn.Conv2d(self.dim // 8, 1, 1, 1, 0))

        self.softmax_weight = nn.Softmax(dim=1)

    def forward(self, frame, event, weight_old):
        batch_size = event.shape[0]
        time_window = event.shape[2]

        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c3_summem = c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        h1_mem = h1_spike = h1_summem = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_summem = torch.zeros(batch_size, cfg_fc[1], device=device)
        
        h3_mem = h3_spike = h3_summem = torch.zeros(batch_size, cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], device=device)
        
       
        frame_fea = torch.cat([frame[:, :, 0, :, :],frame[:, :, -1, :, :]],dim=1)
        frame_fea = self.conv0(frame_fea)

        frame_fea1_2 = self.conv1_f2(frame_fea)
        frame_fea1_3 = self.conv1_f3(frame_fea)
        frame_fea1_4 = self.conv1_f4(frame_fea)

        w1 = self.mlp_f1(frame_fea1_2)
        w2 = self.mlp_f2(frame_fea1_3)
        w3 = self.mlp_f3(frame_fea1_4)
        softmax_weight = self.softmax_weight(torch.cat((w1, w2, w3), 1))
        w1, w2, w3 = softmax_weight.split(1, dim=1)

        frame_fea1 = torch.cat([w1*frame_fea1_2, w2*frame_fea1_3, w3*frame_fea1_4], dim=1)
        frame_fea1 = self.conv1_cat(frame_fea1)

        frame_fea2 = self.conv2_f2(frame_fea1)

        frame_fea3 = self.conv3_f2(frame_fea2)

        frame_fea_fc = frame_fea3.view(batch_size, -1)
        h3_mem, h3_spike = mem_update(frame_fea_fc, h3_mem, h3_spike)


        conv12_weight_new = beta * weight_old[0] + (1 - beta) * self.conv1_f2.weight.data
        conv13_weight_new = beta * weight_old[1] + (1 - beta) * self.conv1_f3.weight.data
        conv14_weight_new = beta * weight_old[2] + (1 - beta) * self.conv1_f4.weight.data
        conv1c_weight_new = beta * weight_old[3] + (1 - beta) * self.conv1_cat.weight.data
        conv22_weight_new = beta * weight_old[4] + (1 - beta) * self.conv2_f2.weight.data
        conv32_weight_new = beta * weight_old[5] + (1 - beta) * self.conv3_f2.weight.data


        weight_old = [conv12_weight_new, conv13_weight_new, conv14_weight_new, \
                      conv1c_weight_new, conv22_weight_new, conv32_weight_new]

        for step in range(time_window):  # simulation time steps
            
            event_fea = event[:, :, step, :, :]

            first2 = nn.functional.conv2d(event_fea, weight_old[0], bias=None, stride=2, padding=1, dilation=1, groups=1).detach()
            first3 = nn.functional.conv2d(event_fea, weight_old[1], bias=None, stride=2, padding=2, dilation=1, groups=1).detach()
            first4 = nn.functional.conv2d(event_fea, weight_old[2], bias=None, stride=2, padding=3, dilation=1, groups=1).detach()
            
            first = torch.cat([w1*first2, w2*first3, w3*first4], dim=1)
            first = nn.functional.conv2d(first, weight_old[3], bias=None, stride=1, padding=0, dilation=1, groups=1).detach()
            c1_mem, c1_spike = mem_update(first, c1_mem, c1_spike)

            second = nn.functional.conv2d(c1_spike, weight_old[4], bias=None, stride=2, padding=1, dilation=1, groups=1).detach()
            c2_mem, c2_spike = mem_update(second, c2_mem, c2_spike)

            thred = nn.functional.conv2d(c2_spike, weight_old[5], bias=None, stride=4, padding=1, dilation=1, groups=1).detach()
            
            c3_mem, c3_spike = mem_update(thred, c3_mem, c3_spike)

            event_fea = c3_spike.view(batch_size, -1)

            x = event_fea + h3_spike

            x = self.fc1(x)
            h1_mem, h1_spike = mem_update(x, h1_mem, h1_spike)
            x = self.fc2(h1_spike)
            h2_mem, h2_spike = mem_update(x, h2_mem, h2_spike)
            h2_summem += h2_mem

        outputs = h2_summem / time_window
        return outputs, weight_old

def generate_model_snn():

    # if model_depth == 18:
    model = SNNCNN3()

    return model

def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model

if __name__ == '__main__':
    event_inputs = torch.rand((32, 3, 8, 180, 180)).cuda()
    frame_inputs = torch.rand((32, 3, 8, 180, 180)).cuda()
    Net = SNNCNN3().cuda()
    output = Net(event_inputs, frame_inputs,weight_old)
    print("output = ", output.shape)
