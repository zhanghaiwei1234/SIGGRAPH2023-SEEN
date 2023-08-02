# -*- coding: utf-8 -*
import torch.nn as nn
import torch
from opts import parse_opts
import numpy as np

opts = parse_opts()
thresh = opts.thresh  # neuronal threshold
spike_thresh = opts.spike_thresh # neuronal threshold
lens = opts.lens  # hyper-parameters of approximate function
decay = opts.decay  # decay constants
beta = opts.beta

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
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

opts.sample_size = int(opts.sample_size)
cfg_kernel = [int(opts.sample_size/2), int(opts.sample_size/4)+1, int(opts.sample_size/16)+1]

cfg_fc = [256, 7]
device = 'cuda:0'

class inference_SNNCNN3(nn.Module):

    def __init__(self, train_model):
        super(inference_SNNCNN3, self).__init__()

        self.conv1_f2 = train_model.module.conv1_f2
        self.conv1_f3 = train_model.module.conv1_f3
        self.conv1_f4 = train_model.module.conv1_f4
        self.conv1_cat = train_model.module.conv1_cat

        self.conv2_f2 = train_model.module.conv2_f2

        self.conv3_f2 = train_model.module.conv3_f2

        self.fc1 = train_model.module.fc1
        self.fc2 = train_model.module.fc2
        self.sa = train_model.module.sa

        self.conv0 = train_model.module.conv0

        self.mlp_f1 = train_model.module.mlp_f1
        self.mlp_f2 = train_model.module.mlp_f2
        self.mlp_f3 = train_model.module.mlp_f3
        
        self.softmax_weight = train_model.module.softmax_weight

    def forward(self, frame, event, weight_old):
        batch_size = event.shape[0]
        time_window = event.shape[2]
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_summem = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(batch_size, cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], device=device)
              
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
        frame_fea1 = torch.cat([w1 * frame_fea1_2, w2 * frame_fea1_3, w3 * frame_fea1_4], dim=1)
        frame_fea1 = self.conv1_cat(frame_fea1)
        frame_fea2 = self.conv2_f2(frame_fea1)
        frame_fea3 = self.conv3_f2(frame_fea2)

        frame_fea_fc = frame_fea3.view(batch_size, -1)
        h3_mem, h3_spike = mem_update(frame_fea_fc, h3_mem, h3_spike)

        for step in range(time_window):  # simulation time steps

            event_fea = event[:, :, step, :, :]
            first2 = nn.functional.conv2d(event_fea, weight_old[0], bias=None, stride=2, padding=1, dilation=1, groups=1).detach()
            first3 = nn.functional.conv2d(event_fea, weight_old[1], bias=None, stride=2, padding=2, dilation=1, groups=1).detach()
            first4 = nn.functional.conv2d(event_fea, weight_old[2], bias=None, stride=2, padding=3, dilation=1, groups=1).detach()

            first = torch.cat([w1 * first2, w2 * first3, w3 * first4], dim=1)
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
        return outputs
