from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
import time

from torchvision.transforms.transforms import ColorJitter
from opts import parse_opts
# from model import generate_model, make_data_parallel
#from model_snn import generate_model_snn, make_data_parallel
#from model_snn_cnn import generate_model_snn, make_data_parallel
from model_snn_cnn import generate_model_snn, make_data_parallel
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from datasets import get_training_data, get_inference_data
from utils import Logger, worker_init_fn, get_lr
from training import train_epoch
import inference


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.event_video_path = opt.root_path / opt.event_video_path
        opt.frame_video_path = opt.root_path / opt.frame_video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1

    opt.event_mean, opt.event_std = get_mean_std(opt.value_scale, data_type='event')
    opt.frame_mean, opt.frame_std = get_mean_std(opt.value_scale, data_type='frame')

    opt.n_input_channels = 3

    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    return opt

def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt, model_parameters):

    event_spatial_transform = []
    frame_spatial_transform = []
    event_normalize = get_normalize_method(opt.event_mean, opt.event_std, opt.no_mean_norm,opt.no_std_norm)
    frame_normalize = get_normalize_method(opt.frame_mean, opt.frame_std, opt.no_mean_norm,opt.no_std_norm)
                                           
    event_spatial_transform = [Resize(opt.sample_size)]
    frame_spatial_transform = [Resize(opt.sample_size)]
    #event_spatial_transform.append(CenterCrop(opt.sample_size))
    #frame_spatial_transform.append(CenterCrop(opt.sample_size))
    
    event_spatial_transform.append(ToTensor())
    frame_spatial_transform.append(ToTensor())
    event_spatial_transform.append(ScaleValue(opt.value_scale))
    frame_spatial_transform.append(ScaleValue(opt.value_scale))
    event_spatial_transform.append(event_normalize)
    frame_spatial_transform.append(frame_normalize)

    event_spatial_transform = Compose(event_spatial_transform)
    frame_spatial_transform = Compose(frame_spatial_transform)

    assert opt.train_t_crop in ['random', 'center']
    # temporal_transform : random
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_data = get_training_data(opt.event_video_path,
                                   opt.frame_video_path, opt.annotation_path,
                                   event_spatial_transform, frame_spatial_transform, temporal_transform)
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
        print('zhanghaiwei')
    else:
        train_sampler = None
        print('zhanghaixu')

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None) ,
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    assert opt.lr_scheduler in ['plateau', 'multistep','singlestep']
    # lr_scheduler:
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.94)

    return (train_loader, train_sampler, train_logger, train_batch_logger,
            optimizer, scheduler)

def get_inference_utils(opt):
    event_normalize = get_normalize_method(opt.event_mean, opt.event_std, opt.no_mean_norm,opt.no_std_norm)
    frame_normalize = get_normalize_method(opt.frame_mean, opt.frame_std, opt.no_mean_norm,opt.no_std_norm)
    event_spatial_transform = [Resize(opt.sample_size)]
    frame_spatial_transform = [Resize(opt.sample_size)]
    #event_spatial_transform.append(CenterCrop(opt.sample_size))
    #frame_spatial_transform.append(CenterCrop(opt.sample_size))

    event_spatial_transform.append(ToTensor())
    frame_spatial_transform.append(ToTensor())
    event_spatial_transform.append(ScaleValue(opt.value_scale))
    event_spatial_transform.append(event_normalize)
    frame_spatial_transform.append(ScaleValue(opt.value_scale))
    frame_spatial_transform.append(frame_normalize)

    event_spatial_transform = Compose(event_spatial_transform)
    frame_spatial_transform = Compose(frame_spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    #inference_sample_duration = random.randint(4, 17)
    temporal_transform.append(TemporalRandomCrop(opt.inference_sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)
    #print(inference_sample_duration)
    inference_data, collate_fn = get_inference_data(
        opt.event_video_path, opt.frame_video_path, opt.annotation_path, opt.inference_subset, event_spatial_transform, frame_spatial_transform, temporal_transform)

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)

    return inference_loader, inference_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
        print('look look look')
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)

def main_worker(index, opt):
    
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)
    os.environ['PYTHONHASHSEED'] = str(opt.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    
    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

   
    model = generate_model_snn()

    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports Distributed DataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = make_data_parallel(model, opt.distributed, opt.device)
    
    #pre_trained_dict = torch.load('/video-emotion-classfication/FER2013/yu/save_170.pth')['state_dict']
    #part_sd = {k: v for k, v in pre_trained_dict.items() if k in ['conv1_f2.weight', 'conv1_f2.bias', 'conv1_f3.weight', 'conv1_f3.bias', 'conv1_f4.weight', 'conv1_f4.bias', 'conv1_cat.weight', 'conv1_cat.bias', 'conv2_f2.weight', 'conv2_f2.bias', 'conv3_f2.weight', 'conv3_f2.bias', 'conv0.weight', 'conv0.bias']}    
    #model.module.state_dict().update(part_sd)
 
    if opt.is_master_node:
        print(model)
        
    parameters = model.parameters()
    criterion = CrossEntropyLoss().to(opt.device)

    if not opt.no_train:
        (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            current_lr = get_lr(optimizer)
            weight = train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer,opt.distributed)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)
        else:
            scheduler.step()

        if opt.inference and i >= 140:
        # if opt.inference:
            for test_num in range(20):
                tt = time.time()
                inference_loader, inference_class_names = get_inference_utils(opt)
                #print('dataloader time', time.time()-tt)
                inference_result_path = opt.result_path / '{}.json'.format(opt.inference_subset + '_epoch_' + str(i) + '_' + str(test_num+1))

                inference.inference(inference_loader, model, weight, inference_result_path, inference_class_names, opt.inference_no_average, opt.output_topk, i, tb_writer, opt.distributed, opt.device, str(test_num+1))

            


if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    print('opt.ngpus_per_node',opt.ngpus_per_node)

    main_worker(-1, opt)