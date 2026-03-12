import time

import torch
import torch.nn as nn
from torchvision import transforms

from datasets import data_transforms
from tools import builder
from utils import dist_utils, misc
from utils.AverageMeter import AverageMeter
from utils.logger import *

train_transforms = transforms.Compose([data_transforms.PointcloudScaleAndTranslate()])


def run_net(args, config, train_writer=None):
    logger = get_logger(args.log_name)
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)

    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    start_epoch = 0

    if args.resume:
        start_epoch, _ = builder.resume_model(base_model, args, logger = logger)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    elif args.use_gpu:
        base_model = nn.DataParallel(base_model).cuda()
        print_log('Using Data parallel ...' , logger = logger)
    else:
        print_log('Using CPU training ...', logger = logger)

    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0
        n_batches = len(train_dataloader)
        rank, world_size = dist_utils.get_dist_info()

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name != 'H5Cluster':
                raise NotImplementedError(f'CMAT stage-2 only supports H5Cluster, but got {dataset_name}')

            if not isinstance(data, (list, tuple)) or len(data) < 2:
                raise RuntimeError('H5Cluster dataloader must return (points, cluster_feats, labels)')

            points = data[0].cuda(non_blocking=True) if args.use_gpu else data[0]
            cluster_feats = data[1].cuda(non_blocking=True) if args.use_gpu else data[1]

            assert points.size(1) == npoints
            points = train_transforms(points)

            loss_dict = base_model(points, cluster_feats=cluster_feats)
            loss = loss_dict.pop('loss_for_backward').mean()
            loss.backward()

            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            for k in list(loss_dict.keys()):
                value = loss_dict[k]
                if isinstance(value, torch.Tensor):
                    if value.ndim > 0:
                        value = value.mean()
                    loss_dict[k] = value.item()
            loss_dict['total_loss'] = loss.item()
            losses.update([loss.item() * 1000])

            if rank == 0:
                if train_writer is not None:
                    train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                if rank == 0:
                    if getattr(config.model, 'return_loss_dict', False):
                        lambda_struct = loss_dict.get('lambda_struct', 0.0)
                        loss_str = (
                            f"Total: {loss_dict['total_loss']:.4f}, "
                            f"Recon: {loss_dict.get('recon_loss', 0.0):.4f}, "
                            f"Struct: {loss_dict.get('struct_loss', 0.0):.4f} "
                            f"(lambda={lambda_struct:.3f})"
                        )
                        print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) %s lr = %.6f' %
                                    (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                                    loss_str, optimizer.param_groups[0]['lr']), logger = logger)
                    else:
                        print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                                    (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                                    ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if scheduler is not None:
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
        epoch_end_time = time.time()

        if rank == 0:
            if train_writer is not None:
                train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            if getattr(config.model, 'return_loss_dict', False):
                print_log('[Training] EPOCH: %d EpochTime = %.3f (s) AvgLoss = %.4f lr = %.6f' %
                    (epoch,  epoch_end_time - epoch_start_time, losses.avg()[0],
                    optimizer.param_groups[0]['lr']), logger = logger)
            else:
                print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                    (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                    optimizer.param_groups[0]['lr']), logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-last', args, logger = logger)
        if epoch > 0 and epoch % 25 == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, None, None, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)

    if train_writer is not None:
        train_writer.close()