from __future__ import print_function

import os
import sys
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from comet_ml import Experiment
import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import imageio

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from progressbar import *
from PIL import Image


class Evaluator(object):
    def __init__(self, args, logger: Experiment = None, is_kopf=False, model=None):
        self.args = args
        self.device = torch.device(args.device)
        self.is_kopf = is_kopf

        if logger is None:
            workspace = 'robharb'
            project_name = 'transparent_in_the_wild'
            api_key = 'nb8eG5Ru2ZHIELzbmanxmDsqP'
            logger = Experiment(api_key=api_key, workspace=workspace, project_name=project_name)

        self.logger = logger

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # test dataloader
        if is_kopf:
            val_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
                                                   split='test',
                                                   mode='val',
                                                   transform=input_transform,
                                                   is_kopf=is_kopf,
                                                   base_size=cfg.TRAIN.BASE_SIZE, root='/home/bic/fast-data/TransKopf')
        else:
            val_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
                                                   split='test',
                                                   mode='val',
                                                   transform=input_transform,
                                                   base_size=cfg.TRAIN.BASE_SIZE, root='/home/bic/fast-data/Trans10K')

        # validation dataloader
        # val_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
        #                                        split='validation',
        #                                        mode='val',
        #                                        transform=input_transform,
        #                                        base_size=cfg.TRAIN.BASE_SIZE)


        val_sampler = make_data_sampler(val_dataset, shuffle=False, distributed=args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        logging.info('**** number of images: {}. ****'.format(len(self.val_loader)))

        self.classes = val_dataset.classes
        # create network
        if model is not None:
            self.model = model
        else:
            self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and cfg.MODEL.BN_EPS_FOR_ENCODER:
                logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
                self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        self.model.to(self.device)
        num_gpu = args.num_gpus

        # metric of easy and hard images
        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed, num_gpu)
        self.metric_easy = SegmentationMetric(val_dataset.num_class, args.distributed, num_gpu)
        self.metric_hard = SegmentationMetric(val_dataset.num_class, args.distributed, num_gpu)

        # number of easy and hard images
        self.count_easy = 0
        self.count_hard = 0

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def save_output(self, image, output, output_boundary, filename, pa, eval_iter):
        out_path = os.path.join('output', self.logger.id, str(eval_iter), 'kopf' if self.is_kopf else 'trans')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        imgs = image.permute((0,2,3,1)).detach().cpu().numpy()
        output = output.permute((0,2,3,1)).detach().cpu().numpy()
        output_boundary = output_boundary.permute((0,2,3,1)).detach().cpu().numpy()

        def unnormalize(img):
            img = img-img.min()
            img = (((img)/(img.max()))*255).astype(np.uint8)
            return img

        assert(imgs.shape[0] == 1)
        for idx in range(imgs.shape[0]):
            eval_iter = str(eval_iter)
            imageio.imwrite(os.path.join(out_path, '{:.2f}_'.format(pa) + filename.split('/')[-1].split('.')[0]+ '.jpg'), unnormalize(imgs[idx]))
            imageio.imwrite(os.path.join(out_path, '{:.2f}_'.format(pa) + filename.split('/')[-1].split('.')[0]+ '_output.jpg'), unnormalize(output[idx]))
            imageio.imwrite(os.path.join(out_path, '{:.2f}_'.format(pa) + filename.split('/')[-1].split('.')[0]+ '_output_boundary.jpg'), unnormalize(output_boundary[idx]))

    def eval(self, eval_size=None, eval_iter=0):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        widgets = ['Inference: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=10 * len(self.val_loader)).start()

        for i, (image, target, boundary, filename) in enumerate(self.val_loader):

            if eval_size is not None and i == eval_size:
                break
            image = image.to(self.device)
            target = target.to(self.device)
            boundary = boundary.to(self.device)

            with torch.no_grad():
                output, output_boundary = model.evaluate(image)

            filename = filename[0]

            if 'hard' in filename or self.is_kopf:
                pa, miou = self.metric_hard.update(output, target)
                self.count_hard += 1
            elif 'easy' in filename:
                pa, miou = self.metric_easy.update(output, target)
                self.count_easy += 1
            else:
                print(filename)
                continue

            self.metric.update(output, target)
            pbar.update(10 * i + 1)

            self.save_output(image, output, output_boundary, filename, pa, eval_iter)

        kopf_prefix = 'kopf_' if self.is_kopf else ''
        header = '######\nEvaluation for KOPF' if self.is_kopf else '######\nEvaluation for TransLab'
        print(header)

        pbar.finish()
        synchronize()
        pixAcc, mIoU, category_iou, mae, mBer, category_Ber = self.metric.get(return_category_iou=True)
        self.logger.log_metrics({'{}pixAcc'.format(kopf_prefix):pixAcc, '{}mIoU'.format(kopf_prefix):mIoU, '{}mae'.format(kopf_prefix): mae, '{}mBer'.format(kopf_prefix): mBer})

        pixAcc_e, mIoU_e, category_iou_e, mae_e, mBer_e, category_Ber_e = self.metric_easy.get(return_category_iou=True)
        self.logger.log_metrics({'{}pixAcc_e'.format(kopf_prefix):pixAcc_e, '{}mIoU_e'.format(kopf_prefix):mIoU_e, '{}mae_e'.format(kopf_prefix): mae_e, '{}mBer_e'.format(kopf_prefix): mBer_e})

        pixAcc_h, mIoU_h, category_iou_h, mae_h, mBer_h, category_Ber_h = self.metric_hard.get(return_category_iou=True)
        self.logger.log_metrics({'{}pixAcc_h'.format(kopf_prefix):pixAcc_h, '{}mIoU_h'.format(kopf_prefix):mIoU_h, '{}mae_h'.format(kopf_prefix): mae_h, '{}mBer_h'.format(kopf_prefix): mBer_h})

        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.2f}, mIoU: {:.2f}, mae: {:.3f}, mBer: {:.2f}'.format(
                pixAcc * 100, mIoU * 100, mae, mBer))
        logging.info('End validation easy pixAcc: {:.2f}, mIoU: {:.2f}, mae: {:.3f}, mBer: {:.2f}'.format(
                pixAcc_e * 100, mIoU_e * 100, mae_e, mBer_e))
        logging.info('End validation hard pixAcc: {:.2f}, mIoU: {:.2f}, mae: {:.3f}, mBer: {:.2f}'.format(
                pixAcc_h * 100, mIoU_h * 100, mae_h, mBer_h))

        headers = ['class id', 'class name', 'iou', 'iou_easy', 'iou_hard', 'ber', 'ber_easy', 'ber_hard']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([
                          cls_name, category_iou[i], category_iou_e[i], category_iou_h[i],
                          category_Ber[i], category_Ber_e[i], category_Ber_h[i]
                          ])
            self.logger.log_metrics({'{}{}_iou'.format(kopf_prefix,cls_name): category_iou[i],
                                     '{}{}_iou_e'.format(kopf_prefix,cls_name): category_iou_e[i],
                                     '{}{}_iou_h'.format(kopf_prefix,cls_name): category_iou_h[i],
                                     '{}{}_ber'.format(kopf_prefix,cls_name): category_Ber[i],
                                     '{}{}_ber_e'.format(kopf_prefix,cls_name): category_Ber_e[i],
                                     '{}{}_ber_h'.format(kopf_prefix,cls_name): category_Ber_h[i]})

            logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                               numalign='center', stralign='center')))
        logging.info('easy images: {}, hard images: {}'.format(self.count_easy, self.count_hard))


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    # comet logging
    workspace = 'robharb'
    project_name = 'transparent_in_the_wild'
    api_key = 'nb8eG5Ru2ZHIELzbmanxmDsqP'

    comet_exp = Experiment(api_key=api_key, workspace=workspace, project_name=project_name)

    evaluator = Evaluator(args, logger=comet_exp, is_kopf=False)
    evaluator.eval()

    kopf_evaluator = Evaluator(args, logger=comet_exp, is_kopf=True)
    kopf_evaluator.eval()
