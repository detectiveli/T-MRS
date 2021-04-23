import os
import pprint
import shutil
import time

import json
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from pedes.data.build import make_dataloader
from pedes.modules import *

POSITIVE_THRESHOLD = 0.5


def cacluate_iou(pred_boxes, gt_boxes):
    x11, y11, x12, y12 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x21, y21, x22, y22 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    interArea = (xB - xA + 1).clip(0) * (yB - yA + 1).clip(0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou


@torch.no_grad()
def test_net(args, config):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    #os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
    config.DATASET.TEST_IMAGE_SET = args.split
    ckpt_path = args.ckpt
    save_path = args.result_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy2(ckpt_path,
                 os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get network
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    print(ckpt_path)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # loader
    test_loader = make_dataloader(config, mode='test', distributed=False)
    test_dataset = test_loader.dataset
    test_database = test_dataset.database

    # test
    ref_ids = []
    pred_boxes = []
    model.eval()
    cur_id = 0

    test_loader.dataset.get_image_features(model.image_feature_extractor)
    score_matrix = torch.zeros((len(test_loader), test_loader.dataset.image_nums)).cuda()#test_loader.dataset.image_nums))  # lihui
    target_matrix = np.zeros((len(test_loader), test_loader.dataset.image_nums))#test_loader.dataset.image_nums))  # lihui

    jump_val = 666

    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
        batch[0] = batch[0].cuda()
    # for nbatch, batch in tqdm(enumerate(test_loader)):
    #     bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size

        for i in range(int(test_loader.dataset.image_nums / jump_val)+1):

            start_i = i*jump_val
            stop_i = (i+1)*jump_val if (i+1)*jump_val < test_loader.dataset.image_nums else test_loader.dataset.image_nums-1

            batch_b = []

            for i in range(stop_i - start_i):
                batch_b.append(batch[0].squeeze(0))
            batch_b = torch.stack(batch_b, dim=0)

            #image_b = test_loader.dataset.val_images[start_i: stop_i]
            boxes_b = test_loader.dataset.val_boxes[start_i: stop_i]
            im_info_b = test_loader.dataset.val_im_info[start_i: stop_i]
            ids_b = test_loader.dataset.val_ids[start_i: stop_i]
            feats = test_loader.dataset.val_feat[start_i: stop_i]

            #image_b = torch.stack(image_b, dim=0)
            boxes_b = torch.stack(boxes_b, dim=0)
            im_info_b = torch.stack(im_info_b, dim=0)
            feats = torch.stack(feats, dim=0).view(-1,feats[0].size(1), feats[0].size(2))

            batch_test = [None, boxes_b, im_info_b, batch_b, None, feats]
            score_none, sim = model(*batch_test)
            score_matrix[nbatch, start_i: stop_i] = sim[:, 0]

            target_matrix[nbatch, start_i: stop_i][np.array(ids_b) == batch[1].numpy()] = 1
            #target_matrix[nbatch, start_i: stop_i][ids_b == batch[1]] = 1

        if nbatch % 10 == 0:
            r1, r5, r10, r50, mAP = compute_topk(score_matrix[:(nbatch + 1)].cpu().numpy(), target_matrix[:(nbatch + 1)])
            print(" r1, r5, r10, r50, mAP: ", r1, r5, r10, r50, mAP)
    #r1, r5, r10 = compute_topk(score_matrix[:(nbatch + 1)].cpu().numpy(), target_matrix[:(nbatch + 1)])
    r1, r5, r10, r50, mAP = compute_topk(score_matrix.cpu().numpy(), target_matrix)
    print(" r1, r5, r10, r50, mAP: ", r1, r5, r10, r50, mAP)

    r1, r5, r10, r50, mAP = compute_topk(score_matrix.cpu().numpy().T, target_matrix.T)
    print(" r1, r5, r10, r50, mAP: ", r1, r5, r10, r50, mAP)
    return None

def compute_topk(sim_cosine, target, k=[1,5,10], reverse=False):
    result = []
    # query = query / query.norm(dim=1,keepdim=True)
    # gallery = gallery / gallery.norm(dim=1,keepdim=True)
    # sim_cosine = torch.matmul(query, gallery.t())
    result.extend(topk(sim_cosine, target))
    # if reverse:
    #     result.extend(topk(sim_cosine, target_query, target_gallery, k=[1,5,10], dim=0))
    return result

def topk(sim, target, k=[1,5,10,50], dim=1):
    result = []
    maxk = max(k)
    size_total = len(target)
    # _, pred_index = sim.topk(maxk, dim, True, True)
    correct = np.zeros(target.shape)
    pred_index_org = np.argsort(-sim,axis=1)# np.argsort(-sim)[:maxk]
    pred_index = pred_index_org[:,:maxk]

    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for i in range(size_total):
        correct[i][:maxk] = target[i][pred_index[i]]

        orig_cmc = target[i][pred_index_org[i]]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc_base = orig_cmc.cumsum()
        cmc = cmc_base
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:maxk])
        num_valid_q += 1.
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = tmp_cmc / (np.arange(tmp_cmc.size) + 1.0)
        tmp_cmc = tmp_cmc * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    mAP = np.mean(all_AP)
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    print(all_cmc[0])
    # if dim == 1:
    #     pred_labels = pred_labels.t()
    # correct = pred_labels.eq(target.view(1,-1).expand_as(pred_labels))

    for topk in k:
        #correct_k = torch.sum(correct[:topk]).float()
        correct_k = np.sum(correct[:,:topk], axis=1)
        correct_k = np.sum(correct_k > 0)
        result.append(correct_k * 100 / size_total)
    result.append(mAP)
    return result

