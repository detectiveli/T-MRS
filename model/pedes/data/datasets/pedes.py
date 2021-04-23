import os
import json
import _pickle as cPickle
from PIL import Image
import base64
import numpy as np
import time
import logging
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist
from common.utils.bbox import bbox_iou_py_vectorized
from common.utils.clip_pad import *

from pycocotools.coco import COCO
from .refer.refer import REFER


class Pedes(Dataset):
    def __init__(self, image_set, root_path, data_path, boxes='gt', proposal_source='official',
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, parts=1, number_sep=1, part_methods='VS', **kwargs):
        """
        RefCOCO+ Dataset

        :param image_set: image folder name
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to dataset
        :param boxes: boxes to use, 'gt' or 'proposal'
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(Pedes, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        self.pedes_annot_files = {
            "trainval": "trainval.json",
        }

        self.vg_proposal = ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome")
        self.proposal_source = proposal_source
        self.boxes = boxes
        self.test_mode = test_mode

        self.data_path = data_path
        self.root_path = root_path
        self.transform = transform
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        # self.coco = COCO(annotation_file=os.path.join(data_path, coco_annot_files['train2014']))
        # self.refer = REFER(data_path, dataset='refcoco+', splitBy='unc')
        # self.refer_ids = []
        # for iset in self.image_sets:
        #     self.refer_ids.extend(self.refer.getRefIds(split=iset))
        # self.refs = self.refer.loadRefs(ref_ids=self.refer_ids)

        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        self.trainval_id_to_cls = {}
        self.image_nums = 0
        self.imgid2entry = {}
        self.ps_map = {}
        self.imgid2psid = {}
        self.trainval_index_to_id = {}
        f = open(os.path.join(self.data_path, self.pedes_annot_files['trainval']))
        self.setting = json.load(f)
        self.database = self.load_annotations()
        # if self.aspect_grouping:
        #     self.group_ids = self.group_aspect(self.database)
        self.part = parts
        self.max_word = 50

        self.val_images = []
        self.val_boxes = []
        self.val_im_info = []
        self.val_ids = []
        self.val_feat = []
        self.diff = 7

        self.use_JPP = False
        if part_methods == 'KS': self.use_JPP = True

        self.number_sep = number_sep
        self.number_parts = self.number_sep * self.part - self.number_sep + 1

        if self.use_JPP:
            f_box = open(os.path.join(self.data_path, 'result.json')) #box_frcnn.json
            self.JPP_boxes = json.load(f_box)
    def data_names(self):
        if self.test_mode:
            return ['image', 'boxes', 'im_info', 'expression']
        else:
            return ['image', 'boxes', 'im_info', 'expression', 'label']

    def __getitem__(self, index):
        idb = self.database[index]
        flipped = False
        # label_id = torch.zeros(2,3)
        image_b = []
        boxes_b = []
        im_info_b = []
        exp_ids_b = []
        label_id_b = []

        # expression
        exp_tokens = idb['caption']
        exp_retokens = self.tokenizer.tokenize(' '.join(exp_tokens))
        if flipped:
            exp_retokens = self.flip_tokens(exp_retokens, verbose=True)
        exp_ids = self.tokenizer.convert_tokens_to_ids(exp_retokens)
        if self.image_sets[0] == 'test':
            max_expression_length = self.max_word  # max([len(exp_ids), len(exp_ids2), len(exp_ids3)])
            exp_ids = clip_pad_1d(exp_ids, max_expression_length, pad=0)
            exp_ids_b = exp_ids.unsqueeze(0)
            return exp_ids_b, idb['id']


        image = self._load_image(idb['image_id'])
        im_info = torch.as_tensor([image.width, image.height, 1.0, 1.0])
        # boxes
        if self.use_JPP:
            path_org = idb['image_id']
            path_key = path_org.split('/')[-2] +'/' + path_org.split('/')[-1]
            boxes = self.JPP_boxes[path_key].copy()
            height_change = image.height / self.part * 1.0
            for i in range(self.number_parts):
                boxes.append(
                    [0, height_change * i / self.number_sep, image.width, height_change * (i / self.number_sep + 1)])
            boxes = torch.as_tensor(boxes).float()

        else:
            boxes = []
            height_change = image.height / self.part * 1.0
            for i in range(self.number_parts):
                boxes.append(
                    [0, height_change * i / self.number_sep, image.width, height_change * (i / self.number_sep + 1)])
            boxes = torch.as_tensor(boxes)
        if self.add_image_as_a_box:
            w0, h0 = im_info[0], im_info[1]
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)
        if self.transform is not None:
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)
            if not self.test_mode:
                boxes = boxes[1:]
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        image_b.append(image)
        boxes_b.append(boxes)
        im_info_b.append(im_info)
        exp_ids = clip_pad_1d(exp_ids, self.max_word, pad=0)
        exp_ids_b.append(exp_ids)
        id_ps = idb['id']
        label_id_b.append(self.trainval_id_to_cls[id_ps])#torch.tensor(cls_id).long()
        # negative samples.
        # 1: correct one, 2: random caption wrong, 3: random image wrong.
        for caption_wrong in range(self.diff):
            idb2 = random.choice(self.database)
            choose_id2 = idb2['id']
            while id_ps == choose_id2:
                idb2 = random.choice(self.database)
                choose_id2 = idb2['id']
            image2 = image.clone()
            im_info2 = im_info.clone()
            boxes2 = boxes.clone()
            # expression
            exp_tokens2 = idb2['caption']
            exp_retokens2 = self.tokenizer.tokenize(' '.join(exp_tokens2))
            if flipped:
                exp_retokens2 = self.flip_tokens(exp_retokens2, verbose=True)
            exp_ids2 = self.tokenizer.convert_tokens_to_ids(exp_retokens2)
            image_b.append(image2)
            boxes_b.append(boxes2)
            im_info_b.append(im_info2)
            exp_ids2 = clip_pad_1d(exp_ids2, self.max_word, pad=0)
            exp_ids_b.append(exp_ids2)
            label_id_b.append(torch.tensor(-1))

        for caption_wrong in range(self.diff):
            idb3 = random.choice(self.database)
            choose_id3 = idb3['id']
            while id_ps == choose_id3:
                idb3 = random.choice(self.database)
                choose_id3 = idb3['id']
            image3 = self._load_image(idb3['image_id'])
            im_info3 = torch.as_tensor([image3.width, image3.height, 1.0, 1.0])

            # boxes
            if self.use_JPP:
                path_org = idb3['image_id']
                path_key = path_org.split('/')[-2] +'/' + path_org.split('/')[-1]
                boxes3 = self.JPP_boxes[path_key].copy()
                height_change3 = image3.height / self.part * 1.0
                for i in range(self.number_parts):
                    boxes3.append([0, height_change3 * i / self.number_sep, image3.width,
                                   height_change3 * (i / self.number_sep + 1)])
                boxes3 = torch.as_tensor(boxes3).float()

            else:
                boxes3 = []
                height_change3 = image3.height / self.part * 1.0
                for i in range(self.number_parts):
                    boxes3.append([0, height_change3 * i / self.number_sep, image3.width,
                                   height_change3 * (i / self.number_sep + 1)])
                boxes3 = torch.as_tensor(boxes3)
            if self.add_image_as_a_box:
                w03, h03 = im_info3[0], im_info3[1]
                image_box3 = torch.as_tensor([[0.0, 0.0, w03 - 1, h03 - 1]])
                boxes3 = torch.cat((image_box3, boxes3), dim=0)
            if self.transform is not None:
                image3, boxes3, _, im_info3, flipped = self.transform(image3, boxes3, None, im_info3, flipped)
                if not self.test_mode:
                    boxes3 = boxes3[1:]
            w3 = im_info3[0].item()
            h3 = im_info3[1].item()
            boxes3[:, [0, 2]] = boxes3[:, [0, 2]].clamp(min=0, max=w3 - 1)
            boxes3[:, [1, 3]] = boxes3[:, [1, 3]].clamp(min=0, max=h3 - 1)

            exp_retokens3 = exp_retokens
            if flipped:
                exp_retokens3 = self.flip_tokens(exp_retokens3, verbose=True)
            exp_ids3 = self.tokenizer.convert_tokens_to_ids(exp_retokens3)
            exp_ids3 = clip_pad_1d(exp_ids3, self.max_word, pad=0)
            image_b.append(image3)
            boxes_b.append(boxes3)
            im_info_b.append(im_info3)
            exp_ids3 = clip_pad_1d(exp_ids3, self.max_word, pad=0)
            exp_ids_b.append(exp_ids3)
            label_id_b.append(torch.tensor(-1))

        image_b = torch.stack(image_b, dim=0)
        boxes_b = torch.stack(boxes_b, dim=0)
        im_info_b = torch.stack(im_info_b, dim=0)
        exp_ids_b = torch.stack(exp_ids_b, dim=0)
        label_id_b = torch.stack(label_id_b, dim=0)

        return image_b, boxes_b, im_info_b, exp_ids_b, label_id_b#self.trainval_id_to_cls[idb['id']]

    @staticmethod
    def flip_tokens(tokens, verbose=True):
        changed = False
        tokens_new = [tok for tok in tokens]
        for i, tok in enumerate(tokens):
            if tok == 'left':
                tokens_new[i] = 'right'
                changed = True
            elif tok == 'right':
                tokens_new[i] = 'left'
                changed = True
        if verbose and changed:
            logging.info('[Tokens Flip] {} -> {}'.format(tokens, tokens_new))
        return tokens_new

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def load_annotations(self):
        tic = time.time()
        entries = []
        imgid2psid = {}
        count = 0

        if self.image_sets[0] == 'test':
            split_value = 'val'
        else:
            split_value = self.image_sets[0]

        # image to ps id
        index_i = 0
        for i, annotation_ps in enumerate(self.setting):
            if annotation_ps['split'] == split_value:
                self.trainval_index_to_id[index_i] = annotation_ps['id']
                index_i += 1
                if annotation_ps['id'] in self.ps_map:
                    self.ps_map[annotation_ps['id']].append(i)
                else:
                    self.ps_map[annotation_ps['id']] = []
                    self.ps_map[annotation_ps['id']].append(i)
        # for cls, id_each in enumerate(self.ps_map):
        #     self.trainval_id_to_cls[id_each] = cls
        cls_id = 0
        if self.image_sets[0] == 'train':
            self.setting = self.setting[:34054]
        if self.image_sets[0] == 'val':
            self.setting = self.setting[34054:37132]
        if self.image_sets[0] == 'test':
            self.setting = self.setting[37132:]
            
        for annotation in self.setting:#[34054:37132]: #[:1000]
            if annotation['split'] != '':# split_value:
                self.image_nums += 1
                image_id = annotation['file_path']
                imgid2psid[image_id] = annotation['id']

                self.imgid2entry[image_id] = []

                if split_value == 'train':
                    for sentences in annotation['captions']:
                        for i in self.ps_map[annotation['id']]:
                            annotation_sameid = self.setting[i]
                            entries.append({"caption": sentences.split(), 'image_id': self.data_path + "/imgs/" + annotation_sameid['file_path'], 'id':annotation['id']})
                else:
                    image_id = annotation['file_path']
                    for sentences in annotation['captions']:
                        entries.append({"caption": sentences.split(), 'image_id': self.data_path + "/imgs/" + image_id, 'id':annotation['id']})
                        count += 1
                if annotation['id'] not in self.trainval_id_to_cls:
                    self.trainval_id_to_cls[annotation['id']] = torch.tensor(cls_id).long() #lihui
                    cls_id += 1

        return entries

        database = []
        for ref_id, ref in zip(self.refer_ids, self.refs):
            gt_x, gt_y, gt_w, gt_h = self.refer.getRefBox(ref_id=ref_id)
            image_fn = os.path.join(self.data_path, iset, 'COCO_{}_{:012d}.jpg'.format(iset, ref['image_id']))
            for sent in ref['sentences']:
                idb = {
                    'sent_id': sent['sent_id'],
                    'ann_id': ref['ann_id'],
                    'ref_id': ref['ref_id'],
                    'image_id': ref['image_id'],
                    'image_fn': image_fn,
                    'width': self.coco.imgs[ref['image_id']]['width'],
                    'height': self.coco.imgs[ref['image_id']]['height'],
                    'raw': sent['raw'],
                    'sent': sent['sent'],
                    'tokens': sent['tokens'],
                    'category_id': ref['category_id'],
                    'gt_box': [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h] if not self.test_mode else None
                }
                database.append(idb)

        print('Done (t={:.2f}s)'.format(time.time() - tic))

        # cache database via cPickle
        if self.cache_db:
            print('caching database to {}...'.format(db_cache_path))
            tic = time.time()
            if not os.path.exists(db_cache_root):
                makedirsExist(db_cache_root)
            with open(db_cache_path, 'wb') as f:
                cPickle.dump(database, f)
            print('Done (t={:.2f}s)'.format(time.time() - tic))

        return database

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        return Image.open(path).convert('RGB')

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def get_image_features(self, Fastrcnn_ext):
        for annotation in tqdm(self.setting):#[34054:37132]):#37132:]):
            if annotation['split'] != '':
                path_image = self.data_path + "/imgs/" + annotation['file_path']
                image = self._load_image(path_image)
                im_info = torch.as_tensor([image.width, image.height, 1.0, 1.0])
                flipped = False
                # boxes
                if self.use_JPP:
                    path_org = annotation['file_path'] #idb['image_id']
                    path_key = path_org.split('/')[-2] +'/' + path_org.split('/')[-1]
                    boxes = self.JPP_boxes[path_key]
                    height_change = image.height / self.part * 1.0
                    for i in range(self.number_parts):
                        boxes.append([0, height_change / self.number_sep * i, image.width,
                                      height_change * (i / self.number_sep + 1)])
                    boxes = torch.as_tensor(boxes).float()
                else:
                    boxes = []
                    height_change = image.height / self.part * 1.0
                    for i in range(self.number_parts):
                        boxes.append([0, height_change / self.number_sep * i, image.width,
                                      height_change * (i / self.number_sep + 1)])
                    boxes = torch.as_tensor(boxes)

                if self.add_image_as_a_box:
                    w0, h0 = im_info[0], im_info[1]
                    image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
                    boxes = torch.cat((image_box, boxes), dim=0)
                if self.transform is not None:
                    image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)
                w = im_info[0].item()
                h = im_info[1].item()
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

                image_b = image.unsqueeze(0)
                boxes_b = boxes.unsqueeze(0)
                im_info_b = im_info.unsqueeze(0)

                box_mask = (boxes_b[:, :, 0] > - 1.5)
                obj_reps = Fastrcnn_ext(images=image_b.cuda(),
                                                        boxes=boxes_b.cuda(),
                                                        box_mask=box_mask.cuda(),
                                                        im_info=im_info_b.cuda(),
                                                        classes=None,
                                                        segms=None)
                self.val_feat.append(obj_reps['obj_reps'])

                #self.val_images.append(image_b)
                self.val_boxes.append(boxes_b.cuda())
                self.val_im_info.append(im_info_b.cuda())
                self.val_ids.append(annotation['id'])
