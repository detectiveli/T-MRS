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

from scipy.io import loadmat

class PA100K(Dataset):
    def __init__(self, image_set, root_path, data_path, boxes='gt', proposal_source='official',
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, **kwargs):
        """
        Market1501 Dataset

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
        super(PA100K, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'
        self.vg_proposal = ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome")
        self.proposal_source = proposal_source
        self.boxes = boxes
        self.test_mode = test_mode

        self.data_path = data_path
        self.root_path = root_path
        self.transform = transform
        self.image_sets = [iset.strip() for iset in image_set.split('+')]

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
        # self.imgid2entry = {}
        self.ps_map = {}
        self.imgid2psid = {}
        self.trainval_index_to_id = {}

        self.image_entries = []
        self.pa100k_attribute = self.generate_data_description()
        self.database = self.load_annotations(self.pa100k_attribute)
        # if self.aspect_grouping:
        #     self.group_ids = self.group_aspect(self.database)
        self.part = 7
        self.max_boxes = 7

        self.max_word = 26

        self.val_images = []
        self.val_boxes = []
        self.val_im_info = []
        self.val_ids = []
        self.val_feat = []

        self.diff = 2


    @property
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
        mlm_labels_b = []

        # expression
        exp_tokens = idb['caption']
        exp_retokens = self.tokenizer.tokenize(' '.join(exp_tokens))
        if flipped:
            exp_retokens = self.flip_tokens(exp_retokens, verbose=True)
        exp_ids = self.tokenizer.convert_tokens_to_ids(exp_retokens)
        if self.image_sets[0] == 'test':
            exp_ids = clip_pad_1d(exp_ids, self.max_word, pad=0)
            exp_ids_b = exp_ids.unsqueeze(0)
            return exp_ids_b, idb['id']


        image = self._load_image(idb['image_id'])
        im_info = torch.as_tensor([image.width, image.height, 1.0, 1.0])

        boxes = []
        height_change = image.height / self.part * 1.0
        for i in range(self.part):
            boxes.append([0, height_change * i, image.width, height_change * (i+1)])
        boxes = torch.as_tensor(boxes)

        if self.add_image_as_a_box:
            w0, h0 = im_info[0], im_info[1]
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            if len(boxes) == 0:
                boxes = image_box
            else:
                boxes = torch.cat((image_box, boxes), dim=0)

        boxes = clip_pad_boxes(boxes, self.max_boxes, pad=-2)

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

        for caption_wrong in range(self.diff*9):
            idb3 = random.choice(self.database)
            choose_id3 = idb3['id']
            while id_ps == choose_id3:
                idb3 = random.choice(self.database)
                choose_id3 = idb3['id']
            image3 = self._load_image(idb3['image_id'])
            im_info3 = torch.as_tensor([image3.width, image3.height, 1.0, 1.0])

            # boxes
            boxes3 = []

            height_change3 = image3.height / self.part * 1.0
            for i in range(self.part):
                boxes3.append([0, height_change3 * i, image3.width, height_change3 * (i + 1)])
            boxes3 = torch.as_tensor(boxes3)

            if self.add_image_as_a_box:
                w03, h03 = im_info3[0], im_info3[1]
                image_box3 = torch.as_tensor([[0.0, 0.0, w03 - 1, h03 - 1]])
                if len(boxes3) == 0:
                    boxes3 = image_box3
                else:
                    boxes3 = torch.cat((image_box3, boxes3), dim=0)
            boxes3 = clip_pad_boxes(boxes3, self.max_boxes, pad=-2)

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

        return image_b, boxes_b, im_info_b, exp_ids_b, label_id_b

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

    def load_annotations(self, pa100k_attribute):
        entries = []

        cls_id = 0
        if self.image_sets[0] == 'test':
            for counter in range(10000):
                each_data = counter + 90000
                img_path = pa100k_attribute['image'][each_data]
                id = pa100k_attribute['att'][each_data]
                sentence = np.array(pa100k_attribute['att_name'])[np.array(id) > 0]

                id = self.listtwo_to_ten(id)
                if id not in self.trainval_id_to_cls:
                    entries.append({"caption": list(sentence), 'image_id': pa100k_attribute['root'] + img_path,
                                    'id': cls_id})
                    self.trainval_id_to_cls[id] = torch.tensor(cls_id).long()  # lihui
                    cls_id += 1
                    self.image_entries.append({"caption": list(sentence), 'image_id': pa100k_attribute['root'] + img_path,
                                    'id': self.trainval_id_to_cls[id]})  #
                else:
                    self.image_entries.append({"caption": list(sentence), 'image_id': pa100k_attribute['root'] + img_path,
                                    'id': self.trainval_id_to_cls[id]})  # entries
                self.image_nums += 1
        else: 
            for each_data in range(80000):
                img_path = pa100k_attribute['image'][each_data]
                id = pa100k_attribute['att'][each_data]
                sentence = np.array(pa100k_attribute['att_name'])[np.array(id) > 0]

                id = self.listtwo_to_ten(id)
                entries.append({"caption": list(sentence), 'image_id': pa100k_attribute['root'] + img_path,
                                'id': id})

                if id not in self.trainval_id_to_cls:
                    self.trainval_id_to_cls[id] = torch.tensor(cls_id).long()  # lihui
                    cls_id += 1

        return entries

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
        for annotation in tqdm(self.image_entries):

            path_image = annotation['image_id']
            image = self._load_image(path_image)
            im_info = torch.as_tensor([image.width, image.height, 1.0, 1.0])
            flipped = False
            # boxes

            boxes = []
            height_change = image.height / self.part * 1.0
            for i in range(self.part):
                boxes.append([0, height_change * i, image.width, height_change * (i + 1)])
            boxes = torch.as_tensor(boxes)

            if self.add_image_as_a_box:
                w0, h0 = im_info[0], im_info[1]
                image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
                if len(boxes) == 0:
                    boxes = image_box
                else:
                    boxes = torch.cat((image_box, boxes), dim=0)

            boxes = clip_pad_boxes(boxes, self.max_boxes, pad=-2)

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
            self.val_feat.append(obj_reps['obj_reps'].cuda())

            #self.val_images.append(image_b)
            self.val_boxes.append(boxes_b.cuda())
            self.val_im_info.append(im_info_b.cuda())
            self.val_ids.append(annotation['id'])

    def generate_data_description(self):
        """
        create a dataset description file, which consists of images, labels
        self.data_path = data_path
        self.root_path = root_path
        self.transform = transform
        """
        dataset = dict()
        dataset['description'] = 'pa100k'
        dataset['root'] = self.data_path + '/release_data/release_data/'
        dataset['image'] = []
        dataset['att'] = []
        dataset['att_name'] = []
        dataset['selected_attribute'] = range(26)
        # load ANNOTATION.MAT
        data = loadmat(self.data_path + '/annotation.mat')
        for idx in range(26):
            dataset['att_name'].append(data['attributes'][idx][0][0])

        for idx in range(80000):
            dataset['image'].append(data['train_images_name'][idx][0][0])
            dataset['att'].append(data['train_label'][idx, :].tolist())

        for idx in range(10000):
            dataset['image'].append(data['val_images_name'][idx][0][0])
            dataset['att'].append(data['val_label'][idx, :].tolist())

        for idx in range(10000):
            dataset['image'].append(data['test_images_name'][idx][0][0])
            dataset['att'].append(data['test_label'][idx, :].tolist())

        return dataset

    def listtwo_to_ten(self, input_two):
        answer = ""
        for i, num in enumerate(input_two):
            answer += str(num)

        return answer

