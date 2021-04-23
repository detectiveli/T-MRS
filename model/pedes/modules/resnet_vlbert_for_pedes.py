import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertMVRCHeadTransform
from .OIM_loss import OIM_Module

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        transform = VisualLinguisticBertMVRCHeadTransform(config.NETWORK.VLBERT)
        # self.linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, 768) #331 1000 35 100 12003 lihui
        # self.OIM_loss = OIM_Module(331, 768)  # config.NETWORK.VLBERT.hidden_size)
        self.OIM_loss = OIM_Module(12003, 768)
        self.linear = nn.Sequential(
            # transform,
            nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
            nn.Linear(config.NETWORK.VLBERT.hidden_size, 768) #331 1000 35 100 12003 lihui
        )

        linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, 1)
        self.final_mlp = nn.Sequential(
            transform,
            nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
            linear
        )

        # init weights
        self.init_weight()

        self.fix_params()



        # self.embeddings_word = torch.nn.Conv1d(in_channels=40, out_channels=1, kernel_size=1)
        # self.embeddings_box = torch.nn.Conv1d(in_channels=6, out_channels=1, kernel_size=1)
        # self.line_cls = nn.utils.weight_norm(nn.Linear(config.NETWORK.VLBERT.hidden_size, 1000), name='weight') #12003

    def init_weight(self):
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      expression,
                      label,
                      ):
        ###########################################

        # visual feature extraction
        batch_size = image.size(0)
        num_options = image.size(1)
        image = image.view(-1, image.size(2), image.size(3), image.size(4))
        boxes = boxes.view(-1, boxes.size(2), boxes.size(3))
        #boxes = boxes
        im_info = im_info.view(-1, im_info.size(2))
        expression = expression.view(-1, expression.size(2))

        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        # max_len = int(box_mask.sum(1).max().item())
        # origin_len = boxes.shape[1]
        # box_mask = box_mask[:, :max_len]
        # boxes = boxes[:, :max_len]
        # label = label[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        _, pooled_output = self.vlbert(text_input_ids,
                                                                 text_token_type_ids,
                                                                 text_visual_embeddings,
                                                                 text_mask,
                                                                 object_vl_embeddings,
                                                                 box_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_text_and_object_separately=False)

        ###########################################
        outputs = {}



        # classifier
        #logits = self.final_mlp(pooled_output)
        ''' 
        logits = self.linear(pooled_output)
        # vil_logit = logits.view(batch_size, num_options)
        score_OIM = self.OIM_loss(logits, label.view(-1))
        loss_c = nn.CrossEntropyLoss(ignore_index=-1)
        cmpc_loss = loss_c(F.softmax(score_OIM, dim=1)*10, label.view(-1))# + criterion(text_logits, label_text)
        # cmpc_loss = loss_c(logits, label.view(-1))
        cls_pred = torch.argmax(score_OIM, dim=1)
        cls_precision = torch.mean((cls_pred[label.view(-1) != -1] == label.view(-1)[label.view(-1) != -1]).float())
        return cls_precision, cmpc_loss
	'''

        # loss
        logits = self.final_mlp(pooled_output)
        vil_logit = logits.view(batch_size, num_options)
        loss = nn.CrossEntropyLoss(ignore_index=-1)
        cls_loss = loss(vil_logit, torch.zeros(batch_size).long().cuda())
        _, preds = torch.max(vil_logit, 1)
        batch_score = float((preds == torch.zeros(batch_size).long().cuda()).sum()) / float(batch_size)


        return batch_score, cls_loss

    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          expression,
                          label,
                          feat = None):

        ###########################################

        # visual feature extraction
        batch_size = boxes.size(0)
        num_options = boxes.size(1)

        if feat is None:
            image = image.view(-1, image.size(2), image.size(3), image.size(4))
            boxes = boxes.view(-1, boxes.size(2), boxes.size(3))
            im_info = im_info.view(-1, im_info.size(2))

            images = image
            box_mask = (boxes[:, :, 0] > - 1.5)
            # max_len = int(box_mask.sum(1).max().item())
            # origin_len = boxes.shape[1]
            # box_mask = box_mask[:, :max_len]
            # boxes = boxes[:, :max_len]

            obj_reps = self.image_feature_extractor(images=images,
                                                    boxes=boxes,
                                                    box_mask=box_mask,
                                                    im_info=im_info,
                                                    classes=None,
                                                    segms=None)
        else:
            boxes = boxes.view(-1, boxes.size(2), boxes.size(3))
            box_mask = (boxes[:, :, 0] > - 1.5)
            # obj_reps = feat
        ############################################
        # prepare text
        expression = expression.view(-1, expression.size(2))
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        if feat is None:
            text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))
            #text_visual_embeddings = feat[:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

            object_linguistic_embeddings = self.object_linguistic_embeddings(
                boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
       	    )
            object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)
            #object_vl_embeddings = torch.cat((feat, object_linguistic_embeddings), -1)
        else:
            # text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))
            text_visual_embeddings = feat[:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

            object_linguistic_embeddings = self.object_linguistic_embeddings(
                boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
       	    )
            # object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)
       	    object_vl_embeddings = torch.cat((feat, object_linguistic_embeddings), -1)


        ###########################################

        # Visual Linguistic BERT

        encoded_layers, pooled_output, att = self.vlbert(text_input_ids,
                                                                 text_token_type_ids,
                                                                 text_visual_embeddings,
                                                                 text_mask,
                                                                 object_vl_embeddings,
                                                                 box_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_text_and_object_separately=False,
                                                                    output_attention_probs=True)

        ###########################################
        outputs = {}

        # classifier
        logits = self.final_mlp(pooled_output)#.squeeze(-1)

        # loss
        vil_logit = logits.view(batch_size, num_options)
        _, preds = torch.max(vil_logit, 1)

        return att, logits

    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        criterion = nn.CrossEntropyLoss()
        # labels_onehot = one_hot_coding(labels, self.num_classes).float()
        # image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        # text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        # image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        # text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

        image_logits = image_embeddings #self.line_cls(image_embeddings)
        text_logits = text_embeddings #self.line_cls(text_embeddings)

        label_img = labels[:, 1, :].contiguous().view(-1)
        label_text = labels[:, 0, :].contiguous().view(-1)

        cmpc_loss = criterion(image_logits, label_img)# + criterion(text_logits, label_text)
        # cmpc_loss = - (F.log_softmax(image_logits, dim=1) + F.log_softmax(text_logits, dim=1)) * labels_onehot
        # cmpc_loss = torch.mean(torch.sum(cmpc_loss, dim=1))
        # classification accuracy for observation
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == label_img).float())
        text_precision = torch.mean((text_pred == label_text).float())

        return cmpc_loss, image_precision, text_precision
