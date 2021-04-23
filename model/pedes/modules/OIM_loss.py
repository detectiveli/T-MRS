from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

class _OIM_Module(Function):
    def __init__(self, LUT, QUEUE, num_classes, num_buffer):
        super(_OIM_Module, self).__init__()
        self.LUT = LUT
        self.QUEUE = QUEUE
        self.momentum = 0.5  # TODO: use exponentially weighted average
        self.num_classes = num_classes
        self.num_buffer = num_buffer
    # @staticmethod
    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        # for i, feat_each, each_person in zip(range(10), inputs, targets): #each person from proposal
        #     if (each_person < 0):
        #         self.QUEUE[:, :] = torch.cat((self.QUEUE[1:, :], feat_each.view(1, -1)), 0)
        outputs_labeled = inputs.mm(self.LUT.t())
        outputs_unlabeled = inputs.mm(self.QUEUE.t())
        return torch.cat((outputs_labeled, outputs_unlabeled), 1)
    # @staticmethod
    def backward(self, grad_outputs):
        # grad_outputs, = grad_outputs
        # print('aaaaaaaaaa')
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((self.LUT, self.QUEUE), 0))
        for feat_each, each_person in zip(inputs, targets): #each person from proposal
            # print(each_person)
            if (each_person > 0) & (each_person < self.num_classes):
                self.LUT[each_person, :] = self.momentum*self.LUT[each_person, :] + (1 - self.momentum)*feat_each
            else:
                self.QUEUE[:, :] = torch.cat((self.QUEUE[1:, :], feat_each.view(1, -1)), 0)


        # print(self.QUEUE[-1])
        return grad_inputs, None

class OIM_Module(nn.Module):
    def __init__(self, num_class, num_feature):
        super(OIM_Module, self).__init__()
        # self.updated = 0
        self.num_feature = num_feature
        self.num_classes = num_class
        self.num_buffer = self.num_classes #cfg.MODEL.ROI_REID_HEAD.NUM_BUFFER
        # self.LUT = torch.zeros(self.num_classes, self.num_feature).cuda()
        # self.QUEUE = torch.zeros(self.num_buffer, self.num_feature).cuda()

        self.register_buffer('LUT', torch.zeros(
            self.num_classes, self.num_feature).cuda())
        self.register_buffer('QUEUE', torch.zeros(
            self.num_buffer, self.num_feature).cuda())
        self.momentum_ = 0.5

    def forward(self, x, person_id):
        labed_score = _OIM_Module(self.LUT, self.QUEUE, self.num_classes, self.num_buffer)(x, person_id)
        return labed_score
