from collections import namedtuple
import torch
from common.trainer import to_cuda


@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch):
    return
    net.eval()
    answer = 0
    # metrics.reset()
    for nbatch, batch in enumerate(val_loader):
        batch = to_cuda(batch)
        # label = batch[label_index_in_batch]
        # datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]

        score, sim = net(*batch)
        answer += score
        # outputs.update({'label': label})
        # metrics.update(outputs)
    if len(val_loader) == 0:
        len_b = 1
    else:
        len_b = len(val_loader)
    answer = answer / len_b
    print("batch score: ", answer)
