import numpy as np

def compute_topk(sim_cosine, target, k=[1,5,10], reverse=False):
    result = []
    # query = query / query.norm(dim=1,keepdim=True)
    # gallery = gallery / gallery.norm(dim=1,keepdim=True)
    # sim_cosine = torch.matmul(query, gallery.t())
    result.extend(topk(sim_cosine, target, k=[1,5,10]))
    # if reverse:
    #     result.extend(topk(sim_cosine, target_query, target_gallery, k=[1,5,10], dim=0))
    return result


def topk(sim, target, k=[1,5,10], dim=1):
    result = []
    maxk = max(k)
    size_total = len(target)
    # _, pred_index = sim.topk(maxk, dim, True, True)
    correct = np.zeros(target.shape)
    pred_index = np.argsort(-sim,axis=1)[:,:maxk]# np.argsort(-sim)[:maxk]
    for i in range(size_total):
        correct[i][:maxk] = target[i][pred_index[i]]
    # if dim == 1:
    #     pred_labels = pred_labels.t()
    # correct = pred_labels.eq(target.view(1,-1).expand_as(pred_labels))

    for topk in k:
        #correct_k = torch.sum(correct[:topk]).float()
        correct_k = np.sum(correct[:,:topk], axis=1)
        correct_k = np.sum(correct_k > 0)
        result.append(correct_k * 100 / size_total)
    return result


score1 = np.load("/home/lihui/scores_JPP.npy") #scores_JPPL_14 JPP6_L_18
score2 = np.load("/home/lihui/scores_2*part_l.npy") #scores_original_s scores_2*part_s
score3 = np.load("/home/lihui/JPP6_L_18.npy")

score4 = np.load("/home/lihui/scores_2*part_s.npy")

scorex = np.load("/home/lihui/scores_11JPP.npy")

target_matrix = np.load("/home/lihui/target.npy") #JPP6_L_18 scores_JPP
# score = (score2 + scorex) / 2.0 
score = (score1 + score2 + score3 + score4) / 4.0
r1, r5, r10 = compute_topk(scorex, target_matrix)
print(" r1, r5, r10: ", r1, r5, r10)
