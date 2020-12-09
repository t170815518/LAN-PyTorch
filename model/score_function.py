import torch
import torch.nn.functional as F


class TransE(torch.nn.Module):
    def __init__(self):
        super(TransE, self).__init__()

    def __call__(self, head, tail, relation):
        # normalization
        head = F.normalize(head, p=2, dim=1)
        relation = F.normalize(relation, p=2, dim=1)
        tail = F.normalize(tail, p=2, dim=1)
        dissimilarity = torch.sum(torch.abs(head + relation - tail), dim=1)
        score = -dissimilarity
        return score
