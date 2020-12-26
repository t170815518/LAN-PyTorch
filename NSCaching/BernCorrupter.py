from collections import defaultdict
import torch


class BernCorrupter:
    def __init__(self, data, n_ent, n_rel):
        self.bern_prob = self.get_bern_prob(data, n_ent, n_rel)
        self.n_ent = n_ent

    def get_bern_prob(self, data, n_ent, n_rel):
        edges = defaultdict(lambda: defaultdict(lambda: set()))
        rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
        for h, r, t in data:
            edges[r][h].add(t)
            rev_edges[r][t].add(h)
        bern_prob = torch.zeros(n_rel)
        for k in edges.keys():
            right = sum(len(tails) for tails in edges[k].values()) / len(edges[k])
            left = sum(len(heads) for heads in rev_edges[k].values()) / len(rev_edges[k])
            bern_prob[k] = right / (right + left)

        return bern_prob
