""" This module contains aggregator (Attention). """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(torch.nn.Module):
    """ Attention aggregators.
    Attributes:
        - mlp_w: relation-specific transforming vector to apply on Tr(e)
        - query_relation_embedding: relation-specific z_q in nn-mechanism
        ...
    """
    def __init__(self, num_relation, num_entity, embedding_dim):
        super(Attention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_relation = num_relation
        self.num_entity = num_entity

        # parameters
        self.mlp_w = torch.nn.Embedding(self.num_entity * 2 + 1, self.embedding_dim)
        nn.init.xavier_normal_(self.mlp_w.weight.data)
        self.query_relation_embedding = torch.nn.Embedding(self.num_relation * 2, self.embedding_dim)
        nn.init.xavier_normal_(self.query_relation_embedding.weight.data)
        self.att_w = torch.nn.Parameter(torch.zeros(size=(self.embedding_dim * 2, self.embedding_dim * 2)))
        nn.init.xavier_normal_(self.att_w.data)
        self.att_v = torch.nn.Parameter(torch.zeros(size=(1, self.embedding_dim * 2)))
        nn.init.xavier_normal_(self.att_v.data)

        self.mask_emb = torch.cat([torch.ones([self.num_entity, 1]), torch.zeros([1, 1])], 0).\
            to(torch.cuda.current_device())
        self.mask_weight = torch.cat([torch.zeros([self.num_entity, 1]), torch.ones([1, 1])*1e19], 0).\
            to(torch.cuda.current_device())

    def forward(self, input, neighbor, query_relation_id, weight):
        input_shape = input.shape
        max_neighbors = input_shape[1]
        hidden_size = input_shape[2]

        input_relation = neighbor[:, :, 0]
        input_entity = neighbor[:, :, 1]

        transformed = self.mlp_w(input_relation)
        transformed = self._transform(input, transformed)

        mask = self.mask_emb[input_entity]
        transformed = transformed * mask

        query_relation = self.query_relation_embedding(query_relation_id)
        query_relation = query_relation.unsqueeze(1)
        query_relation = query_relation.expand(-1, max_neighbors, -1)

        attention_logit = self.mlp(query_relation, transformed, max_neighbors)
        mask_logit = self.mask_weight[input_entity]
        attention_logit = attention_logit - torch.reshape(mask_logit, [-1, max_neighbors])
        attention_weight = F.softmax(attention_logit, dim=1)
        attention_weight = attention_weight + weight[:, :, 0] / (weight[:, :, 1] + 1)

        attention_weight = torch.reshape(attention_weight, [-1, max_neighbors, 1])
        output = torch.sum(transformed * attention_weight, dim=1)
        attention_weight = torch.reshape(attention_weight, [-1, max_neighbors])
        return output, attention_weight

    def _transform(self, e, r):
        normed = F.normalize(r, p=2, dim=2)
        return e - torch.sum(e * normed, 2, keepdim=True) * normed

    def mlp(self, query, transformed, max_len):
        """ Neural network attention """
        hidden = torch.cat([query, transformed], dim=2)
        hidden = torch.reshape(hidden, [-1, self.embedding_dim * 2])
        hidden = torch.tanh(torch.matmul(hidden, self.att_w))
        hidden = torch.reshape(hidden, [-1, max_len, self.embedding_dim * 2])
        attention_logit = torch.sum(hidden * self.att_v, dim=2)
        return attention_logit
