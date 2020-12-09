import torch
import torch.nn as nn
import torch.nn.functional as F
from model.aggregator import Attention
from model.score_function import TransE


class LAN(torch.nn.Module):
    def __init__(self, args, num_entity, num_relation):
        super(LAN, self).__init__()

        self.max_neighbor = args.max_neighbor
        self.embedding_dim = args.embedding_dim
        self.learning_rate = args.learning_rate
        self.aggregate_type = args.aggregate_type
        self.score_function = args.score_function
        self.loss_function = args.loss_function
        self.use_relation = args.use_relation
        self.margin = args.margin
        self.weight_decay = args.weight_decay

        self.num_entity = num_entity
        self.num_relation = num_relation

        # parameters
        self.entity_embedding = torch.nn.Embedding(self.num_entity + 1, self.embedding_dim)
        nn.init.xavier_normal_(self.entity_embedding.weight.data)
        # Below: without + 1 because the we dont embed dummy relation
        self.relation_embedding_out = torch.nn.Embedding(self.num_relation, self.embedding_dim)
        nn.init.xavier_normal_(self.relation_embedding_out.weight.data)

        self.encoder = Attention(self.num_relation, self.num_entity, self.embedding_dim)
        self.decoder = TransE()

    def loss(self, feed_dict):
        for key, value in feed_dict.items():
            feed_dict[key] = torch.from_numpy(value).to(device=torch.cuda.current_device())

        neighbor_weight_ph = feed_dict['neighbor_weight_ph']
        neighbor_weight_pt = feed_dict['neighbor_weight_pt']
        neighbor_weight_nh = feed_dict['neighbor_weight_nh']
        neighbor_weight_nt = feed_dict['neighbor_weight_nt']
        neighbor_head_pos = feed_dict['neighbor_head_pos']
        neighbor_head_neg = feed_dict['neighbor_head_neg']
        neighbor_tail_pos = feed_dict['neighbor_tail_pos']
        neighbor_tail_neg = feed_dict['neighbor_tail_neg']
        input_relation_ph = feed_dict['input_relation_ph']
        input_relation_pt = feed_dict['input_relation_pt']
        input_relation_nh = feed_dict['input_relation_nh']
        input_relation_nt = feed_dict['input_relation_nt']
        input_triplet_pos = feed_dict['input_triplet_pos']
        input_triplet_neg = feed_dict['input_triplet_neg']

        encoder = self.encoder
        decoder = self.decoder

        head_pos_embedded, _ = self.encode(encoder, neighbor_head_pos, input_relation_ph,
                                           neighbor_weight_ph)
        tail_pos_embedded, _ = self.encode(encoder, neighbor_tail_pos, input_relation_pt,
                                           neighbor_weight_pt)

        head_neg_embedded, _ = self.encode(encoder, neighbor_head_neg, input_relation_nh,
                                           neighbor_weight_nh)
        tail_neg_embedded, _ = self.encode(encoder, neighbor_tail_neg, input_relation_nt,
                                           neighbor_weight_nt)

        emb_relation_pos_out = self.relation_embedding_out(input_relation_ph)
        emb_relation_neg_out = self.relation_embedding_out(input_relation_nh)

        positive_score = self.decode(decoder, head_pos_embedded, tail_pos_embedded, emb_relation_pos_out)
        negative_score = self.decode(decoder, head_neg_embedded, tail_neg_embedded, emb_relation_neg_out)

        ph_origin_embedded = self.entity_embedding(input_triplet_pos[:, 0])
        pt_origin_embedded = self.entity_embedding(input_triplet_pos[:, 2])
        nh_origin_embedded = self.entity_embedding(input_triplet_neg[:, 0])
        nt_origin_embedded = self.entity_embedding(input_triplet_neg[:, 2])

        origin_positive_score = self.decode(decoder, ph_origin_embedded, pt_origin_embedded,
                                                   emb_relation_pos_out)
        origin_negative_score = self.decode(decoder, nh_origin_embedded, nt_origin_embedded,
                                                   emb_relation_neg_out)

        # margin loss
        loss = torch.mean(F.relu(self.margin - positive_score + negative_score))
        loss += torch.mean(F.relu(self.margin - origin_positive_score + origin_negative_score))

        return loss

    def encode(self, encoder, neighbor_ids, query_relation, weight):
        """ TODO: check neighbor_ids content """
        neighbor_embedded = self.entity_embedding(neighbor_ids[:, :, 1])
        if self.use_relation == 1:
            return encoder(neighbor_embedded, neighbor_ids, query_relation, weight)
        else:
            return encoder(neighbor_embedded, neighbor_ids[:, :, 0])

    def decode(self, decoder, head_embedded, tail_embedded, relation_embedded):
        score = decoder(head_embedded, tail_embedded, relation_embedded)
        return score

    def get_positive_score(self, feed_dict):
        for key, value in feed_dict.items():
            feed_dict[key] = torch.from_numpy(value).to(device=torch.cuda.current_device())

        neighbor_head_pos = feed_dict['neighbor_head_pos']
        neighbor_tail_pos = feed_dict['neighbor_tail_pos']
        input_relation_ph = feed_dict['input_relation_ph']
        input_relation_pt = feed_dict['input_relation_pt']
        neighbor_weight_ph = feed_dict['neighbor_weight_ph']
        neighbor_weight_pt = feed_dict['neighbor_weight_pt']

        head_pos_embedded, _ = self.encode(self.encoder, neighbor_head_pos, input_relation_ph,
                                           neighbor_weight_ph)
        tail_pos_embedded, _ = self.encode(self.encoder, neighbor_tail_pos, input_relation_pt,
                                           neighbor_weight_pt)

        emb_relation_pos_out = self.relation_embedding_out(input_relation_ph)
        return self.decode(self.decoder, head_pos_embedded, tail_pos_embedded, emb_relation_pos_out)
