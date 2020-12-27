import os
import random
import logging
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger()


class DataSet:
    def __init__(self, args, logger):
        self.data_dir = args.data_dir
        self.max_neighbor = args.max_neighbor
        self.corrupt_mode = args.corrupt_mode
        self.n_sample = args.n_neg
        self.N_1 = args.N_1
        self.N_2 = args.N_2
        self.load_data(logger)

        # for NSCaching
        self.head_cache = defaultdict(lambda: set())
        self.tail_cache = defaultdict(lambda: set())
        self.head_cache_tensor = {}
        self.tail_cache_tensor = {}
        self.corrupter = None
        self.model = None
        self.cache = None
        self.head_cache_id = None
        self.tail_cache_id = None
        self.head_all = None
        self.tail_all = None

    def load_data(self, logger):
        train_path = os.path.join(self.data_dir, 'train')
        test_path = os.path.join(self.data_dir, 'test')
        relation_path = os.path.join(self.data_dir, 'relation2id.txt')
        entity_idx_path = os.path.join(self.data_dir, 'entity2id.txt')

        triplets_train, graph_train, self.num_relation, self.num_entity = \
            self.doc_to_tensor_graph(train_path)

        self.r_PAD = self.num_relation * 2
        self.e_PAD = self.num_training_entity
        self.num_sample = len(triplets_train)

        try:
            self.i2r = self.build_relation_dict(relation_path, self.r_PAD, self.num_relation)
        except:
            self.i2r = None
        try:
            self.i2e = self.build_entity_dict(entity_idx_path)
        except:
            self.i2e = None

        logger.info('got {} entities for training'.format(self.num_training_entity))
        logger.info('got {} relations for training'.format(self.num_relation))

        self.graph_train, self.weight_graph = self.sample_neighbor(graph_train)

        triplets_test = self.doc_to_tensor(test_path)

        self.task = 'link_prediction'
        # construct answer poor for filter results
        self.triplets_train_pool = set(triplets_train)
        self.triplets_true_pool = set(triplets_train + triplets_test)
        self.predict_mode = ""

        self.triplets_train = np.asarray(triplets_train)
        self.triplets_test = np.asarray(triplets_test)

        logger.info('got {} triplets for train'.format(len(self.triplets_train)))
        logger.info('got {} triplets for test'.format(len(self.triplets_test)))

    def doc_to_tensor_graph(self, data_path_train):
        triplet_train = []
        graph = defaultdict(list)
        train_entity = {}
        cnt_entity = 0
        cnt_relation = 0

        cnt_entity, cnt_relation = self.__read_train_file(cnt_entity, cnt_relation, data_path_train, train_entity,
                                                          triplet_train)

        self.num_training_entity = cnt_entity

        for triplet in triplet_train:
            head, relation, tail = triplet
            # hpt, tph = self.relation_dist[relation]
            graph[head].append([relation, tail, 0.])
            graph[tail].append([relation + cnt_relation, head, 0.])

        cnt_train = len(graph)

        self.count_imply(graph, cnt_relation)

        graph = self.process_graph(graph)
        cnt_all = len(graph)

        return triplet_train, graph, cnt_relation, cnt_entity

    def count_imply(self, graph, cnt_relation):
        co_relation = np.zeros((cnt_relation * 2 + 1, cnt_relation * 2 + 1), dtype=np.float32)
        freq_relation = defaultdict(int)

        for entity in graph:
            relation_list = list(set([neighbor[0] for neighbor in graph[entity]]))
            for n_i in range(len(relation_list)):
                r_i = relation_list[n_i]
                freq_relation[r_i] += 1
                for n_j in range(n_i + 1, len(relation_list)):
                    r_j = relation_list[n_j]
                    co_relation[r_i][r_j] += 1
                    co_relation[r_j][r_i] += 1

        for r_i in range(cnt_relation * 2):
            co_relation[r_i] = (co_relation[r_i] * 1.0) / freq_relation[r_i]
        self.co_relation = co_relation.transpose()
        for r_i in range(cnt_relation * 2):
            co_relation[r_i][r_i] = co_relation[r_i].mean()
        print('finish calculating co relation')

    def process_graph(self, graph):
        """ Adds the denominator of logic attention to the graph"""
        for entity in graph:
            relation_list = defaultdict(int)
            for neighbor in graph[entity]:
                relation_list[neighbor[0]] += 1
            if len(relation_list) == 1:
                continue
            for rel_i in relation_list:
                other_relation_list = [rel for rel in relation_list if rel != rel_i]
                imply_i = self.co_relation[rel_i]
                j_imply_i = imply_i[other_relation_list].max()
                for _idx, neighbor in enumerate(graph[entity]):
                    if neighbor[0] == rel_i:
                        graph[entity][_idx][2] = j_imply_i
        print('finish processing graph')
        return graph

    def build_relation_dict(self, data_path, pad, cnt):
        i2n = {}
        with open(data_path, 'r') as fr:
            for line in fr:
                line = line.strip().split('\t')
                # name = '/'.join(line[0].split('/')[-2:])
                name = line[0]
                idx = int(line[1])
                # if idx >= cnt:
                #     continue
                i2n[idx] = name
                i2n[idx + cnt] = name + "_reverse"
        i2n[pad] = 'PADDING_RELATION'
        return i2n

    def build_entity_dict(self, data_path):
        i2n = {}
        with open(data_path, 'r') as fr:
            for line in fr:
                line = line.strip().split('\t')
                name = line[0]
                idx = int(line[1])
                i2n[idx] = name
        i2n[self.e_PAD] = 'PADDING_ENTITY'
        return i2n

    def sample_neighbor(self, graph):
        """ TODO: add the "random" sample """
        sample_graph = np.ones((self.num_entity, self.max_neighbor, 2), dtype=np.int64)
        weight_graph = np.ones((self.num_entity, self.max_neighbor), dtype=np.float32)
        # initialize with all entries padded
        sample_graph[:, :, 0] *= self.r_PAD
        sample_graph[:, :, 1] *= self.e_PAD

        cnt = 0
        for entity in graph:
            num_neighbor = len(graph[entity])
            cnt += num_neighbor
            num_sample = min(num_neighbor, self.max_neighbor)
            # sample_id = random.sample(range(len(graph[entity])), num_sample)
            sample_id = range(len(graph[entity]))[:num_sample]
            # sample_graph[entity][:num_sample] = np.asarray(graph[entity])[sample_id]
            sample_graph[entity][:num_sample] = np.asarray(graph[entity])[sample_id][:, 0:2]
            weight_graph[entity][:num_sample] = np.asarray(graph[entity])[sample_id][:, 2]

        return sample_graph, weight_graph

    def doc_to_tensor(self, data_path):
        """ Prepares list of triplet for training. """
        triplet_tensor = []
        with open(data_path, 'r') as fr:
            for line in fr:
                line = line.strip().split('\t')
                line = [int(_id) for _id in line]
                if line[0] >= self.num_entity or line[2] >= self.num_entity:
                    continue
                if line[1] >= self.num_relation:
                    continue
                if len(line) == 4:
                    head, relation, tail, label = line
                    if label != 1:
                        label = -1
                    triplet_tensor.append((head, relation, tail, label))
                else:
                    head, relation, tail = line
                    triplet_tensor.append((head, relation, tail))
        return triplet_tensor

    def batch_iter_epoch(self, data, batch_size, num_negative=1, corrupt=True, shuffle=True, is_use_cache=False):
        """ Returns prepared information in np.ndarray to feed into the model.
        """
        data_size = len(data)
        if data_size % batch_size == 0:
            num_batches_per_epoch = int(data_size / batch_size)
        else:
            num_batches_per_epoch = int(data_size / batch_size) + 1

        # Shuffle the data at each epoch
        if shuffle:
            shuffled_indices = np.random.permutation(np.arange(data_size))
        else:
            shuffled_indices = np.arange(data_size)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            real_batch_num = end_index - start_index
            batch_indices = shuffled_indices[start_index:end_index]
            batch_positive = data[batch_indices]
            try:
                h_idx = self.head_cache_id[batch_indices]
                t_idx = self.tail_cache_id[batch_indices]
            except TypeError:  # when cache is not used 
                pass
            neighbor_head_pos = self.graph_train[batch_positive[:, 0]]  # [:, :, 0:2]
            neighbor_tail_pos = self.graph_train[batch_positive[:, 2]]  # [:, :, 0:2]
            batch_relation_ph = np.asarray(batch_positive[:, 1])
            batch_relation_pt = batch_relation_ph + self.num_relation
            neighbor_imply_ph = self.weight_graph[batch_positive[:, 0]].reshape(-1, self.max_neighbor, 1)
            neighbor_imply_pt = self.weight_graph[batch_positive[:, 2]].reshape(-1, self.max_neighbor, 1)
            query_weight_ph = self.co_relation[batch_relation_ph]
            query_weight_pt = self.co_relation[batch_relation_pt]
            batch_weight_ph = query_weight_ph[np.arange(real_batch_num).repeat(self.max_neighbor),
                                              neighbor_head_pos[:, :, 0].reshape(-1)].reshape(real_batch_num,
                                                                                              self.max_neighbor, 1)
            batch_weight_pt = query_weight_pt[np.arange(real_batch_num).repeat(self.max_neighbor),
                                              neighbor_tail_pos[:, :, 0].reshape(-1)].reshape(real_batch_num,
                                                                                              self.max_neighbor, 1)
            batch_weight_ph = np.concatenate((batch_weight_ph, neighbor_imply_ph), axis=2)
            batch_weight_pt = np.concatenate((batch_weight_pt, neighbor_imply_pt), axis=2)

            if corrupt:
                if is_use_cache:
                    h_rand, t_rand = self.negative_sampling(batch_positive, h_idx, t_idx)
                    prob = self.corrupter.bern_prob[batch_positive[:, 1]]
                    selection = torch.bernoulli(prob).type(torch.ByteTensor)

                    n_h = torch.from_numpy(batch_positive[:, 0]).cuda()
                    n_t = torch.from_numpy(batch_positive[:, 2]).cuda()
                    n_r = torch.from_numpy(batch_positive[:, 1]).cuda()

                    if n_h.size() != h_rand.size():
                        n_h = n_h.unsqueeze(1).expand_as(h_rand)
                        n_t = n_t.unsqueeze(1).expand_as(h_rand)
                        n_r = n_r.unsqueeze(1).expand_as(h_rand)
                        h = h.unsqueeze(1)
                        r = r.unsqueeze(1)
                        t = t.unsqueeze(1)

                    n_h[selection] = h_rand[selection]
                    n_t[~selection] = t_rand[~selection]

                    n_h = n_h.cpu().numpy().tolist()
                    n_t = n_t.cpu().numpy().tolist()
                    n_r = n_r.cpu().numpy().tolist()
                    batch_negative = [(h, r, t) for h, r, t in zip(n_h, n_r, n_t)]

                    self.update_cache(batch_positive, h_idx, t_idx)
                else:
                    batch_negative = []
                    for triplet in batch_positive:
                        id_head_corrupted = triplet[0]
                        id_tail_corrupted = triplet[2]
                        id_relation = triplet[1]

                        for n_neg in range(num_negative):
                            if self.corrupt_mode == 'both':
                                head_prob = np.random.binomial(1, 0.5)
                                if head_prob:
                                    id_head_corrupted = random.sample(range(self.num_training_entity), 1)[0]
                                else:
                                    id_tail_corrupted = random.sample(range(self.num_training_entity), 1)[0]
                            else:
                                if 'tail' in self.predict_mode:
                                    id_head_corrupted = random.sample(range(self.num_training_entity), 1)[0]
                                elif 'head' in self.predict_mode:
                                    id_tail_corrupted = random.sample(range(self.num_training_entity), 1)[0]
                            batch_negative.append([id_head_corrupted, triplet[1], id_tail_corrupted])

                batch_negative = np.asarray(batch_negative)
                neighbor_head_neg = self.graph_train[batch_negative[:, 0]]
                neighbor_tail_neg = self.graph_train[batch_negative[:, 2]]
                neighbor_imply_nh = self.weight_graph[batch_negative[:, 0]].reshape(-1, self.max_neighbor, 1)
                neighbor_imply_nt = self.weight_graph[batch_negative[:, 2]].reshape(-1, self.max_neighbor, 1)

                batch_relation_nh = batch_negative[:, 1]
                batch_relation_nt = batch_relation_nh + self.num_relation
                query_weight_nh = self.co_relation[batch_relation_nh]
                query_weight_nt = self.co_relation[batch_relation_nt]
                batch_weight_nh = query_weight_nh[
                    np.arange(real_batch_num).repeat(self.max_neighbor), neighbor_head_neg[:, :, 0].reshape(
                        -1)].reshape(real_batch_num, self.max_neighbor, 1)
                batch_weight_nt = query_weight_nt[
                    np.arange(real_batch_num).repeat(self.max_neighbor), neighbor_tail_neg[:, :, 0].reshape(
                        -1)].reshape(real_batch_num, self.max_neighbor, 1)
                batch_weight_nh = np.concatenate((batch_weight_nh, neighbor_imply_nh), axis=2)
                batch_weight_nt = np.concatenate((batch_weight_nt, neighbor_imply_nt), axis=2)
                feed_dict = {
                    "neighbor_head_pos": neighbor_head_pos,
                    "neighbor_tail_pos": neighbor_tail_pos,
                    "neighbor_head_neg": neighbor_head_neg,
                    "neighbor_tail_neg": neighbor_tail_neg,
                    "input_relation_ph": batch_relation_ph,
                    "input_relation_pt": batch_relation_pt,
                    "input_relation_nh": batch_relation_nh,
                    "input_relation_nt": batch_relation_nt,
                    "input_triplet_pos": batch_positive,
                    "input_triplet_neg": batch_negative,
                    "neighbor_weight_ph": batch_weight_ph,
                    "neighbor_weight_pt": batch_weight_pt,
                    "neighbor_weight_nh": batch_weight_nh,
                    "neighbor_weight_nt": batch_weight_nt
                }
                yield feed_dict
            else:
                yield [batch_weight_ph, batch_weight_pt,
                       batch_positive, batch_relation_pt, neighbor_head_pos, neighbor_tail_pos]

    def next_sample_eval(self, triplet_evaluate, is_test):
        if is_test:
            answer_pool = self.triplets_true_pool
        else:
            answer_pool = self.triplets_train_pool

        # construct two batches for head and tail prediction
        batch_predict_head = [triplet_evaluate]
        # replacing head
        id_heads_corrupted_list = range(self.num_training_entity)
        id_heads_corrupted_set = set(id_heads_corrupted_list)
        id_heads_corrupted_set.discard(triplet_evaluate[0])  # remove the golden head
        for head in id_heads_corrupted_list:
            if (head, triplet_evaluate[1], triplet_evaluate[2]) in answer_pool:
                id_heads_corrupted_set.discard(head)
        # apart from the test case (0-th element), add triplets formed with other entities
        batch_predict_head.extend([(head, triplet_evaluate[1], triplet_evaluate[2]) for head in id_heads_corrupted_set])

        batch_predict_tail = [triplet_evaluate]
        # replacing tail
        # id_tails_corrupted = set(random.sample(range(self.num_entity), 1000))
        id_tails_corrupted_list = range(self.num_training_entity)
        id_tails_corrupted_set = set(id_tails_corrupted_list)
        id_tails_corrupted_set.discard(triplet_evaluate[2])  # remove the golden tail
        for tail in id_tails_corrupted_list:
            if (triplet_evaluate[0], triplet_evaluate[1], tail) in answer_pool:
                id_tails_corrupted_set.discard(tail)
        batch_predict_tail.extend([(triplet_evaluate[0], triplet_evaluate[1], tail) for tail in id_tails_corrupted_set])

        if 'head' in self.predict_mode:  # and self.corrupt_mode == 'partial':
            return np.asarray(batch_predict_tail)
        elif 'tail' in self.predict_mode:  # and self.corrupt_mode == 'partial':
            return np.asarray(batch_predict_head)
        else:
            return np.asarray(batch_predict_tail), np.asarray(batch_predict_head)

    def prepare_forward(self, entity_id):
        """
        Return triplets that include (entity_id, query_rel), where query_rel is every relation in KG.
        :param entity_id: int the entity id
        """
        batch = []
        batch_size = self.num_relation * 2
        for i in range(batch_size):
            batch.append((entity_id, i, 0))
        yield self.batch_iter_epoch(batch, batch_size=batch_size, num_negative=0, corrupt=False, shuffle=False)

    # def initialize_cache(self):
    #     for h, r, t in self.triplets_train:
    #         self.tail_cache[(h, r)].add(t)
    #         self.head_cache[(r, t)].add(h)
    #
    #     for k in self.tail_cache.keys():
    #         self.tail_cache_tensor[k] = torch.sparse.FloatTensor(torch.LongTensor([list(self.tail_cache[k])]),
    #                                                             torch.ones(len(self.tail_cache[k])), torch.Size([self.num_entity]))
    #     for k in self.head_cache.keys():
    #         self.head_cache_tensor[k] = torch.sparse.FloatTensor(torch.LongTensor([list(self.head_cache[k])]),
    #                                                torch.ones(len(self.head_cache[k])), torch.Size([self.num_entity]))
    #     logger.info("Head cache index size = {}\Tail cache index size = {}".format(len(self.head_cache), len(self.tail_cache)))
    #     return self.head_cache_tensor, self.tail_cache_tensor

    def get_cache(self):
        head_cache = {}
        tail_cache = {}
        head_all = []
        tail_all = []
        head_cache_id = []
        tail_cache_id = []
        count_h = 0
        count_t = 0

        for h, r, t in self.triplets_train:
            if not (t, r) in self.head_cache:
                head_cache[(t, r)] = count_h
                head_all.append([h])
                count_h += 1
            else:
                head_all[head_cache[(t, r)]].append(h)

            if not (h, r) in tail_cache:
                tail_cache[(h, r)] = count_t
                tail_all.append([t])
                count_t += 1
            else:
                tail_all[tail_cache[(h, r)]].append(t)

            head_cache_id.append(head_cache[(t, r)])
            tail_cache_id.append(tail_cache[(h, r)])

        head_cache_id = np.array(head_cache_id, dtype=int)
        tail_cache_id = np.array(tail_cache_id, dtype=int)

        # initialize cache
        head_cache = np.random.randint(low=0, high=self.num_entity, size=(count_h, self.N_1))
        tail_cache = np.random.randint(low=0, high=self.num_entity, size=(count_t, self.N_1))

        self.head_cache = head_cache
        self.tail_cache = tail_cache
        self.head_cache_id = head_cache_id
        self.tail_cache_id = tail_cache_id
        self.head_all = head_all
        self.tail_all = tail_all

    def negative_sampling(self, data, head_idx, tail_idx, sample='basic'):
        randint = np.random.randint(low=0, high=self.N_1, size=(data.shape[0],))
        h_idx = self.head_cache[head_idx, randint]
        t_idx = self.tail_cache[tail_idx, randint]

        h_rand = torch.LongTensor(h_idx).cuda()
        t_rand = torch.LongTensor(t_idx).cuda()
        return h_rand, t_rand

    def update_cache(self, data, head_idx, tail_idx):
        head = torch.from_numpy(data[:, 0]).cuda(torch.device("cuda:0"))
        tail = torch.from_numpy(data[:, 2]).cuda(torch.device("cuda:0"))
        rela = torch.from_numpy(data[:, 1]).cuda(torch.device("cuda:0"))

        head_idx, head_uniq = np.unique(head_idx, return_index=True)
        tail_idx, tail_uniq = np.unique(tail_idx, return_index=True)

        tail_h = tail[head_uniq]
        rela_h = rela[head_uniq]
        rela_t = rela[tail_uniq]
        head_t = head[tail_uniq]

        # get candidate for updating the cache
        h_cache = self.head_cache[head_idx]
        t_cache = self.tail_cache[tail_idx]
        h_cand = np.concatenate([h_cache, np.random.choice(self.num_entity, (len(head_idx), self.N_2))], 1)
        t_cand = np.concatenate([t_cache, np.random.choice(self.num_entity, (len(tail_idx), self.N_2))], 1)
        h_cand = torch.from_numpy(h_cand).type(torch.LongTensor).cuda()
        t_cand = torch.from_numpy(t_cand).type(torch.LongTensor).cuda()

        # expand for computing scores/probs
        rela_h = rela_h.unsqueeze(1).expand(-1, self.N_1 + self.N_2)
        tail_h = tail_h.unsqueeze(1).expand(-1, self.N_1 + self.N_2)
        head_t = head_t.unsqueeze(1).expand(-1, self.N_1 + self.N_2)
        rela_t = rela_t.unsqueeze(1).expand(-1, self.N_1 + self.N_2)

        h_probs = self.model.prob(h_cand, rela_h, tail_h)
        t_probs = self.model.prob(head_t, rela_t, t_cand)

        # use IS to update the cache
        h_new = torch.multinomial(h_probs, self.N_1, replacement=False)
        t_new = torch.multinomial(t_probs, self.N_1, replacement=False)

        h_idx = torch.arange(0, len(head_idx)).type(torch.LongTensor).unsqueeze(1).expand(-1, self.N_1)
        t_idx = torch.arange(0, len(tail_idx)).type(torch.LongTensor).unsqueeze(1).expand(-1, self.N_1)
        h_rep = h_cand[h_idx, h_new]
        t_rep = t_cand[t_idx, t_new]

        self.head_cache[head_idx] = h_rep.cpu().numpy()
        self.tail_cache[tail_idx] = t_rep.cpu().numpy()

    def __read_train_file(self, cnt_entity, cnt_relation, data_path_train, train_entity, triplet_train):
        with open(data_path_train, 'r') as fr:
            for line in fr:
                line = line.strip().split('\t')
                line = [int(_id) for _id in line]
                assert len(line) == 3
                head, relation, tail = line
                triplet_train.append((head, relation, tail))
                # graph[head].append((relation, tail))
                # graph[tail].append((relation, head))
                train_entity[head] = 1
                train_entity[tail] = 1
                if head >= cnt_entity:
                    cnt_entity = head + 1
                if tail >= cnt_entity:
                    cnt_entity = tail + 1
                if relation >= cnt_relation:
                    cnt_relation = relation + 1
        return cnt_entity, cnt_relation

    # def __do_cluster(self, entity_emb_path):
    #     entity_embeddings = np.loadtxt(entity_emb_path)
    #     kmeans = KMeans(n_clusters=self.n_clusters).fit(entity_embeddings)
    #     labels_lst = kmeans.labels_.tolist()
    #
    #     labels = {}
    #     for entity_id, cluster_id in enumerate(labels_lst):
    #         labels[entity_id] = cluster_id
    #     return labels
