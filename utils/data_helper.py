import os
import random
import numpy as np
from collections import defaultdict


class DataSet:
    def __init__(self, args, logger):
        self.data_dir = args.data_dir
        self.max_neighbor = args.max_neighbor
        self.corrupt_mode = args.corrupt_mode
        self.load_data(logger)

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
            graph[tail].append([relation+cnt_relation, head, 0.])

        cnt_train = len(graph)

        self.count_imply(graph, cnt_relation)

        graph = self.process_graph(graph)
        cnt_all = len(graph)

        return triplet_train, graph, cnt_relation, cnt_entity

    def count_imply(self, graph, cnt_relation):
        co_relation = np.zeros((cnt_relation*2+1, cnt_relation*2+1), dtype=np.float32)
        freq_relation = defaultdict(int)

        for entity in graph:
            relation_list = list(set([neighbor[0] for neighbor in graph[entity]]))
            for n_i in range(len(relation_list)):
                r_i = relation_list[n_i]
                freq_relation[r_i] += 1
                for n_j in range(n_i+1, len(relation_list)):
                    r_j = relation_list[n_j]
                    co_relation[r_i][r_j] += 1
                    co_relation[r_j][r_i] += 1

        for r_i in range(cnt_relation*2):
            co_relation[r_i] = (co_relation[r_i] * 1.0) / freq_relation[r_i]
        self.co_relation = co_relation.transpose()
        for r_i in range(cnt_relation*2):
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

    def batch_iter_epoch(self, data, batch_size, num_negative=1, corrupt=True, shuffle=True):
        """ Returns prepared information in np.ndarray to feed into the model.
        TODO: add checking to avoid negative samples become correct
        """
        data_size = len(data)
        if data_size % batch_size == 0:
            num_batches_per_epoch = int(data_size/batch_size)
        else:
            num_batches_per_epoch = int(data_size/batch_size) + 1

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
            neighbor_head_pos = self.graph_train[batch_positive[:, 0]] #[:, :, 0:2]
            neighbor_tail_pos = self.graph_train[batch_positive[:, 2]] #[:, :, 0:2]
            batch_relation_ph = np.asarray(batch_positive[:, 1])
            batch_relation_pt = batch_relation_ph + self.num_relation
            neighbor_imply_ph = self.weight_graph[batch_positive[:, 0]].reshape(-1, self.max_neighbor, 1)
            neighbor_imply_pt = self.weight_graph[batch_positive[:, 2]].reshape(-1, self.max_neighbor, 1)
            query_weight_ph = self.co_relation[batch_relation_ph]
            query_weight_pt = self.co_relation[batch_relation_pt]
            batch_weight_ph = query_weight_ph[np.arange(real_batch_num).repeat(self.max_neighbor),
                                              neighbor_head_pos[:, :, 0].reshape(-1)].reshape(real_batch_num, self.max_neighbor, 1)
            batch_weight_pt = query_weight_pt[np.arange(real_batch_num).repeat(self.max_neighbor),
                                              neighbor_tail_pos[:, :, 0].reshape(-1)].reshape(real_batch_num, self.max_neighbor, 1)
            batch_weight_ph = np.concatenate((batch_weight_ph, neighbor_imply_ph), axis=2)
            batch_weight_pt = np.concatenate((batch_weight_pt, neighbor_imply_pt), axis=2)

            if corrupt:
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
                batch_weight_nh = query_weight_nh[np.arange(real_batch_num).repeat(self.max_neighbor), neighbor_head_neg[:, :, 0].reshape(-1)].reshape(real_batch_num, self.max_neighbor, 1)
                batch_weight_nt = query_weight_nt[np.arange(real_batch_num).repeat(self.max_neighbor), neighbor_tail_neg[:, :, 0].reshape(-1)].reshape(real_batch_num, self.max_neighbor, 1)
                batch_weight_nh = np.concatenate((batch_weight_nh, neighbor_imply_nh), axis=2)
                batch_weight_nt = np.concatenate((batch_weight_nt, neighbor_imply_nt), axis=2)
                yield [batch_weight_ph, batch_weight_pt, batch_weight_nh, batch_weight_nt,
                    batch_positive, batch_negative, batch_relation_ph, batch_relation_pt, batch_relation_nh, batch_relation_nt, neighbor_head_pos, neighbor_tail_pos, neighbor_head_neg, neighbor_tail_neg]
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

        if 'head' in self.predict_mode: # and self.corrupt_mode == 'partial':
            return np.asarray(batch_predict_tail)
        elif 'tail' in self.predict_mode: # and self.corrupt_mode == 'partial':
            return np.asarray(batch_predict_head)
        else:
            return np.asarray(batch_predict_tail), np.asarray(batch_predict_head)

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
