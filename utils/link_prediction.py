import random
import torch


def run_link_prediction(args, model, dataset, epoch, logger, is_test=False):
    logger.info('evaluating the current model...')
    rank_head = 0
    rank_tail = 0
    hit10_head = 0
    hit10_tail = 0
    hit3_head = 0
    hit3_tail = 0
    max_rank_head = 0
    max_rank_tail = 0
    min_rank_head = None
    min_rank_tail = None
    acc_head = 0
    acc_tail = 0
    rec_rank_head = 0
    rec_rank_tail = 0

    if is_test:
        evaluate_size = len(dataset.triplets_test)
        evaluate_data = dataset.triplets_test.tolist()
    else:
        if args.evaluate_size == 0:
            evaluate_size = len(dataset.triplets_dev)
        else:
            evaluate_size = args.evaluate_size
        evaluate_data = dataset.triplets_test.tolist()

    cnt_sample = 0
    for triplet in random.sample(evaluate_data, evaluate_size):
        # using head predict tail, and using tail to predict head
        sample_predict_head, sample_predict_tail = dataset.next_sample_eval(triplet, is_test=is_test)

        def eval_by_batch(data_eval):
            prediction_all = []
            for batch_eval in dataset.batch_iter_epoch(data_eval, 4096, corrupt=False, shuffle=False):
                batch_weight_ph, batch_weight_pt, batch_triplet, batch_relation_tail, batch_neighbor_head, \
                    batch_neighbor_tail = batch_eval
                # batch_triplet, batch_relation_tail, batch_neighbor_head, batch_neighbor_tail = batch_eval
                batch_relation_head = batch_triplet[:, 1]
                feed_dict = {
                        "neighbor_head_pos": batch_neighbor_head,
                        "neighbor_tail_pos": batch_neighbor_tail,
                        "input_relation_ph": batch_relation_head,
                        "input_relation_pt": batch_relation_tail,
                        "neighbor_weight_ph": batch_weight_ph,
                        "neighbor_weight_pt": batch_weight_pt,
                }
                prediction_batch = model.get_positive_score(feed_dict)
                prediction_all.append(prediction_batch)

            return torch.cat(prediction_all, dim=0)

        prediction_head = eval_by_batch(sample_predict_head)
        prediction_tail = eval_by_batch(sample_predict_tail)

        rank_head_current = int((-prediction_head).argsort().argmin() + 1)
        rank_tail_current = int((-prediction_tail).argsort().argmin() + 1)

        rank_head += rank_head_current
        rec_rank_head += 1.0 / rank_head_current
        if rank_head_current <= 10:
            hit10_head += 1
        if rank_head_current <= 3:
            hit3_head += 1
        if max_rank_head < rank_head_current:
            max_rank_head = rank_head_current
        if min_rank_head == None:
            min_rank_head = rank_head_current
        elif min_rank_head > rank_head_current:
            min_rank_head = rank_head_current
        if rank_head_current == 1:
            acc_head += 1

        rank_tail += rank_tail_current
        rec_rank_tail += 1.0 / rank_tail_current
        if rank_tail_current <= 10:
            hit10_tail += 1
        if rank_tail_current <= 3:
            hit3_tail += 1
        if max_rank_tail < rank_tail_current:
            max_rank_tail = rank_tail_current
        if min_rank_tail == None:
            min_rank_tail = rank_tail_current
        elif min_rank_tail > rank_tail_current:
            min_rank_tail = rank_tail_current
        if rank_tail_current == 1:
            acc_tail += 1

    rank_head_mean = rank_head // evaluate_size
    hit10_head = hit10_head * 1.0 / evaluate_size
    hit3_head = hit3_head * 1.0 / evaluate_size
    acc_head = acc_head * 1.0 / evaluate_size
    rec_rank_head = rec_rank_head / evaluate_size

    rank_tail_mean = rank_tail // evaluate_size
    hit10_tail = hit10_tail * 1.0 / evaluate_size
    hit3_tail = hit3_tail * 1.0 / evaluate_size
    acc_tail = acc_tail * 1.0 / evaluate_size
    rec_rank_tail = rec_rank_tail / evaluate_size

    performance_info_head = '[head] epoch {} MR: {:d}, MRR: {:.3f}, hit@10: {:.3f}%, hit@3: {:.3f}%, hit@1: {:.3f}%'.format(epoch,
                                                                                                                rank_head_mean,
                                                                                                                rec_rank_head,
                                                                                                                hit10_head * 100,
                                                                                                                hit3_head * 100,
                                                                                                                acc_head * 100)
    performance_info_tail = '[tail] epoch {} MR: {:d}, MRR: {:.3f}, hit@10: {:.3f}%, hit@3: {:.3f}%, hit@1: {:.3f}%'.format(epoch,
                                                                                                                rank_tail_mean,
                                                                                                                rec_rank_tail,
                                                                                                                hit10_tail * 100,
                                                                                                                hit3_tail * 100,
                                                                                                                acc_tail * 100)

    print(performance_info_head)
    logger.info(performance_info_head)
    print(performance_info_tail)
    logger.info(performance_info_tail)

    return min_rank_head
