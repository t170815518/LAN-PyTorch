import argparse
import logging
import time
import datetime
import os
import torch
from utils.data_helper import DataSet
from utils.link_prediction import run_link_prediction
from model.framework import LAN

logger = logging.getLogger()


def main():
    config = parse_arguments()
    run_training(config)


def parse_arguments():
    """ Parses arguments from CLI. """
    parser = argparse.ArgumentParser(description="Configuration for LAN model")
    parser.add_argument('--data_dir', '-D', type=str, default="data/FB15k-237")
    parser.add_argument('--save_dir', '-S', type=str, default="data/FB15k-237")
    # model
    parser.add_argument('--use_relation', type=int, default=1)
    parser.add_argument('--embedding_dim', '-e', type=int, default=100)
    parser.add_argument('--max_neighbor', type=int, default=64)
    parser.add_argument('--n_neg', '-n', type=int, default=1)
    parser.add_argument('--aggregate_type', type=str, default='attention')
    parser.add_argument('--score_function', type=str, default='TransE')
    parser.add_argument('--loss_function', type=str, default='margin')
    parser.add_argument('--margin', type=float, default='1.0')
    parser.add_argument('--corrupt_mode', type=str, default='both')
    # training
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--evaluate_size', type=int, default=250)
    parser.add_argument('--steps_per_display', type=int, default=100)
    parser.add_argument('--epoch_per_checkpoint', type=int, default=50)
    # gpu option
    parser.add_argument('--gpu_fraction', type=float, default=0.2)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--allow_soft_placement', type=bool, default=False)

    return parser.parse_args()


def run_training(config):
    # set up GPU
    config.device = torch.device("cuda:0")

    set_up_logger(config)

    logger.info('args: {}'.format(config))

    # prepare data
    logger.info("Loading data...")
    dataset = DataSet(config, logger)
    logger.info("Loading finish...")

    model = LAN(config, dataset.num_training_entity, dataset.num_relation)
    save_path = os.path.join(config.save_dir, "train_model.pt")
    model.to(config.device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # training
    num_batch = dataset.num_sample // config.batch_size
    logger.info('Train with {} batches'.format(num_batch))
    best_performance = float("inf")

    for epoch in range(config.num_epoch):
        st_epoch = time.time()
        loss_epoch = 0.
        cnt_batch = 0

        for batch_data in dataset.batch_iter_epoch(dataset.triplets_train, config.batch_size, config.n_neg):
            model.train()

            st_batch = time.time()

            batch_weight_ph, batch_weight_pt, batch_weight_nh, batch_weight_nt, batch_positive, batch_negative, \
                batch_relation_ph, batch_relation_pt, batch_relation_nh, batch_relation_nt, batch_neighbor_hp, \
                batch_neighbor_tp, batch_neighbor_hn, batch_neighbor_tn = batch_data
            feed_dict = {
                "neighbor_head_pos": batch_neighbor_hp,
                "neighbor_tail_pos": batch_neighbor_tp,
                "neighbor_head_neg": batch_neighbor_hn,
                "neighbor_tail_neg": batch_neighbor_tn,
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

            loss_batch = model.loss(feed_dict)

            cnt_batch += 1
            loss_epoch += loss_batch.item()

            loss_batch.backward()
            optim.step()
            model.zero_grad()

            en_batch = time.time()

            # print an overview every some batches
            if (cnt_batch + 1) % config.steps_per_display == 0 or (cnt_batch + 1) == num_batch:
                batch_info = 'epoch {}, batch {}, loss: {:.3f}, time: {:.3f}s'.format(epoch, cnt_batch, loss_batch,
                                                                                      en_batch - st_batch)
                print(batch_info)
                logger.info(batch_info)
        en_epoch = time.time()
        epoch_info = 'epoch {}, mean loss: {:.3f}, time: {:.3f}s'.format(epoch, loss_epoch / cnt_batch,
                                                                         en_epoch - st_epoch)
        print(epoch_info)
        logger.info(epoch_info)

        # evaluate the model every some steps
        if (epoch + 1) % config.epoch_per_checkpoint == 0 or (epoch + 1) == config.num_epoch:
            model.eval()
            st_test = time.time()
            with torch.no_grad():
                performance = run_link_prediction(config, model, dataset, epoch, logger, is_test=False)
            if performance < best_performance:
                best_performance = performance
                torch.save(model.state_dict(), save_path)
                time_str = datetime.datetime.now().isoformat()
                saved_message = '{}: model at epoch {} save in file {}'.format(time_str, epoch, save_path)
                print(saved_message)
                logger.info(saved_message)

            en_test = time.time()
            test_finish_message = 'testing finished with time: {:.3f}s'.format(en_test - st_test)
            print(test_finish_message)
            logger.info(test_finish_message)

    finished_message = 'Training finished'
    print(finished_message)
    logger.info(finished_message)


def set_up_logger(config):
    checkpoint_dir = config.save_dir
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(checkpoint_dir + 'train.log', 'w+')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


if __name__ == '__main__':
    main()
