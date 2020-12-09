import argparse
import logging
import torch
from utils.data_helper import DataSet

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
    parser.add_argument('--epoch_per_checkpoint', type=int, default=1)
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