""" This module is to evaluate the model with complete test dataset and export them to txt file. """
import torch
import logging
from types import SimpleNamespace
from utils.data_helper import DataSet
from utils.link_prediction import run_link_prediction
from model.framework import LAN


logger = logging.getLogger()


def evaluate(model_type, model_path, config):
    set_up_logger()
    # set up GPU
    config.device = torch.device("cuda:0")
    # load the dataset
    dataset = DataSet(config, logger)

    model = model_type(config, dataset.num_training_entity, dataset.num_relation)
    model.load_state_dict(torch.load(model_path))
    model.to(config.device)

    # evaluation
    model.eval()
    with torch.no_grad():
        run_link_prediction(config, model, dataset, 0, logger, is_test=True)

    print("Evaluation completes.")


def set_up_logger():
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("complete_evaluation.log", 'w+')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


if __name__ == '__main__':
    # simple config information, which should match with the train config
    config = SimpleNamespace()
    config.data_dir = "data/FB15k-237"
    config.max_neighbor = 64
    config.use_relation = 1
    config.margin = 1.0
    config.embedding_dim = 100
    config.corrupt_mode = "both"
    config.evaluate_size = 0
    config.N_1 = 60
    config.N_2 = 30
    config.n_neg = 1
    config.learning_rate = 0.001
    config.aggregate_type = "LAN"
    config.score_function = "TransE"
    config.loss_function = "TransE"
    config.weight_decay = 0

    evaluate(LAN, "data/FB15k-237/train_model.pt", config)

