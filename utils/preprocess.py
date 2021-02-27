import os
from random import sample


def preprocess(file_path):
    entity2id = {}
    rel2id = {}
    entity_cnt = 0
    rel_cnt = 0

    for file_name in ["train.txt", "test.txt"]:
        translated_data = []
        with open(os.path.join(file_path, file_name), "r") as f:
            for line in f:
                e1, rel, e2 = line.strip().split("\t")
                if e1 not in entity2id.keys():
                    entity2id[e1] = entity_cnt
                    entity_cnt += 1
                if e2 not in entity2id.keys():
                    entity2id[e2] = entity_cnt
                    entity_cnt += 1
                if rel not in rel2id.keys():
                    rel2id[rel] = rel_cnt
                    rel_cnt += 1
                e1_id = entity2id[e1]
                e2_id = entity2id[e2]
                rel_id = rel2id[rel]
                translated_data.append((e1_id, rel_id, e2_id))

        with open(os.path.join(file_path, file_name), "w") as f:
            for e1, rel, e2 in translated_data:
                f.write("{}\t{}\t{}\n".format(e1, rel, e2))

    for d, file_name in zip([entity2id, rel2id], ["entity2id.txt", "relation2id.txt"]):
        with open(os.path.join(file_path, file_name), "w+") as f:
            for name, index in d.items():
                f.write("{}\t{}\n".format(name, index))


def sample_data(data_path, frac=0.5):
    with open(os.path.join(data_path, "train"), "r") as f:
        lines = f.readlines()
    lines = sample(lines, round(len(lines) * frac))
    dir = "{}_{}".format(data_path, frac)
    os.mkdir(dir)
    with open(os.path.join(dir, "train"), "w+") as f:
        f.writelines(lines)


if __name__ == '__main__':
    # preprocess("data/FB15k-237")
    sample_data("data/FB15k-237")
