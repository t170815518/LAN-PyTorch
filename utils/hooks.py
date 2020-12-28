import numpy as np


batch_count = 0


def save_weights(module, input, output):
    output, logic_attention, nn_attention = output

    global batch_count
    batch_count = batch_count + 1

    if module.is_save_attention:
        logic_attention = logic_attention.cpu().numpy().reshape((-1, module.max_neighbor))
        nn_attention = nn_attention.detach().cpu().numpy().reshape((-1, module.max_neighbor))
        # save weights to txt file
        np.savetxt("logic_attention_batch{}.txt".format(batch_count), logic_attention)
        np.savetxt("nn_attention_batch{}.txt".format(batch_count), nn_attention)
    return output  # only output relevant tensors
