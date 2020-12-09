# LAN-PyTorch
Implementation of Logic Attention Network in PyTorch (currently without inductive test)

**Notes**:
1. This repo currently does not support inductive testing, e.g. aux and dev data in the original repo are not 
needed and removed. 
1. This repo currently contains only Attention aggregator and TransE score. 
1. L2 regularization is temporarily removed. 

The model must be trained with GPU. 
## Data requirement 
Dataset should be put under `data` directory, and it should contain files including `train`, `test`, `entity2id.txt` and 
`relation2id.txt`. To translate the data in index format, see `utils.preprocess` (the function will overwritten the original files!)
## Evaluation Methods 
Filter out the known entities (i.e., the ones in train set)