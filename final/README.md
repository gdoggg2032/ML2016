# Factorization Machines


This repo contains implementation of Factorization Machine, Field-aware Factorization Machine and some variants.

  - Element-wise Framework
    - Factorization Machine
    - Weighted Factorization Machine
  - List-wise Framework  
    - Factorization Machine Neuron Network
    - Factorization Machine with Numeric Features
    - Field-aware Factorization Machine

The Dataset is [Outbrain Click Prediction][outbrain] from Kaggle Competition.
Our Team name is NTU_R05922027_1丁讚讚讚.
# Usage
#### Feature Extraction
```
python feature_extract_final.py --num_features [num_features]
```
- `num_features`: the parameter to extract top `num_features` features which are mentioned in Report.pdf.
- `feature_extract_final.py` will print out the hash_size(`max_features`) and write it to file `./feature_num`. 

#### modeling
```
python lwfm.py --max_features [max_features] --mode [mode]
```
- `max_features`: the hash_size
- `mode`: `0` for training with validation set, if you don't use validation set, you should create a fake one. `1` for testing set prediction.

[outbrain]: <https://www.kaggle.com/c/outbrain-click-prediction>





