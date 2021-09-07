# DataSelection
Data selection<br />
<br />

## Installation:

Cloning the repository to the local disk
```
git submodule update --init
git config --global user.name "First and Last Name"
git config --global user.email "xxx@email.com"
```

Setting up the python environment
```
conda update -n base -c defaults conda
python create_environment.py
conda activate DataSelection
pip install -e .
```

Linters:

Please make sure that `Autopep8` and `Flake8` are activated in your IDE (e.g. VS-Code or Pycharm)<br />
<br />

## Datasets
### CIFAR10H
###### About
The CIFAR10H dataset is the CIFAR10 test set but all the samples have been labelled by multiple annotators.
We use the CIFAR training set for validation.
###### How to use it
The dataset will automatically be downloaded to your machine when you run code for the first time
on your machine with the [cifar10h dataset class](DataSelection/datasets/cifar10h.py) 
(or corresponding [PL module](DataSelection/deep_learning/self_supervised/cifar10h_datamodule.py)).

### Chest X-ray datasets
#### Full Kaggle Pneumonia Detection challenge dataset
###### About
For our experiments, in particular for unsupervised pretraining we use the full Kaggle training set (stage 1) from the
[Pneumonia Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). The dataset class for this dataset
can be found in the [kaggle_cxr.py](DataSelection/datasets/kaggle_cxr.py) file. This dataset class loads the full 
set with binary labels based on the bounding boxes provided for the competition.
#### Noisy subset of Kaggle dataset for data selection
The images released as part of the Kaggle Challenge, where originally released as part of the NIH chest x-ray datasets. 
Before starting the competition, 30k images have been selected as the images for competitions. The labels for these images
have then been adjudicated to label them with bounding boxes indicating "pneumonia-life opacities". In order to evaluate 
our data selection model on medical dataset, we have sampled a small subset of the Kaggle dataset (4000 samples, balanced) 
for which we have access to the original labels provided in the NIH dataset. This dataset uses the kaggle dataset with noisy labels
as the original labels from RSNA and the clean labels are the Kaggle labels. Originally the dataset had 14 classes, we 
created a new binary label to label each image as "pneumonia-like" or "non-pneumonia-like" depending on the original label
prior to adjudication. The original (binarized) labels along with their corresponding adjudicated label, can be found in
the [noisy_kaggle_dataset.csv](DataSelection/datasets/noisy_kaggle_dataset.csv) file. The dataset class for this dataset
is the [rsna_cxr.py](DataSelection/datasets/rsna_cxr.py) file. This dataset class will automatically load the labels
from the aforementioned file.

###### How to use it 
The code will assume that the Kaggle dataset is present on your machine (see dataset above for instructions) and that 
your config points to the correct `dataset_dir` location. 

## How to train supervised models
#### General
The main entry point for training a supervised model (vanilla or coteaching) is [train.py](DataSelection/deep_learning/train.py). 
The code requires you to provide a config file specifying the dataset to use, the training specification (batch size, scheduler etc...),
whether to use vanilla or coteaching training, which augmentation to use ...

Train the model using the command
```python
python DataSelection/deep_learning/train.py  --config <path to config>
```

For each of the dataset used in our experiments, we have defined a config to run training easily off the shelf.

#### CIFAR10H
In order to run: 
* vanilla resnet training please use the [DataSelection/configs/models/cifar10h/resnet.yaml](DataSelection/configs/models/cifar10h/resnet.yaml) config.
* co-teaching resnet training:  [DataSelection/configs/models/cifar10h/resnet_co_teaching.yaml](DataSelection/configs/models/cifar10h/resnet_co_teaching.yaml) config
* co-teaching resnet with graph and mean teacher [DataSelection/configs/models/cifar10h/resnet_co_teaching_ssup_ema_graph.yaml](DataSelection/configs/models/cifar10h/resnet_co_teaching_ssup_ema_graph.yaml)
 
#### Noisy Kaggle set
To run any model on this dataset, you will need to first make sure you have the dataset uploaded onto your machine (see dataset section).
In order to run:
* vanilla densenet121 training please use the [DataSelection/configs/models/rsna/densenet121_scratch.yaml](DataSelection/configs/models/rsna/densenet121_scratch.yaml) config.
* co-teaching densenet121 training:  [DataSelection/configs/models/rsna/densenet121_scratch_coteaching.yaml](DataSelection/configs/models/rsna/densenet121_scratch_coteaching.yaml) config
* co-teaching densenet121 with graph and mean teacher [DataSelection/configs/models/rsna/densenet121_scratch_coteaching_ema_graph.yaml](DataSelection/configs/models/rsna/densenet121_scratch_coteaching_ema_graph.yaml)
 
## How to pretrain embeddings with an unsupervised SimCLR model
#### General
For the unsupervised training of our models, we rely on PyTorch Lightning and Pytorch Lightining bolts. The main entry point
for model training is [DataSelection/deep_learning/self_supervised/main.py](DataSelection/deep_learning/self_supervised/main.py).
You will also need to feed in a config file to specify which dataset to use etc.. 
Command to use run `main --config path/to/config`
#### CIFAR10H
To train embeddings with contrastive learning on CIFAR10H use the 
[DataSelection/deep_learning/self_supervised/configs/cifar10h_simclr.yaml](DataSelection/deep_learning/self_supervised/configs/cifar10h_simclr.yaml)
config. 

You can find the model checkpoint at `logs/cifar10h/self_supervised/simclr_seed_<seed>/checkpoints/last.ckpt`

#### Kaggle 
For unsupervised pretraining we use the full Kaggle set (not only the small noisy subset as we don't need the labels anyways)
To train a model on this dataset, please use the [DataSelection/deep_learning/self_supervised/configs/kaggle_simclr.yaml](DataSelection/deep_learning/self_supervised/configs/kaggle_simclr.yaml)
config. 

You can find the model checkpoint at `logs/kaggle/self_supervised/simclr_seed_<seed>/checkpoints/last.ckpt`

## How to finetune a model based on frozen SimCLR embeddings
### General
After having trained your unsupervised models, you can learn a simple linear head of top of the (frozen) embeddings. For
this you can again use [train.py](DataSelection/deep_learning/train.py), simply updating your config to specify 
the location of the trained encoder's checkpoint in the `train.self_supervision.checkpoints` field.
### CIFAR10H
To finetune a linear head on CIFAR10H use the 
[DataSelection/configs/models/cifar10h/resnet_self_supervision.yaml](DataSelection/configs/models/cifar10h/resnet_self_supervision.yaml)
config. 

### Kaggle 
To finetune a linear head on the full Kaggle set use the [DataSelection/configs/models/rsna/self_supervised.yaml](DataSelection/configs/models/rsna/self_supervised.yaml)
config.

## How to run the data selection simulation
To run the data selection simulation you will need to run [benchmark 2](DataSelection/scripts/benchmark2.py) with
a list of configs in the `--config` arguments as well as a list of seeds to use for sampling in the `--seeds` arguments.

```
python benchmark2.py --config path/config1 path/config2 --seeds 1 2 3
```
You will need to provide `selector_configs` as config arguments. A selector config will allow you to specify which
selector to use and which model config to use for inference. All selectors config can be found in the 
[configs/selection](DataSelection/configs/selection) folder. 

## References:

Scikit-learn pages on Graph Diffusion and How LP can be used in an active learning setting:

* https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html

* https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits_active_learning.html

Graph diffusion for semi-supervised classification:

* http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.115.3219

Diffusion with priors (Section 3.4)

* https://openaccess.thecvf.com/content_cvpr_2018/papers/Douze_Low-Shot_Learning_With_CVPR_2018_paper.pdf

Bald Score and Ensembles to capture cases with high epistemic uncertainty:

* https://oatml.cs.ox.ac.uk/blog/2019/06/24/batchbald.html

* https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf