<div align="center">
  
  # Subjects Recommendation System - University of Information Technology
  ![image](https://github.com/HiImKing1509/uit_subjects_recommendation_system/assets/84212036/d5fc4ec3-14c9-403a-970a-0e3020496659)

</div>

# Hello, You are developer

1. [Installation](#installation)
	- [Environment](#install_env)
	- [PhoBERT Pre-trained models for Vietnamese](#install_phobert)
2. [Training](#training)
	- [Training entire Database with brute-force thresholds](#train)
	- [Training entire Database with defined thresholds](#train_single)
3. [Inference](#testing)

<a name="installation"></a>
## Installation
___

<a name="install_env"></a>
### Environment
With the developer option, recommendation systems are build-depend on Pytorch, PhoBERT pre-trained models. Below are quick steps for installation:
```
conda create -n rcmsys
conda activate rcmsys
git clone https://github.com/HiImKing1509/uit_subjects_recommendation_system
cd uit_subjects_recommendation_system
pip install -r requirements.txt
```

<a name="install_phobert"></a>
### PhoBERT Pre-trained models for Vietnamese

- Clone `transformers` sources as follows:

```
git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
cd transformers
pip3 install -e .
```

- Install `transformers` with pip: `pip install transformers` or install `transformers` from <a href="https://huggingface.co/docs/transformers/installation#installing-from-source">huggingface</a>. 

- Install `tokenizers` with pip: `pip3 install tokenizers`


<a name="training"></a>
## Training
___
<a name="train"></a>
### Training entire Database with brute-force thresholds

Retraining our entire database, including 1814 students and 279 subjects. Two thresholds take top similarity subjects, and similarity students will run pair-wise brute-force manner, and the results will be saved in the results folder. The command for training is as follows:

```
python train.py     \
        --option train      \
        --attention-matrix available    \
        --save-attention-matrix 0
```

Explanation:
 - `--option` parameter describes the type of training, with values is `train`, running on pair-wise brute-force manner thresholds.

- `--attention-matrix` describes how to get an Attention matrix, with two options `available` (getting pre-create data from the database) or `unavailable` (getting from retokenizing the subject's description sentences).

- `--save-attention-matrix` confirm whether you want to save a new attention matrix, values `0` or `1`.

<a name="train_single"></a>
### Training entire Database with defined thresholds

Retraining our entire database, including 1814 students and 279 subjects. Two thresholds take top similarity subjects, and similarity students will be defined in advance, and the results will be saved in the results folder. The command for training is as follows:

```
python train.py     \
        --option train_single      \
        --attention-matrix available    \
        --save-attention-matrix 1   \
        --threshold-i 20    \
        --threshold-j 8
```

Explanation:
- `--option` parameter describes the type of training, with the value `train_single`, running on defined thresholds.

- `--attention-matrix` describes how to get an Attention matrix, with two options `available` (getting pre-create data from the database) or `unavailable` (getting from retokenizing the subject's description sentences).

- `--save-attention-matrix` confirm whether you want to save a new attention matrix, values `0` or `1`.

- `--threshold-i` defined the threshold to get top similarity subjects, `20` by default.

- `--threshold-j` defined the threshold to get top similarity students, `8` by default.

<a name="testing"></a>
## Inference
___

Using a data point that includes student name and subject name as testing for observation output of recommendation system. The command for training is as follows:

```
python test_single.py     \
        --attention-matrix ./data/attention_matrix.csv    \
        --query-student EAA0B693XPvAibaEXe99j2P15eeB04XwhZ0tzlI4    \
        --query-subject CS112
```

Explanation:
- `--attention-matrix` defined the path to get the attention matrix

- `--query-student` student name
 
- `--query-subject` subject name
