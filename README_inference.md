<div align="center">
  
  # Subjects Recommendation System - University of Information Technology
  ![image](https://github.com/HiImKing1509/uit_subjects_recommendation_system/assets/84212036/d5fc4ec3-14c9-403a-970a-0e3020496659)

</div>

# I am here to suggest some subjects for you

1. [Installation](#installation)
2. [Usage](#usage)

<a name="installation"></a>
## Installation
___

<a name="install_env"></a>
### Environment
With the user option, you must install some basic libraries such as pandas and sklearn, omit step clone PhoBERT pre-trained model. Below are quick steps for installation:
```
conda create -n rcmsys
conda activate rcmsys
git clone https://github.com/HiImKing1509/uit_subjects_recommendation_system
cd uit_subjects_recommendation_system
conda install conda install -c anaconda scikit-learn
conda install -c anaconda pandas
```

<a name="usage"></a>
## Usage
___

Your information does not belong in our database; therefore, you must provide some data such as query subject, scores, gender, faculty, and training system. Our recommendation system will process and suggest related subjects based on these data.
```
python test_custom.py     \
        --attention-matrix ./data/attention_matrix.csv    \
        --query-student 2052xxx    \
        --gender male   \
        --faculty khmt  \
        --training-system cqui  \
        --query-subject CS112   \
        --query-subject-score 8.1   \
        --it001 9.3     \
        --it002 9.3     \
        --it003 8.6     \
        --it004 9.2     \
        --it005 8.5     \
        --it006 9.9     \
        --it007 9.0
```

Explanation:
- `--attention-matrix` defined the path to get the attention matrix

- `--query-student` your name

- `--gender` your gender

- `--faculty` your faculty

- `--training-system` your training system in university

- `--query-subject` query subject to get related others

- `--query-subject-score` query subject score

- `--it00[1-7]` subjects IT00[1-7] score, `-10.0` by default (haven't learned yet)

### Example
![image](https://github.com/HiImKing1509/uit_subjects_recommendation_system/assets/84212036/3b0beb9f-c0b7-49ca-8bf3-a2ad698d7752)

