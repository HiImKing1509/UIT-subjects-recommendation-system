import pandas as pd
import argparse

from datetime import datetime
import pytz
timezone = pytz.timezone('Asia/Ho_Chi_Minh')

from tqdm import tqdm
from augmentation.data_augmentation import add_condition_description
from get_attention_matrix import get_attention_matrix
from similarity_subjects import get_related_subjects
from similarity_students import get_related_students
from get_score_single import get_score_single
from phobert_pretrain_weights import load_phobert_pretrain
from tools.train import train
from tools.train_single import train_single

def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Train model')

    # Add arguments
    parser.add_argument('-o', '--option', default='train', type=str, help='Description of type of training')
    parser.add_argument('-a', '--attention-matrix', default='available', type=str, help='Description of type of get attention matrix')
    parser.add_argument('-s', '--save-attention-matrix', default=0, type=int, help='Description of type of get attention matrix')
    parser.add_argument('-i', '--threshold-i', default=20, type=int, help='Description of threshold for taking similar subjects')
    parser.add_argument('-j', '--threshold-j', default=8, type=int, help='Description of threshold for taking similar students')
    args = parser.parse_args()
    
    return args

def main():

    args = parse_args()
    print(f"Your options: {args}")
    # Load dataset
    dataset_original = pd.read_csv('./data/datasets.csv')
    dataset_original = dataset_original[[        \
        'mssv', 'gioitinh', 
        'CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT',   \
        'CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN',  \
        'mamh', 'tenmh' ,'mota',    \
        'nganh_BB', 'nganh_BMAV', 'nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT',     \
        'diem_hp',      \
        'trangthai',        \
    ]]
    dataset_original = add_condition_description(dataset_original)
    dataset_original = dataset_original.drop_duplicates(subset=['mssv', 'mamh'], keep='last', inplace=False)
    dataset = dataset_original.drop(dataset_original[(dataset_original['nganh_BB'] == 1) | (dataset_original['nganh_BMAV'] == 1)].index)

    print(f'Successfully load datasets')
    
    if args.save_attention_matrix not in [0, 1]:
        raise ValueError(f'Error: -s or --save-attention-matrix value should belong to 0 (not save) or 1 (save)')
    
    if args.attention_matrix == 'available':
        attention_matrix = pd.read_csv('./data/attention_matrix.csv')
        print(f'Successfully load attention matrix!')
    elif args.attention_matrix == 'unavailable':
        print(f'WARNINGS: Loading PhoBERT pre-trained models and running to tokenize the large sentences in the database will take a few minutes.')
        phobert, tokenizer = load_phobert_pretrain()
        if args.save_attention_matrix == 1:
            current_time = datetime.now(timezone)
            year = current_time.year
            month = current_time.month
            day = current_time.day
            hour = current_time.hour
            minute = current_time.minute
            second = current_time.second
            microsecond = current_time.microsecond
            save_attn_matrix = f'{year}_{month}_{day}_{hour}_{minute}_{second}_{microsecond}'
        else:
            save_attn_matrix = None
        attention_matrix = get_attention_matrix(
            dataset=dataset,
            dataset_original=dataset_original,
            phobert=phobert,
            tokenizer=tokenizer,
            save_attn_matrix=save_attn_matrix)
            
        print(f'Successfully load attention matrix!')
    else:
        raise ValueError('Error: -a or --attention-matrix describes the mode of getting attention matrix. \n- "available" allows you to read the available matrix in our resources. \n- "unavailable" allows you to rerun the progress of creating an attention subjects matrix, including load PhoBERT pre-trained models and tokenizing subjects described sentences. This action will take a few minutes.')
        
    
    if args.option == 'train':
        print(f'WARNINGS: With the "train" option, two thresholds are ignored due to the brute force approach in config. More information in tools/train.py')
        train(
            dataset_original=dataset_original, 
            attention_matrix=attention_matrix)
    elif args.option == 'train_single':
        if args.threshold_i < 15 or args.threshold_i > 22:
            raise ValueError('Based on the density experiment, we temporarily provide some range limitations for threshold taking the number of similarity subjects, with the minimum value equal to 15 and 22 for the maximum. These results are assumed that the system will likely recommend high-quality expected subjects.')
        if args.threshold_j < 1 or args.threshold_j > 10:
            raise ValueError('Based on the density experiment, we temporarily provide some range limitations for threshold taking the number of similarity students, with the minimum value equal to 1 and 10 for the maximum. These results are assumed that the system will likely recommend high-quality expected subjects.')
        train_single(
            dataset_original=dataset_original,
            attention_matrix=attention_matrix,
            threshold_i=args.threshold_i,
            threshold_j=args.threshold_j,)
    else:
        raise ValueError('Error: -o or --option option describes the type of training, two options you can select for your demand: \n- "train" if you want to re-train the entire database in brute-force manner on two threshold values. Our database includes 1814 students and 279 subjects.\n- "train_single" if you are going to train on the custom-established threshold in advance.')

    
    
if __name__ == '__main__':
    main()