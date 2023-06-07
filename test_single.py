import pandas as pd
import argparse
from augmentation.data_augmentation import add_condition_description
from tools.test_single import test_single
from custom_student import input_prediction

def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Train model')

    # Add arguments
    parser.add_argument('-a', '--attention-matrix', default='./data/attention_matrix.csv', type=str, help='Description of type of get attention matrix')
    parser.add_argument('-st', '--query-student', type=str, help='Description of query student')
    parser.add_argument('-sb', '--query-subject', type=str, help='Description of query subject')
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

    if dataset_original[dataset_original[['mssv', 'mamh']].isin((args.query_student, args.query_subject.upper()))].all(axis=1).any():
        try:
            attention_matrix = pd.read_csv(args.attention_matrix)
            print(f'Successfully load attention matrix!')    
        except Exception as e:
            print(e)
                        
        list_result, result = test_single (
            dataset_original=dataset_original,
            mssv=args.query_student,
            ten_mh=args.query_subject.upper(),
            attention_matrix=attention_matrix,
        )
        
        print(f'The system recommend: {list_result} with confidence score is {result}')
    else:
        raise ValueError(f'The information of {args.query_student} student studies {args.query_subject.upper()} subject are not exist!')
    
if __name__ == '__main__':
    main()