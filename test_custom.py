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
    parser.add_argument('-g', '--gender', type=str, help="Description of student's gender")
    parser.add_argument('-f', '--faculty', type=str, help="Description of student's faculty")
    parser.add_argument('-t', '--training-system', type=str, help="Description of student's training system")
    parser.add_argument('-sb', '--query-subject', type=str, help='Description of query subject')
    parser.add_argument('-sc', '--query-subject-score', type=float, help='Description of query subject score')
    parser.add_argument('-it1', '--it001', default=0.0, type=float, help='Description of IT001 subject score')
    parser.add_argument('-it2', '--it002', default=0.0, type=float, help='Description of IT002 subject score')
    parser.add_argument('-it3', '--it003', default=0.0, type=float, help='Description of IT003 subject score')
    parser.add_argument('-it4', '--it004', default=0.0, type=float, help='Description of IT004 subject score')
    parser.add_argument('-it5', '--it005', default=0.0, type=float, help='Description of IT005 subject score')
    parser.add_argument('-it6', '--it006', default=0.0, type=float, help='Description of IT006 subject score')
    parser.add_argument('-it7', '--it007', default=0.0, type=float, help='Description of IT007 subject score')
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

    data_mh = list(dataset_original['mamh'].unique())
    
    print(f'Successfully load datasets')
    
    try:
        attention_matrix = pd.read_csv(args.attention_matrix)
        print(f'Successfully load attention matrix!')    
    except Exception as e:
        print(e)
            
    if args.gender not in ['m', 'male', 'f', 'female']:
        raise ValueError('Error: -g or --gender has value should belong to ["m", "male", "f", "female"] with "m", "f" are short for "male" and "female", respectively.')
    else:
        if args.gender in ['m', 'male']: gender = 'Nam'
        else: gender = 'Nữ'
    
    if args.faculty not in ['CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT', 'cnpm', 'httt', 'khmt', 'ktmt', 'kttt', 'mmt&tt']:
        raise ValueError('Error: -f or --faculty has value should belong to ["CNPM", "HTTT", "KHMT", "KTMT", "KTTT", "MMT&TT", "cnpm", "httt", "khmt", "ktmt", "kttt", "mmt&tt"], these values stand for entire faculty of the University of Information Technology.')
    
    if args.training_system not in ['CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN', 'clc', 'cntn', 'cqui', 'cttt', 'kstn']:
        raise ValueError('Error: -t or --training-system has value should belong to ["CLC", "CNTN", "CQUI", "CTTT", "KSTN", "clc", "cntn", "cqui", "cttt", "kstn"], these values stand for entire training systems of the University of Information Technology.')
    
    if args.query_subject not in data_mh:
        raise ValueError(f'Error: -sb or --query-subject has value should belong to {data_mh}, these values stand for entire subjects of the University of Information Technology until 2018. In our attempt, we provide {len(data_mh)} subjects in the database. {args.query_subject} does not belong in our supported subjects. This subject code has probably been replaced or is the new subject.')
        
    if args.query_subject_score < 0.0 and args.query_subject_score > 10.0:
        raise ValueError(f'Error: -sc or --query-subject-score has value should belong to range [0, 10]')
    
    lst_score_inp = [args.it001, args.it002, args.it003, args.it004, args.it005, args.it006, args.it007]
    for i in range(0, 7):
        if lst_score_inp[i] == -10:
            continue
        elif lst_score_inp[i] < 0.0 and lst_score_inp[i] > 10.0:
            raise ValueError(f'Error: -it{i+1} or --it00{i+1} has value should belong to range [0, 10]')

    dataset_original = dataset_original.append([
        input_prediction(_mssv = args.query_student, _gioitinh = gender, _khoa = args.faculty.upper(), _hedaotao = args.training_system.upper(), _mamh = 'IT001', _diem_hp = args.it001),
        input_prediction(_mssv = args.query_student, _gioitinh = gender, _khoa = args.faculty.upper(), _hedaotao = args.training_system.upper(), _mamh = 'IT002', _diem_hp = args.it002),
        input_prediction(_mssv = args.query_student, _gioitinh = gender, _khoa = args.faculty.upper(), _hedaotao = args.training_system.upper(), _mamh = 'IT003', _diem_hp = args.it003),
        input_prediction(_mssv = args.query_student, _gioitinh = gender, _khoa = args.faculty.upper(), _hedaotao = args.training_system.upper(), _mamh = 'IT004', _diem_hp = args.it004),
        input_prediction(_mssv = args.query_student, _gioitinh = gender, _khoa = args.faculty.upper(), _hedaotao = args.training_system.upper(), _mamh = 'IT005', _diem_hp = args.it005),
        input_prediction(_mssv = args.query_student, _gioitinh = gender, _khoa = args.faculty.upper(), _hedaotao = args.training_system.upper(), _mamh = 'IT006', _diem_hp = args.it006),
        input_prediction(_mssv = args.query_student, _gioitinh = gender, _khoa = args.faculty.upper(), _hedaotao = args.training_system.upper(), _mamh = 'IT007', _diem_hp = args.it007),
        input_prediction(_mssv = args.query_student, _gioitinh = gender, _khoa = args.faculty.upper(), _hedaotao = args.training_system.upper(), _mamh = args.query_subject.upper(), _diem_hp = args.query_subject_score),
    ], ignore_index=True)
    df_students = dataset_original[[
        'mssv', 'gioitinh', 
        'CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT',   \
        'CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN',  \
        'mamh',     \
        'diem_hp',      \
    ]]
    print(df_students.tail(8))
    
    list_result, result = test_single(
        dataset_original=dataset_original,
        mssv=args.query_student,
        ten_mh=args.query_subject.upper(),
        attention_matrix=attention_matrix,
    )
    
    print(f'The system recommend: {list_result}')
    
if __name__ == '__main__':
    main()