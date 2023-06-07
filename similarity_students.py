import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# Phase 2
def get_related_students(
        dataset_original=None, 
        mamh_query=None, 
        mssv_query=None,
        t_threshold=8):
    
    df_students = dataset_original[[
        'mssv', 'gioitinh', 
        'CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT',   \
        'CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN',  \
        'mamh',     \
        'diem_hp',      \
    ]]
    df_students = df_students[df_students['mamh'].isin(['IT001', 'IT002' ,'IT003', 'IT004', 'IT005', 'IT006', 'IT007'] + [mamh_query])]
    one_hot_encoded = pd.pivot_table(data=df_students, index=df_students.index, columns='mamh', values='diem_hp', fill_value=-10.0)

    # Concatenate the one-hot encoded columns with the original DataFrame
    df_students_encoded = pd.concat([df_students, one_hot_encoded], axis=1)
    df_students_merged = df_students_encoded.groupby('mssv').max().reset_index()
    df_students_merged = df_students_merged.drop(columns=['mamh', 'diem_hp'])
    
    similarity_matrix = cosine_similarity(df_students_merged.iloc[:, 1:])
    similarity_df = pd.DataFrame(similarity_matrix, index=df_students_merged['mssv'], columns=df_students_merged['mssv'])
    list_related_students = similarity_df[similarity_df.index != mssv_query].nlargest(t_threshold, columns=mssv_query).index
    return list_related_students