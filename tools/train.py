from tqdm import tqdm

from datetime import datetime
import pytz
timezone = pytz.timezone('Asia/Ho_Chi_Minh')

from similarity_subjects import get_related_subjects
from similarity_students import get_related_students
from get_score_single import get_score_single

def train(
    dataset_original=None,
    attention_matrix=None,
):
    i = 15
    while 15 <= i and i <= 22:
        j = 1
        while 1 <= j and j <= 10:
            dataset_train = dataset_original.drop(dataset_original[(dataset_original['nganh_BB'] == 1) | (dataset_original['nganh_BMAV'] == 1)].index)
            lst_users = list(dataset_train['mssv'].unique())
            score = 0.0
            lst_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            print(f'\n=============== threshold_get_subjects: {i}, threshold_get_students: {j} ===============')

            for index_user, user in tqdm(enumerate(lst_users)):
                data_user = dataset_train[dataset_train['mssv'].isin([user])]
                data_user.drop_duplicates(subset=['mssv', 'mamh'], keep='last', inplace=False)

                lst_user_mamh = list(data_user['mamh'])
                # print(f'\nUser: {index_user+1} with the number of subjects: {len(lst_user_mamh)} -------------')
                result_user_single = 0.0
                for index, user_mamh in enumerate(lst_user_mamh):
                    list_related_subjects = get_related_subjects(
                        mamh_query=user_mamh, 
                        attention_matrix=attention_matrix,
                        k_threshold=i)
                    
                    list_related_students = get_related_students(
                        dataset_original=dataset_original, 
                        mamh_query=user_mamh, 
                        mssv_query=user,
                        t_threshold=j)
                    
                    list_result, result = get_score_single(
                        dataset_original=dataset_original,
                        list_related_subjects=list_related_subjects,
                        list_related_students=list_related_students,
                        mamh_query=user_mamh,
                        mssv_query=user)
                    # print(f'\nUser: {user} ------ Subject: {user_mamh} ------ list: {list_result} ------ Score = {round(result, 2)}')
                    if len(list_result) <= 7:
                        lst_count[len(list_result)] += 1
                    else:
                        lst_count[-1] += 1

                    result_user_single += result
                # print(f'Total loss single for {user}: {result_user_single / len(lst_user_mamh)}')
                # print(f'\n================================================================================\n')
                
                score += (result_user_single / (index + 1))
                if index_user % 100 == 99:
                    print(f'\n{index_user + 1} students, threshold_get_subjects: {i}, threshold_get_students: {j}, Score: {score * 100 / (index_user + 1)}')
                    
            print(f'Total score: {score * 100 / (index_user + 1)}')
            print(f'Count: {lst_count}')

            current_time = datetime.now(timezone)
            year = current_time.year
            month = current_time.month
            day = current_time.day
            hour = current_time.hour
            minute = current_time.minute
            second = current_time.second
            microsecond = current_time.microsecond
            save_file = f'{year}_{month}_{day}_{hour}_{minute}_{second}_{microsecond}'
            with open(f'./results/result_train_{save_file}.log', 'a') as file:
                file.write('{')
                file.write(f'threshold_get_subjects: {i}, threshold_get_students: {j}, score: {score * 100 / (index_user + 1)}, count: {lst_count}')
                file.write('}\n')

            j += 1
        i += 1