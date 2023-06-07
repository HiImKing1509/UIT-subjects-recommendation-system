from similarity_subjects import get_related_subjects
from similarity_students import get_related_students
from get_score_single import get_score_single


def test_single(
    dataset_original=None,
    mssv=None,
    ten_mh=None,
    attention_matrix=None,
):

    list_related_subjects = get_related_subjects(
        mamh_query=ten_mh, 
        attention_matrix=attention_matrix,
        k_threshold=20)

    list_related_students = get_related_students(
        dataset_original=dataset_original, 
        mamh_query=ten_mh, 
        mssv_query=mssv,
        t_threshold=8)

    list_result, result = get_score_single(
        dataset_original=dataset_original,
        list_related_subjects=list_related_subjects,
        list_related_students=list_related_students,
        mamh_query=ten_mh,
        mssv_query=mssv)

    return list_result, result