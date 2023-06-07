# Phase calculate score
def get_score_single(
        dataset_original,
        list_related_subjects=None,
        list_related_students=None,
        mamh_query=None,
        mssv_query=None):
    
    if len(list_related_students) == 0:
        return [], 0.0
    for index, student in enumerate(list_related_students):
        history_subjects = list(dataset_original[dataset_original['mssv'].isin([student])]['mamh'].unique())
        if index == 0:
            intersection = list(set(history_subjects) & set(list_related_subjects))
        else:
            intersection = list(set(history_subjects) & set(intersection))

    intersection.remove(mamh_query)
    if len(intersection) == 0:
        return [], 0.0
    sv_subjects = list(dataset_original[dataset_original['mssv'].isin([mssv_query])]['mamh'].unique())
    result = list(set(sv_subjects) & set(intersection))
    score_result = len(result) / len(intersection)
    return (intersection, score_result)