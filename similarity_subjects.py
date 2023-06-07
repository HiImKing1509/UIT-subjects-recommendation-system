import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Phase 1
def get_related_subjects(
        mamh_query=None, 
        attention_matrix=None, 
        k_threshold=20):
    
    si = attention_matrix[mamh_query]
    try:
        rcm_df = pd.DataFrame(attention_matrix.corrwith(si).sort_values(ascending=False)).reset_index(drop=False)
    except:
        pass
    rcm_df = rcm_df.rename(columns={rcm_df.columns[0]: 'mamh'})
    related_subjects = rcm_df.head(k_threshold)['mamh'].tolist()
    return related_subjects