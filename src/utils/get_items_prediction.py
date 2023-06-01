import pandas as pd
def get_items_prediction(dataset, attn_matrix, name_mh, mssv_query, head_of_related_mh=30, threshold_get_mssv=0.9):
    
    # Get related items based on similary to others
    si = attn_matrix[name_mh]
    rcm_df = pd.DataFrame(attn_matrix.corrwith(si).sort_values(ascending=False)).reset_index(drop=False)
    get_head = rcm_df.head(head_of_related_mh)['index'].tolist()

    # Get more information
    scores = dataset[['mssv', 'mamh', 'diem_hp']].copy()
    scores = scores[scores['mamh'].isin(get_head)].copy()
    scores.drop_duplicates(subset=['mssv', 'mamh', 'diem_hp'], keep='last', inplace=True)
    # scores = scores.sort_values(by=['nam_th', 'hocky'])

    # Initialize matrix [mamh, mssv] and [mssv, mamh]
    users_matrix_by_mssv = scores.pivot_table(index=["mamh"],columns=["mssv"],values="diem_hp")
    users_matrix_by_mamh = scores.pivot_table(index=["mssv"],columns=["mamh"],values="diem_hp")

    # Get information of sinhvien
    similarity = users_matrix_by_mssv[mssv_query]
    rcm_df = pd.DataFrame(users_matrix_by_mssv.corrwith(similarity).sort_values(ascending=False)).reset_index(drop=False)
    rcm_df = rcm_df[rcm_df[0] >= threshold_get_mssv].copy()
    # rcm_df = rcm_df.drop(rcm_df.index[0]) # Drop index 0 due to unconsider itself

    merged_table = pd.merge(rcm_df['mssv'], users_matrix_by_mamh, on='mssv', how='inner')

    sim = merged_table[name_mh]
    rcm_df_result = pd.DataFrame(merged_table.corrwith(sim).sort_values(ascending=False)).reset_index(drop=False)
    rcm_df_result = rcm_df_result.rename(columns={rcm_df_result.columns[0]: 'index'})
    result = rcm_df_result.loc[rcm_df_result[0] > 0.5, 'index'].tolist()
    if name_mh in result:
        result.remove(name_mh)
    return result