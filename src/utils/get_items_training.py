def get_items(dataset, attn_matrix, name_mh, mssv_query, head_of_related_mh=30, threshold_get_mssv=0.9):
    
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

    filtered_df = dataset[(dataset['mssv'] == mssv_query) & (dataset['mamh'].isin(result))]
    df_result = filtered_df.drop_duplicates(subset=['mssv', 'mamh'], keep='last', inplace=False)
    # df_result = filtered_df.sort_values(by=['nam_th', 'hocky'])

    if len(result) < len(df_result):
        print('Error')

    penalty_percentage = 100.0
    if len(result) != 0:
        unit_percentage = penalty_percentage / len(result)
    else: unit_percentage = 100.0
    lst = list(set(df_result['mamh'].tolist()))

    for mh in result:
        if mh not in lst:
            penalty_percentage -= unit_percentage
        else:
            diem_hp = df_result.loc[df_result['mamh'] == mh, 'diem_hp'].iloc[0]
            if diem_hp < 5.0: 
                weight = math.floor(unit_percentage / 20.0 * 10.0) - np.e/(np.log10(result.index(mh)+0.5)+1)
            elif diem_hp <= 6.0: 
                weight = math.floor(unit_percentage / 20.0 * 4.0) - (np.e-0.5)/(np.log10(result.index(mh)+0.5)+1)
            elif diem_hp <= 7.0: 
                weight = math.floor(unit_percentage / 20.0 * 3.0) - (np.e-1.0)/(np.log10(result.index(mh)+0.5)+1)
            elif diem_hp <= 8.0: 
                weight = math.floor(unit_percentage / 20.0 * 2.0) - (np.e-1.5)/(np.log10(result.index(mh)+0.5)+1)
            elif diem_hp < 9.0:
                weight = math.floor(unit_percentage / 20.0 * 1.0) - (np.e-2.0)/(np.log10(result.index(mh)+0.5)+1)
            else:
                weight = 0.0
            penalty_percentage -= weight
    if penalty_percentage < 0.0 or len(result) == 0:
        penalty_percentage = 0.0

    return (len(df_result), len(result)), penalty_percentage