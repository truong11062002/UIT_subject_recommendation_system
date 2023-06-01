# Training
losses = 0.0
penalty = 0.0
lst_users = list(dataset_original['mssv'].unique())

for index_user, user in tqdm(enumerate(lst_users)):
    
    # print(f'\n------------ User: {index_user+1} -------------')

    data_user = filter(dataset, 'mssv', [user])
    data_user.drop_duplicates(subset=['mssv', 'mamh'], keep='last', inplace=False)

    lst_user_mamh = list(data_user['mamh'])

    loss_user = 0.0
    penalty_percentage_user = 0.0
    for index, user_mamh in enumerate(lst_user_mamh):
        (missing, full), penalty_percentage = get_items(
            dataset=dataset,
            attn_matrix=attention_matrix,
            name_mh=user_mamh,
            mssv_query=user,
            head_of_related_mh=18,
            threshold_get_mssv=0.9
        )
        
        if full != 0: loss_user += missing / full
        penalty_percentage_user += penalty_percentage

    losses += (loss_user / len(lst_user_mamh))
    penalty += (penalty_percentage_user / len(lst_user_mamh))
    print(f'\nUser {index_user}, Number of subjects: {len(data_user)}, Appearance rate: {round(loss_user * 100.0/ len(lst_user_mamh), 2)}%, Penalty: {round(penalty_percentage_user / len(lst_user_mamh), 2)}% ===> Total appearance rate: {round(losses * 100.0 / (index_user + 1), 2)}%, Total penalty: {round(penalty / (index_user + 1), 2)}%')

    if index_user == 5:
        break
print(f'Total Appearance rate: {round(losses * 100.0 / (index_user + 1), 2)}%, Total penalty: {round(penalty / (index_user + 1), 2)}%')
print(f'Evaluating metrics: {hmean([round(losses * 100.0 / (index_user + 1), 2), round(penalty / (index_user + 1), 2)])}')