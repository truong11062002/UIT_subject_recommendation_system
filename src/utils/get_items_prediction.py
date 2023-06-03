import pandas as pd
import numpy as np
import torch
from itertools import product
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

def rename(df, object_name):
    df = df.rename(columns=object_name)
    return df
def filter(dataset, name_col: str, name_value: list):
    data = dataset[dataset[name_col].isin(name_value)]
    return data

def get_feature_vector(lst):
    lst_tokenizer = [torch.tensor([tokenizer.encode(key)]) for key in lst]
    with torch.no_grad():
        feature_vectors = {}
        for index, input_ids in enumerate(lst_tokenizer):
            feature_vectors[lst[index]] = phobert(input_ids).pooler_output
    return feature_vectors

def add_condition_description(dataset):
    # Additional conditions for better similarity among clusters involving subjects 
    conditions = {
        'nganh_BB': 'Môn học bắt buộc',
        'nganh_BMAV': 'Môn học ngoại ngữ',
        'nganh_CNPM': 'Môn học thuộc khoa Công Nghệ Phần Mềm',
        'nganh_HTTT': 'Môn học thuộc khoa Hệ Thống Thông Tin',
        'nganh_KHMT': 'Môn học thuộc khoa Khoa Học Máy Tính',
        'nganh_KTMT': 'Môn học thuộc khoa Kĩ Thuật Máy Tính',
        'nganh_KTTT': 'Môn học thuộc khoa Kĩ Thuật Thông Tin',
        'nganh_MMT&TT': 'Môn học thuộc khoa Mạng Máy Tính và Truyền Thông'
    }
    for condition, value in conditions.items():
        mask = dataset[condition] == 1
        dataset.loc[mask, 'monhoc_encode'] = dataset['tenmh'].astype(str) + ': ' + dataset['mota'].astype(str) + ' ' + value
    return dataset

def get_items_prediction(dataset_original_usage, attn_matrix, name_mh, mssv_query, head_of_related_mh=20, threshold_get_mssv=0.5, threshold_get_list=0.5):
    
    
    # Get related items based on similary to others
    si = attn_matrix[name_mh]
    # print(f'name_mh: {name_mh}')
    # print(f'attn_matrix columns: {attn_matrix.columns}')
    # print('IT001' in attn_matrix.columns)
    rcm_df = pd.DataFrame(attn_matrix.corrwith(si).sort_values(ascending=False)).reset_index(drop=False)
    rcm_df = rcm_df.rename(columns={rcm_df.columns[0]: 'mamh'})
    get_head = rcm_df.head(head_of_related_mh)['mamh'].tolist()

    # Get some feature vectors
    features_danhgia_diem = get_feature_vector(['Không xác định', 'Kém', 'Yếu', 'Trung bình', 'Khá', 'Giỏi', 'Xuất sắc'])
    features_danhgia_gioitinh = get_feature_vector(['Giới tính nam', 'Giới tính nữ'])
    features_danhgia_khoa = get_feature_vector([
        'thuộc khoa Công Nghệ Phần Mềm',
        'thuộc khoa Hệ Thống Thông Tin',
        'thuộc khoa Khoa Học Máy Tính',
        'thuộc khoa Kĩ Thuật Máy Tính',
        'thuộc khoa Kĩ Thuật Thông Tin',
        'thuộc khoa Mạng Máy Tính và Truyền Thông'
    ])
    features_danhgia_nganh = get_feature_vector([
        'thuộc ngành Công Nghệ Phần Mềm',
        'thuộc ngành Hệ Thống Thông Tin',
        'thuộc ngành Khoa Học Máy Tính',
        'thuộc ngành Kĩ Thuật Máy Tính',
        'thuộc ngành Kĩ Thuật Thông Tin',
        'thuộc ngành Mạng Máy Tính và Truyền Thông'
    ])
    features_danhgia_hedaotao = get_feature_vector([
        'Hệ chất lượng cao',
        'Hệ cử nhân tài năng',
        'Hệ chính quy đại trà',
        'Hệ chương trình tiên tiến',
        'Hệ kĩ sư tài năng'
    ])

    # Get LICH SU HOC TAP
    get_lichsuhoctap = dataset_original_usage.loc[dataset_original_usage['nganh_BB'] == 1].copy()
    
    get_lichsuhoctap = get_lichsuhoctap[get_lichsuhoctap['mamh'].isin(['IT001', 'IT002' ,'IT003', 'IT004', 'IT005', 'IT006', 'IT007'])]
    get_lichsuhoctap = get_lichsuhoctap.drop_duplicates(subset=['mssv', 'mamh'], keep='last', inplace=False)
    mssv_unique = pd.DataFrame(dataset_original_usage['mssv'].unique(), columns=['mssv'])
    df_lichsuhoctap = pd.merge(get_lichsuhoctap, mssv_unique, on='mssv', how='right')
    df_lichsuhoctap = df_lichsuhoctap.fillna(100.0, inplace=False)

    # Featurization score which students reached
    def assign_tensor(row):
        if row['diem_hp'] < 5.0:
            return np.mean(features_danhgia_diem['Kém'].numpy())
        elif row['diem_hp'] >= 5.0 and row['diem_hp'] < 6.0:
            return np.mean(features_danhgia_diem['Yếu'].numpy())
        elif row['diem_hp'] >= 6.0 and row['diem_hp'] < 7.0:
            return np.mean(features_danhgia_diem['Trung bình'].numpy())
        elif row['diem_hp'] >= 7.0 and row['diem_hp'] < 8.0:
            return np.mean(features_danhgia_diem['Khá'].numpy())
        elif row['diem_hp'] >= 8.0 and row['diem_hp'] < 9.0:
            return np.mean(features_danhgia_diem['Giỏi'].numpy())
        elif row['diem_hp'] >= 9.0 and row['diem_hp'] <= 10.0:
            return np.mean(features_danhgia_diem['Xuất sắc'].numpy())
        elif row['diem_hp'] >= 99.9:
            return np.mean(features_danhgia_diem['Không xác định'].numpy())
    df_lichsuhoctap['lichsuhoctap'] = [None] * len(df_lichsuhoctap)
    df_lichsuhoctap['lichsuhoctap'] = df_lichsuhoctap.apply(assign_tensor, axis=1)
    df_lichsuhoctap = df_lichsuhoctap.pivot_table(index=['mssv'],columns=['mamh'], values='lichsuhoctap')
    df_lichsuhoctap = df_lichsuhoctap.reset_index()
    df_lichsuhoctap = df_lichsuhoctap.drop(100.0, axis=1)
    df_lichsuhoctap = df_lichsuhoctap.fillna(np.mean(features_danhgia_diem['Không xác định'].numpy()), inplace=False)
    
    df_lichsuhoctap['features'] = df_lichsuhoctap.apply(lambda row: torch.tensor([np.sum((row['IT001'] * 1, row['IT002'] * 2, row['IT003'] * 3, row['IT004'] * 4, row['IT005'] * 5, row['IT006'] * 6, row['IT007'] * 7))]), axis=1)
    
    # st.write(df_lichsuhoctap.head(10))
    # Successfully take the feature vectors based on history of students
    matrix_df_lichsuhoctap = df_lichsuhoctap.drop(['IT001', 'IT002', 'IT003', 'IT004', 'IT005', 'IT006', 'IT007'], axis=1)

    # Get score of students
    scores = dataset_original_usage[[
        'mssv', 'gioitinh', \
        'CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT', \
        'CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN', \
        'mamh', 'diem_hp', \
        'nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT']].copy()
    scores = scores[scores['mamh'].isin(get_head)].copy()
    scores.drop_duplicates(subset=['mssv', 'mamh', 'diem_hp'], keep='last', inplace=True)
    lst_names = scores[[
        'mssv', 'gioitinh', \
        'CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT', \
        'CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN', \
        'nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT']].copy().drop_duplicates()
    lst_mhs = scores['mamh'].unique()
    df = pd.DataFrame([(*row, mh) for (index, row), mh in list(product(lst_names.iterrows(), lst_mhs))], columns=['mssv', 'gioitinh', 'CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT', 'CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN', 'nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT', 'mamh'])
    scores = pd.merge(df, scores, on=[
        'mssv', 'gioitinh', \
        'CNPM', 'HTTT', 'KHMT', 'KTMT', 'KTTT', 'MMT&TT', \
        'CLC', 'CNTN', 'CQUI', 'CTTT', 'KSTN', \
        'mamh', \
        'nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT'], how='left')
    scores.fillna(100.0, inplace=True)

    best_sinhvien_features_merged = pd.merge(matrix_df_lichsuhoctap, scores, on='mssv', how='right')


    # Concat with the characters and informations of students
    def concat_feature_vectors(conditions, choices, dataset, feature_vector, get_mean=False):
        for condition, choice in zip(conditions, choices):
            mask = condition.to_numpy()
            features_to_concat = np.sum(feature_vector[choice].numpy())
            if get_mean == False:
                dataset.loc[mask, 'features'] = dataset.loc[mask, 'features'].apply(lambda arr: torch.cat((arr, torch.tensor([features_to_concat]))))
            else:
                dataset.loc[mask, 'features'] = dataset.loc[mask, 'features'].apply(lambda arr: torch.cat((arr, torch.tensor([features_to_concat]))).sum().item())
        return dataset

    ## Giới tính
    best_sinhvien_features_merged = concat_feature_vectors(
        conditions = [
            (best_sinhvien_features_merged['gioitinh'] == 1),
            (best_sinhvien_features_merged['gioitinh'] == 0)
        ],
        choices = ['Giới tính nam', 'Giới tính nữ'],
        dataset = best_sinhvien_features_merged,
        feature_vector=features_danhgia_gioitinh,
    )

    ## Khoa
    best_sinhvien_features_merged = concat_feature_vectors(
        conditions = [
            (best_sinhvien_features_merged['CNPM'] == 1),
            (best_sinhvien_features_merged['HTTT'] == 1),
            (best_sinhvien_features_merged['KHMT'] == 1),
            (best_sinhvien_features_merged['KTMT'] == 1),
            (best_sinhvien_features_merged['KTTT'] == 1),
            (best_sinhvien_features_merged['MMT&TT'] == 1),
        ],
        choices = [
            'thuộc khoa Công Nghệ Phần Mềm',
            'thuộc khoa Hệ Thống Thông Tin',
            'thuộc khoa Khoa Học Máy Tính',
            'thuộc khoa Kĩ Thuật Máy Tính',
            'thuộc khoa Kĩ Thuật Thông Tin',
            'thuộc khoa Mạng Máy Tính và Truyền Thông',
        ],
        dataset = best_sinhvien_features_merged,
        feature_vector=features_danhgia_khoa,
    )

    ## Hệ đào tạo
    best_sinhvien_features_merged = concat_feature_vectors(
        conditions = [
            (best_sinhvien_features_merged['CLC'] == 1),
            (best_sinhvien_features_merged['CNTN'] == 1),
            (best_sinhvien_features_merged['CQUI'] == 1),
            (best_sinhvien_features_merged['CTTT'] == 1),
            (best_sinhvien_features_merged['KSTN'] == 1),
        ],
        choices = [
            'Hệ chất lượng cao',
            'Hệ cử nhân tài năng',
            'Hệ chính quy đại trà',
            'Hệ chương trình tiên tiến',
            'Hệ kĩ sư tài năng',
        ],
        dataset = best_sinhvien_features_merged,
        feature_vector=features_danhgia_hedaotao,
    )

    ## Môn học thuộc nhóm ngành nào?
    best_sinhvien_features_merged = concat_feature_vectors(
        conditions = [
            (best_sinhvien_features_merged['nganh_CNPM'] == 1),
            (best_sinhvien_features_merged['nganh_HTTT'] == 1),
            (best_sinhvien_features_merged['nganh_KHMT'] == 1),
            (best_sinhvien_features_merged['nganh_KTMT'] == 1),
            (best_sinhvien_features_merged['nganh_KTTT'] == 1),
            (best_sinhvien_features_merged['nganh_MMT&TT'] == 1),
        ],
        choices = [
            'thuộc ngành Công Nghệ Phần Mềm',
            'thuộc ngành Hệ Thống Thông Tin',
            'thuộc ngành Khoa Học Máy Tính',
            'thuộc ngành Kĩ Thuật Máy Tính',
            'thuộc ngành Kĩ Thuật Thông Tin',
            'thuộc ngành Mạng Máy Tính và Truyền Thông',
        ],
        dataset = best_sinhvien_features_merged,
        feature_vector=features_danhgia_nganh,
    )

        ## Điểm môn học đạt được
    best_sinhvien_features_merged = concat_feature_vectors(
        conditions = [
            (best_sinhvien_features_merged['diem_hp'] < 5.0),
            ((best_sinhvien_features_merged['diem_hp'] < 6.0) & (best_sinhvien_features_merged['diem_hp'] >= 5.0)),
            ((best_sinhvien_features_merged['diem_hp'] < 7.0) & (best_sinhvien_features_merged['diem_hp'] >= 6.0)),
            ((best_sinhvien_features_merged['diem_hp'] < 8.0) & (best_sinhvien_features_merged['diem_hp'] >= 7.0)),
            ((best_sinhvien_features_merged['diem_hp'] < 9.0) & (best_sinhvien_features_merged['diem_hp'] >= 8.0)),
            ((best_sinhvien_features_merged['diem_hp'] <= 10.0) & (best_sinhvien_features_merged['diem_hp'] >= 9.0)),
            (best_sinhvien_features_merged['diem_hp'] == 100.0),
        ],
        choices = ['Kém', 'Yếu', 'Trung bình', 'Khá', 'Giỏi', 'Xuất sắc', 'Không xác định'],
        dataset = best_sinhvien_features_merged,
        feature_vector=features_danhgia_diem,
        get_mean=True
    )

    # Initialize matrix [mamh, mssv] and [mssv, mamh] with corresponding features
    users_matrix_by_mssv = best_sinhvien_features_merged.pivot_table(index=["mamh"],columns=["mssv"],values="features")
    users_matrix_by_mamh = best_sinhvien_features_merged.pivot_table(index=["mssv"],columns=["mamh"],values="features")

    # Finding similarity among students and input query student
    student_similarity = users_matrix_by_mssv[mssv_query]
    rcm_df = pd.DataFrame(users_matrix_by_mssv.corrwith(student_similarity).sort_values(ascending=False)).reset_index(drop=False)
    rcm_df = rcm_df[rcm_df[0] >= threshold_get_mssv].copy()
    try:
        rcm_df = rcm_df.drop(mssv_query)
    except:
        print('Pass')

    # Completely filter matrix table where only contain interested subjects and others similarity students
    merged_table = pd.merge(rcm_df['mssv'], users_matrix_by_mamh, on='mssv', how='inner')

    sim = merged_table[name_mh]
    rcm_df_result = pd.DataFrame(merged_table.corrwith(sim).sort_values(ascending=False)).reset_index(drop=False)
    rcm_df_result = rcm_df_result.rename(columns={rcm_df_result.columns[0]: 'index'})
    result = rcm_df_result.loc[rcm_df_result[0] > threshold_get_list, 'index'].tolist()
    
    if (len(result) <= 1):
        result = rcm_df_result.loc[rcm_df_result[0] > 0.05, 'index'].tolist()
    
    if name_mh in result:
        result.remove(name_mh)
    # breakpoint()
    return result

# Load dataset
dataset_original = pd.read_csv('./src/data/data_norm_final.csv')
attention_matrix = pd.read_csv('./src/data/attention_matrix.csv')
# dataset_original_usage = dataset_original.copy()
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



result = get_items_prediction(
            dataset_original_usage=dataset_original,
            attn_matrix=attention_matrix, 
            name_mh= 'NT101',
            mssv_query= '9C8782A5XPvAibaEXe9lF2o7VejWNskIJ3b/KH8Y',
            head_of_related_mh=20,
            threshold_get_mssv=0.1,
            threshold_get_list=0.5
        )
print(result)

    # filtered_df = dataset_original_usage[(dataset_original_usage['mssv'] == mssv_query) & (dataset_original_usage['mamh'].isin(result))]
    # df_result = filtered_df.drop_duplicates(subset=['mssv', 'mamh'], keep='last', inplace=False)

    # if len(result) < len(df_result):
    #     print('Error')

    # penalty_percentage = 100.0
    # if len(result) != 0:
    #     unit_percentage = penalty_percentage / len(result)
    # else: unit_percentage = 100.0
    # lst = list(set(df_result['mamh'].tolist()))

    # for mh in result:
    #     if mh not in lst:
    #         penalty_percentage -= unit_percentage
    #     else:
    #         diem_hp = df_result.loc[df_result['mamh'] == mh, 'diem_hp'].iloc[0]
    #         if diem_hp < 5.0: 
    #             weight = math.floor(unit_percentage / 20.0 * 10.0) - np.e/(np.log10(result.index(mh)+0.5)+1)
    #         elif diem_hp <= 6.0: 
    #             weight = math.floor(unit_percentage / 20.0 * 4.0) - (np.e-0.5)/(np.log10(result.index(mh)+0.5)+1)
    #         elif diem_hp <= 7.0: 
    #             weight = math.floor(unit_percentage / 20.0 * 3.0) - (np.e-1.0)/(np.log10(result.index(mh)+0.5)+1)
    #         elif diem_hp <= 8.0: 
    #             weight = math.floor(unit_percentage / 20.0 * 2.0) - (np.e-1.5)/(np.log10(result.index(mh)+0.5)+1)
    #         elif diem_hp < 9.0:
    #             weight = math.floor(unit_percentage / 20.0 * 1.0) - (np.e-2.0)/(np.log10(result.index(mh)+0.5)+1)
    #         else:
    #             weight = 0.0
    #         penalty_percentage -= weight
    # if penalty_percentage < 0.0 or len(result) == 0:
    #     penalty_percentage = 0.0

    # print(result)
    # return (len(df_result), len(result)), penalty_percentage