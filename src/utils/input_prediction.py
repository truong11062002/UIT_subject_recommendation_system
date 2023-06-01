import numpy as np
def input_prediction(
    subjects,
    _mssv = np.nan,
    _gioitinh = np.nan,
    _khoa = np.nan,
    _hedaotao = np.nan,
    _mamh = np.nan,
    # _ten_mh = np.nan,
    _diem_hp = np.nan,
    _trangthai = np.nan,
):
    if subjects['mamh'].str.contains(_mamh).any() == True:
        _mamh_encode = subjects.loc[subjects['mamh'] == _mamh, 'monhoc_encode'].values[0]
        _nganh_bb = subjects.loc[subjects['mamh'] == _mamh, 'nganh_BB'].values[0]
        _nganh_bmav = subjects.loc[subjects['mamh'] == _mamh, 'nganh_BMAV'].values[0]
        _nganh_cnpm = subjects.loc[subjects['mamh'] == _mamh, 'nganh_CNPM'].values[0]
        _nganh_httt = subjects.loc[subjects['mamh'] == _mamh, 'nganh_HTTT'].values[0]
        _nganh_khmt = subjects.loc[subjects['mamh'] == _mamh, 'nganh_KHMT'].values[0]
        _nganh_ktmt = subjects.loc[subjects['mamh'] == _mamh, 'nganh_KTMT'].values[0]
        _nganh_kttt = subjects.loc[subjects['mamh'] == _mamh, 'nganh_KTTT'].values[0]
        _nganh_mmttt = subjects.loc[subjects['mamh'] == _mamh, 'nganh_MMT&TT'].values[0]
    else:
        _mamh_encode = ''
        _nganh_bb = -1
        _nganh_bmav = -1
        _nganh_cnpm = -1
        _nganh_httt = -1
        _nganh_khmt = -1
        _nganh_ktmt = -1
        _nganh_kttt = -1
        _nganh_mmttt = -1

    new_data_point = {
        'mssv': _mssv,
        'gioitinh': 1 if _gioitinh == 'Nam' else 0,
        # khoa
        'CNPM': 1 if _khoa == 'Công nghệ phần mềm' else 0,
        'HTTT': 1 if _khoa == 'Hệ thống thông tin' else 0,
        'KHMT': 1 if _khoa == 'Khoa học máy tính' else 0,
        'KTMT': 1 if _khoa == 'Kỹ thuật máy tính' else 0,
        'KTTT': 1 if _khoa == 'Kỹ thuật thông tin' else 0,
        'MMT&TT': 1 if _khoa == 'Mạng máy tính và truyền thông' else 0,

        # he dao tao
        'CLC': 1 if _hedaotao == 'Chất lượng cao' else 0,
        'CNTN': 1 if _hedaotao == 'Cử nhân tài năng' else 0,
        'CQUI': 1 if _hedaotao == 'Chính quy' else 0,
        'CTTT': 1 if _hedaotao == 'Chương trình tiên tiến' else 0,
        'KSTN': 1 if _hedaotao == 'Kỹ sư tài năng' else 0,

        # mon hoc
        'mamh': _mamh,
        # 'tenmh': _ten_mh,
        'monhoc_encode': _mamh_encode,
        'nganh_BB': _nganh_bb,
        'nganh_BMAV': _nganh_bmav,
        'nganh_CNPM': _nganh_cnpm,
        'nganh_HTTT': _nganh_httt,
        'nganh_KHMT': _nganh_khmt,
        'nganh_KTMT': _nganh_ktmt,
        'nganh_KTTT': _nganh_kttt,
        'nganh_MMT&TT': _nganh_mmttt,

        # diem hoc phan
        'diem_hp': _diem_hp,
        'trangthai': 1 if _trangthai == 'Bình thường' else 2 if _trangthai == 'Trả nợ' else 3, # 1: bình thường; 2: trả nợ; 3: cải thiện;
    }

    # new_data_point
    return new_data_point
