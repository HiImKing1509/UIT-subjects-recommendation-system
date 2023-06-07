def input_prediction(
    _mssv = None,
    _gioitinh = None,
    _khoa = None,
    _hedaotao = None,
    _mamh = None,
    _diem_hp = None,
):
    new_data_point = {
        'mssv': _mssv,
        'gioitinh': 1 if _gioitinh == 'Nam' else 0,
        # khoa
        'CNPM': 1 if _khoa == 'CNPM' else 0,
        'HTTT': 1 if _khoa == 'HTTT' else 0,
        'KHMT': 1 if _khoa == 'KHMT' else 0,
        'KTMT': 1 if _khoa == 'KTMT' else 0,
        'KTTT': 1 if _khoa == 'KTTT' else 0,
        'MMT&TT': 1 if _khoa == 'MMT&TT' else 0,

        # he dao tao
        'CLC': 1 if _hedaotao == 'CLC' else 0,
        'CNTN': 1 if _hedaotao == 'CNTN' else 0,
        'CQUI': 1 if _hedaotao == 'CQUI' else 0,
        'CTTT': 1 if _hedaotao == 'CTTT' else 0,
        'KSTN': 1 if _hedaotao == 'KSTN' else 0,

        # diem hoc phan
        'mamh': _mamh,
        'diem_hp': _diem_hp,
    }

    # new_data_point
    return new_data_point