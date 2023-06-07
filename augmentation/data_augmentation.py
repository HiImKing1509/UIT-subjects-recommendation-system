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
    
# def filter(dataset, name_col: str, name_value: list):
#     data = dataset[dataset[name_col].isin(name_value)]
#     return data