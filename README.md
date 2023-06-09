> <a href="https://github.com/HiImKing1509/uit_subjects_recommendation_system/blob/master/README_en.md">English here</a>

<div align="center">

  # HỆ THỐNG KHUYẾN NGHỊ MÔN HỌC - TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN
  ### SUBJECTS RECOMMENDATION SYSTEM - UNIVERSITY OF INFORMATION TECHNOLOGY
</div>

![image](https://github.com/HiImKing1509/uit_subjects_recommendation_system/assets/84212036/8f6a314a-c8eb-4d73-8d94-90803a005665)

* [Động lực nghiên cứu](#dlnc)
* [Đặc trưng](#dt)
* [Hướng dẫn](#hdsd)
* [Người đóng góp](#ndg)
* [Kết quả thực nghiệm](#kqua)

<a name="dlnc"></a>
### Động lực nghiên cứu
___
Nắm được tầm quan trọng của việc chọn lựa môn học ở trường đại học, đặc biệt là trong thời đại công nghệ 4.0 hiện nay với sự phát triển vượt bậc các mô hình học máy, chúng tôi tiến hành xây dựng một hệ thống khuyến nghị các môn học liên quan dựa trên mô hình học máy, nhằm cải thiện kết quả học tập của sinh viên trường Đại học Công nghệ Thông tin.

<a name="dt"></a>
### Đặc trưng
___
**Đầu vào**:
- Thông tin sinh viên:
    1. Giới tính: *Nam/Nữ*
    2. Khoa theo học: *Khoa học máy tính/ Công nghệ phần mềm/ ...*
    3. Hệ đào tạo: *Chất lượng cao/ Chính quy/ ...*
    4. Điểm học phần các môn IT001, IT002, IT003, IT004, IT005, IT006, IT007 
- Môn học truy vấn: Ví dụ CS106, NT101, ...
- Điểm số môn học truy vấn

**Đầu ra**: Danh sách các môn học khuyến nghị

![Alt Text](./images/input_output.png)

<a name="hdsd"></a>
### Mô tả
___

Chúng tôi cung cấp mã nguồn mở tại <a href=https://github.com/HiImKing1509/uit_subjects_recommendation_system>repository</a>, hỗ trợ sử dụng cho 2 đối tượng chính:

- **<a href="https://github.com/HiImKing1509/uit_subjects_recommendation_system/blob/master/README_implementation.md">Nhà phát triển</a>**: Cài đặt môi trường, thực hiện nghiên cứu và điều chỉnh mã nguồn với mục tiêu cải thiện chất lượng hệ thống khuyến nghị. 
- **<a href="https://github.com/HiImKing1509/uit_subjects_recommendation_system/blob/master/README_inference.md">Người sử dụng</a>**: Nhu cầu sử dụng hệ thống và nhận khuyến nghị môn học.

<a name="kqua"></a>
### Kết quả thực nghiệm
___
| Threshold get similarity Subjects    | Threshold get similarity Students          | Score |
|:-------:|:-------------:|:----------------------:|
|15 |	1 |	73.16 |
|15	| 2	| 78.19 |
|15	| 3	| 79.22 |
|15	| 4	| 79.02 |
|15	| 5	| 78.24 |
|15	| 6	| 77.37 |
|15	| 7	| 76.37 |
|15	| 8	| 75.42 |
|15	| 9	| 74.61 |
|15	| 10 | 73.75 |
|16	| 1	| 73.02 |
|16	| 2	| 78.12 |
|16	| 3	| 79.20 |
|16	| 4	| 79.07 |
|16	| 5	| 78.30 |
|16	| 6	| 77.45 |
|16	| 7	| 76.47 | 
|16	| 8	| 75.53 |
|16	| 9	| 74.73 |
|16	| 10 | 73.88 |
|17	| 1	| 72.97 |
|17	| 2	| 78.19 |
|17	| 3	| 79.34 |
|17	| 4	| 79.27 |
|17	| 5	| 79.53 |
|17	| 6	| 77.67 |
|17	| 7	| 76.71 |
|17	| 8	| 75.79 |
|17	| 9	| 74.98 |
|17	| 10 | 74.15 |
|18	| 1 | 72.95 |
|18	| 2 | 78.29 |
|18	| 3 | 79.61 |
|18	| 4 | 79.67 |
|18	| 5 | 79.07 |
|18	| 6 | 78.31 |
|18	| 7 | 77.41 |
|18	| 8 | 76.57 |
|18	| 9 | 75.80 |
|18	| 10 | 74.96 |
|19	| 1 | 73.10 |
|19	| 2 | 78.48 |
|19	| 3 | 79.88 |
|19	| 4 | 79.97 |
|19	| 5 | 79.38 |
|19	| 6 | 78.67 |
|19	| 7 | 77.77 |
|19	| 8 | 76.95 |
|19	| 9 | 76.18 |
|19	| 10 | 75.35 |
|20	| 1 | 73.03 |
|20	| 2 | 78.45 |
|20	| 3 | 79.91 |
|20	| 4 | 80.06 |
|20	| 5 | 79.57 |
|20	| 6 | 78.92 |
|20	| 7 | 78.10 |
|20	| 8 | 77.34 |
|20	| 9 | 76.63 |
|20	| 10 | 75.85 |
|21	| 1 | 73.15 |
|21	| 2 | 78.63 |
|21	| 3 | 80.17 |
|21	| 4 | 80.43 |
|21	| 5 | 80.04 |
|21	| 6 | 79.49 |
|21	| 7 | 78.74 |
|21	| 8 | 78.05 |
|21	| 9 | 77.40 |
|21	| 10 | 76.66 |
|22	| 1 | 73.04 |
|22	| 2 | 78.65 |
|22	| 3 | 80.24 |
|22	| 4 | 80.54 |
|22	| 5 | 80.21 |
|22	| 6 | 79.67 |
|22	| 7 | 78.93 |
|22	| 8 | 78.24 |
|22	| 9 | 77.58 |
|22	| 10 | 76.84 |




<a name="ndg"></a>
### Đóng góp
___
Hệ thống được hoàn thành dựa trên những đóng góp tích cực của các thành viên:

| STT    | ID          | Tên thành viên              | Github                                               | Email                   |
| ------ |:-------------:| ----------------------:|-----------------------------------------------------:|-------------------------:
| 1      | 20521494      | Huỳnh Viết Tuấn Kiệt |[hiimking1509](https://github.com/HiImKing1509)          |20521494@gm.uit.edu.vn   |
| 2      | 20522087      | Nguyễn Nhật Trường |[truong11062002](https://github.com/truong11062002)          |20522087@gm.uit.edu.vn   |
| 3      | 20520276      | Nguyễn Đức Anh Phúc |[PhucNDA](https://github.com/PhucNDA)          |20520276@gm.uit.edu.vn   |
| 4      | 20520355      | Lê Thị Phương Vy |[Ceci-june](https://github.com/Ceci-june)          |20520355@gm.uit.edu.vn   |
| 5      | 20520309      | Lại Chí Thiện |[laichithien](https://github.com/laichithien)          |20520309@gm.uit.edu.vn   |
