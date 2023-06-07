import torch
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def get_attention_matrix(
        dataset=None,
        dataset_original=None,
        phobert=None,
        tokenizer=None,
        save_attn_matrix=None):
    # INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
    monhoc = dict(zip(dataset['mamh'], dataset['monhoc_encode']))
    lst_keys = list(monhoc.keys())
    monhoc['PH001'] = 'Nhập môn điện tử: Đây là môn học ở giai đoan kiến thức đai cương. Môn học này trình bày các khái niệm và ̣phương pháp cơ bản về điện tử. Giới thiệu về nguyên lý hoạt động của của các linh kiện điện tử cơ bản (điện trở, tụ điện, nguồn điện, transistor,….). Ứng dụng các linh kiện điện tử này vào các mạch điện thực tế. Môn học thuộc khoa Kĩ Thuật Máy Tính'
    # monhoc[lst_keys[19]] = 'Xác suất thống kê: Môn học này trình bày các khái niệm và phương pháp về: Lý thuyết xác suất (Không gian xác suất; Biến ngẫu nhiên; Hàm đặc trưng; Dãy các biến ngẫu nhiên; Các quy luật phân phối xác suất; Các định lý giới hạn phân phối xác suất) và Thống kê (Mẫu ngẫu nhiên; Ước lượng điểm và ước lượng khoảng; Kiểm định các giả thiết thống kê; Phân tích tương quan và hồi quy; Một số vấn đề về quá trình ngẫu nhiên). Ngoài ra, môn học này còn giới thiệu về cách thức nhận diện, phân tích và xử lý một vấn đề thực tế; xử lý các số liệu thống kê; để từ đó giúp cho người dùng đưa ra các suy luận phù hợp (nhằm hỗ trợ cho quá trình ra quyết định). Môn học bắt buộc'
    monhoc['SE102'] = 'Nhập môn phát triển game: Kỹ thuật lập trình DirectX để xây dựng các game 2D đơn giản như Tetris, Battle City, Mario, Contras... Nội dung bao gồm: Giới thiệu ngành game, kỹ thuật lập trình Windows dùng C++ và Windows SDK, kỹ thuật làm chuyển động và lập trình DirectX cơ bản, kỹ thuật làm việc với Sprite và xử lý thiết bị nhập, thảo luận về các kỹ thuật hỗ trợ như phép biến đổi, lập trình DirectSound, hiển thị chữ ..., bàn luận về Game Engine và cách xây dựng một game engine đơn giản. Môn học thuộc khoa Công Nghệ Phần Mềm'
    monhoc['SE101'] = 'Phương pháp mô hình hóa: Trình bày các kiến trúc, nền tảng về các phương pháp mô hình hóa thông tin, tri thức, biểu diễn vấn đề và lời giải, mô hình hóa hệ thống. Các phương pháp mô hình hóa và biểu diễn vấn đề như mô hình hóa và biểu diễn dữ liệu, mô hình hóa và biểu diễn quan hệ, mô hình hóa và biểu diễn tiến trình, mô hình hóa và biểu diễn tri thức (SDLC, JSD, SSM, OOA)... làm quen với các công cụ dùng để biểu diễn mô hình như công cụ CASE (upper và lower), các ngôn ngữ mô phỏng mô hình hóa như ngôn ngữ UML, VRML, ... nhằm hiện thực hóa một hệ thống. Học phần bao gồm dẫn nhập và giới thiệu những khái niệm về các mô hình đặc trung hiện nay; giới thiệu về phương pháp luận dùng cho mô hình hóa và giới thiệu cụ thể về các mô hình biểu diễn thông tin, dữ liệu, thời gian thực. Môn học thuộc khoa Công Nghệ Phần Mềm'
    monhoc['NT118'] = 'Phát triển ứng dụng trên thiết bị di động: Thiết kế giao diện cho các thiết bị di động trên Google Android. Xây dựng ứng dụng Native app lẫn cross platform app. Trong ứng dụng native app, sử dụng Java để thể hiện chương trình trên Android; Trong ứng dụng Native app, sinh viên sử dụng HTML, CSS, JavaScript để tạo ra một ứng dụng chuyển tiếp, liên lạc và swipe, hình ảnh động. Tích hợp các dịch vụ web hiện có từ Google và Amazon. Các nội dung bao gồm: Giới thiệu về tính toán di động khắp mọi nơi, tính toán cảm ngữ cảnh, giới thiệu hệ điều hành Android và các phương pháp lập trình trên Android. Các phương pháp lập trình nâng cao: đa luồng, đa hành vi, kết nối SQLite, Web Services; Khái niệm cross platform, thiết kế web di động, ứng dụng cho Điện thoại di động. Đánh dấu cho điện thoại di động. Web Apps di động và tính năng. Giới thiệu PhoneGap. Bản địa hóa ứng dụng. Khoa Mạng Máy Tính và Truyền Thông'
    monhoc['NT205'] = 'Tấn công mạng: Lý thuyết về những lỗ hổng bảo mật phổ biến tồn tại trong hệ thống mạng, hệ điều hành, ứng dụng; Các phương pháp tấn công dựa vào các lỗ hổng đã phát hiện; Các bước thực hiện tấn công chiếm quyền điều khiển hệ thống, thay đổi dữ liệu hay từ chối dịch vụ…; Xây dựng hệ thống phòng thủ ngăn chặn các cuộc tấn công. Đối với hệ Cử nhân tài năng: Trình bày chuyên sâu hơn về các giao thức mạng và việc tận dụng các lỗ hổng trong giao thức để tấn công; cách thức tấn công trên webserver cấu hình mạnh; các phương pháp tấn công ứng dụng web; cách thức tấn công và phòng chống lại các cuộc tấn công mạng trong tương lai; Bổ sung các bài tập nâng cao về việc sử dụng các công cụ crack password phức tạp và leo thang đặc quyền, xoá dấu vết, tấn công DDoS, cách thức điều khiển các zombie và xây dựng các mạng BotNet. Môn học thuộc khoa Mạng Máy Tính và Truyền Thông'
    monhoc['SE334'] = 'Các phương pháp lập trình: Học phần này trình bày các kiến trúc, nền tảng về các phương pháp, kỹ thuật lập trình thường dùng khi thiết kế và xây dựng một chương trình máy tính. Ngôn ngữ C++, Java, các thư viện hỗ trợ trong lập trình song song. Học phần gồm: Giới thiệu các kỹ thuật và các nguyên lý cơ bản của lập trình; Giới thiệu cụ thể về các phương pháp và kỹ thuật lập trình như đệ qui, tối ưu mã chương trình, phương pháp lập trình cấu trúc, lập trình hướng đối tượng, lập trình đa nhiệm, song song; Giới thiệu kỹ thuật thiết kế kiến trúc và giao diện chương trình. Môn học thuộc khoa Công Nghệ Phần Mềm'
    monhoc['NT505'] = 'Khóa luận tốt nghiệp: Một công trình nghiên cứu khoa học dành cho sinh viên. Trong khóa luận, sinh viên nêu rõ những vấn đề do sinh viên thực hiện được dưới sự hướng dẫn của giảng viên như: ứng dụng, quy trình hoạt động, hệ thống triển khai. Ngoài ra khóa luận cần có những đánh giá, phương hướng phát triển tiếp theo của đề tài. Trong khóa luận nêu rõ kết quả thực hiện của sinh viên, đây là thành phần quan trọng nhất của khóa luận. Đề tài khóa luận tốt nghiệp là một đề tài được nghiên cứu và triển khai chuyên sâu gắn với yêu cầu thực tế cho thấy khả năng nghiên cứu và làm việc độc lập của sinh viên. Trong khóa luận tốt nghiệp, cần xác định rõ những vấn đề do sinh viên thực hiện được dưới sự hướng dẫn của giảng viên như: ứng dụng, quy trình hoạt động, hệ thống triển khai, tính mới của nghiên cứu. Ngoài ra khóa luận cần có những đánh giá, phương hướng phát triển tiếp theo của đề tài. Trong khóa luận cần nêu rõ kết quả nghiên cứu của sinh viên. Môn học thuộc khoa Mạng Máy Tính và Truyền Thông'
    monhoc['NT211'] = 'An ninh nhân sự, định danh và chứng thực: Khái niệm căn bản về định danh, xác thực và ứng dụng của chúng trong quản lý truy cập. Môn học trang bị cho sinh viên ngành an ninh thông tin: Khái niệm nền tảng về an ninh liên quan tới con người; Kiến thức về định danh cùng các công nghệ định danh hiện đại; Kiến thức về xác thực và những công nghệ liên quan đến xác thực; Ứng dụng định danh và xác thực trong hệ thống CNTT. Đối với hệ Cử nhân tài năng: Trình bày chuyên sâu hơn các nội dung Sinh trắc và các phương pháp chính; Quản lý tài khoản với Token; Quản lý tài khoản liên hợp; Tấn công thẻ thông minh; Bổ sung các bài tập nâng cao ở các nội dung trình bày chuyên sâu trên và các nội dung Bẻ mật khẩu phức tạp; Thiết kế, xây dựng và triển khai hệ thống cấp chứng chỉ số ở quy mô lớn. Môn học thuộc khoa Mạng Máy Tính và Truyền Thông'
    monhoc['SE220'] = 'Thiết kế Game: Kiến thức, kỹ năng lĩnh vực thiết kế game. Lý thuyết về diễn biến tâm lý người chơi, bản chất của game, tại sao game hấp dẫn. Kỹ thuật thiết kế game, lịch sử trong thiết kế game. Tập trung vào thiết kế giao diện game như cách xây dựng menu, bố trí các thành phần giao diện, biểu tượng, thiết kế HUD. Bàn về thiết kế cảnh chơi như cách đặt thử thách, xây dựng bối cảnh, tạo hồn cho cảnh chơi... Môn học thuộc khoa Công Nghệ Phần Mềm'
    monhoc['SE320'] = 'Lập trình đồ họa 3 chiều với Direct3D: Lập trình ứng dụng đồ họa 3 chiều và hướng dẫn sử dụng bộ thư viện đồ họa DirectX để xây dựng ứng dụng. Trình bày về cơ sở toán học ứng dụng trong đồ họa 3 chiều và quy trình dựng hình 3 chiều, trình bày về Direct3D bao gồm các vấn đề đi từ cơ bản đến nâng cao, ứng dụng các kiến thức đã học vào xây dựng trò chơi Tetris 3D. Kết thúc khóa học, sinh viên sẽ có khả năng tự thiết kế và lập trình ứng dụng đồ họa 3 chiều đơn giản trên môi trường Windows. Môn học thuộc khoa Công Nghệ Phần Mềm'
    monhoc['NT534'] = 'An toàn mạng máy tính nâng cao: Cách phòng chống tấn công từ chối dịch vụ, các hoạt động ngầm trên Internet, bàn luận về các giải pháp kĩ thuật trong việc ngăn chặn cũng như đối phó với ngăn chặn trong việc quản lý truy cập trên Internet. Ngoài ra, môn này cũng đề cập các nguy cơ từ các loại mã độc tinh vi đối với an toàn mạng. Đối với hệ tài năng: Môn an toàn mạng đề cập các chủ đề căn bản của an toàn mạng. Môn này đề cập đến các vấn đề chuyên sâu hơn ví dụ như là làm thế nào để phòng chống tấn công từ chối dịch vụ, các hoạt động ngầm trên Internet, bàn luận về các giải pháp kĩ thuật trong việc ngăn chặn cũng như đối phó với ngăn chặn trong việc quản lý truy cập trên Internet. Ngoài ra, môn này cũng đề cập các nguy cơ từ các loại mã độc tinh vi đối với an toàn mạng. Cuối cùng, các kỹ thuật client side, server-side honeypot cũng được giới thiệu để nghiên cứu, thu thập mã độc. Môn học thuộc khoa Mạng Máy Tính và Truyền Thông'
    monhoc['IE307'] = 'Công nghệ lập trình đa nền tảng cho ứng dụng di động: Môn học trình bày nguyên lý cơ bản của các Framework về lập trình di động đa nền tảng (React Native, PhoneGap, Xamarin...) và đặc biệt là Xamarin Framework. Cung cấp các Controls cơ bản của Xamarin, và áp dụng để xây dựng ứng dụng đa nền tảng: Label, Entry, Button, Image, Switch, ListView, DatePicker, TimePicker. Bên cạnh đó, môn học còn cung cấp thêm các vấn đề nâng cao của Xamarin, để tiếp tục tự nghiên cứu sử dụng về sau của Camera, Notification, Google Map APIs, Grial, RESTful API, Syncfusion... Môn học trang bị kỹ năng làm việc nhóm theo môi trường doanh nghiệp, đọc hiểu yêu cầu của khách hàng về ứng dụng di động, Phân tích & Thiết kế các ứng dụng di động để xây dựng một ứng dụng di động đa nền tảng cơ bản chạy trên iOS, Android & Windows Phone theo yêu cầu. Môn học thuộc khoa Kĩ Thuật Thông Tin'
    lst_input_ids = [torch.tensor([tokenizer.encode(monhoc[key])]) for key in lst_keys]

    with torch.no_grad():
        features = {}
        for index, input_ids in enumerate(lst_input_ids):
            features[lst_keys[index]] = phobert(input_ids).pooler_output

    lst_supervision_extra_dic = {
        'nganh_CNPM': 'Cung cấp sự hiểu biết các đặc trưng chính của phần mềm, khái niệm chu trình phần mềm, các hoạt động kỹ thuật, cung cấp kiến thức thực nghiệm về chọn lựa kỹ thuật, công cụ, mô hình chu trình dự án, các kiến thức độ quan trọng đảm bảo chất lượng (quality assurance), quản lý dự án trong phát triển phần mềm',
        'nganh_HTTT': 'Nghiên cứu các hệ thống thông tin quản trị doanh nghiệp, ngân hàng như ERP, Supply Chain Management; Nghiên cứu các ứng dụng xây dựng hệ thống thông tin phục vụ Thương Mại Điện Tử; Phát triển các nghiên cứu nhằm tăng cường khai thác tri thức từ CSDL, quản trị các kho dữ liệu lớn, tìm kiếm thông tin trên web, tìm kiếm ngữ nghĩa, mạng xã hội; Phát triển các nghiên cứu liên ngành giữa tin học và các ngành khoa học khác như: xử lý ngôn ngữ tự nhiên, sinh học, hoá học, môi trường, ...',
        'nganh_KHMT': 'Đào tạo bài bản về Trí tuệ nhân tạo (Artificial Intelligence - AI) đáp ứng nhu cầu về nghiên cứu, xây dựng và phát triển các sản phẩm, giải pháp thông minh phục vụ cho cuộc sống. Chương trình đào tạo của Khoa cung cấp cho sinh viên nhiều lựa chọn theo các định hướng nghề nghiệp như Trí tuệ Nhân tạo (AI), Thị giác Máy tính (Computer Vision), Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)…. Với các kiến thức nền tảng sinh viên hoàn toàn có thể tham gia nghiên cứu và phát triển các ứng dụng thông minh như: hệ thống nhận diện khuôn mặt (Face Recognition System), hệ thống Chatbot, hệ thống tìm kiếm – truy vấn thông tin (Retrieval System) ...',
        'nganh_KTMT': 'Lập trình các phần mềm nhúng trên các thiết bị di động (Smartphone, Tablet, iphone, ipad, ...), các vi xử lý-vi điều khiển trong các hệ thống công nghiệp, xe ô tô, điện gia dụng, ngôi nhà thông minh,… (Chuyên ngành hệ thống nhúng và IoT); thiết kế mạch điện - điện tử, mạch điều khiển trong công nghiệp, vi mạch, chip,... (Chuyên ngành thiết kế vi mạch)',
        'nganh_KTTT': 'Thiết kế, xây dựng và quản lý các dự án nghiên cứu và ứng dụng CNTT, chủ yếu trong lĩnh vực dữ liệu không gian-thời gian (địa lý, tài nguyên, môi trường, viễn thám. . .). Tập trung vào những ứng dụng về GIS trên thiết bị di động và trao đổi dữ liệu với máy chủ; Vận hành, quản lý, giám sát; phân tích và phát triển các ứng dụng CNTT tại các doanh nghiệp; Khai thác dữ liệu và thông tin ứng dụng cho các doanh nghiệp trong vấn đề phân tích định lượng; Xây dựng, phát triển các ứng dụng về lãnh vực truyền thông xã hội và công nghệ Web',
        'nganh_MMT&TT': 'Quản trị mạng và hệ thống tại các ngân hàng, các trung tâm dữ liệu, các nhà cung cấp dịch vụ Internet (ISP); Thiết kế mạng chuyên nghiệp: xây dựng các mạng máy tính an toàn, hiệu quả cho các đơn vị có yêu cầu; Phát triển phần mềm mạng; Phát triển phần mềm mạng; Xây dựng và phát triển các ứng dụng truyền thông: VoIP, hội nghị truyền hình',
    }
    lst_supervision_keys = list(lst_supervision_extra_dic.keys())
    lst_supervision_token = [torch.tensor([tokenizer.encode(lst_supervision_extra_dic[key])]) for key in ['nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT']]
    with torch.no_grad():
        features_supervision = {}
        for index, input_ids in enumerate(lst_supervision_token):
            features_supervision[lst_supervision_keys[index]] = phobert(input_ids).pooler_output

    data_temp = dataset_original[['mamh', 'nganh_BB', 'nganh_BMAV', 'nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT']]

    conditions = [
        dataset_original['nganh_BB'] == 1,
        dataset_original['nganh_BMAV'] == 1,
        dataset_original['nganh_CNPM'] == 1,
        dataset_original['nganh_HTTT'] == 1,
        dataset_original['nganh_KHMT'] == 1,
        dataset_original['nganh_KTMT'] == 1,
        dataset_original['nganh_KTTT'] == 1,
        dataset_original['nganh_MMT&TT'] == 1   
    ]
    values = ['nganh_BB', 'nganh_BMAV', 'nganh_CNPM', 'nganh_HTTT', 'nganh_KHMT', 'nganh_KTMT', 'nganh_KTTT', 'nganh_MMT&TT']
    data_temp['nganh'] = np.select(conditions, values, default=0)

    data_temp = data_temp.drop_duplicates()

    for index, key in enumerate(features.keys()):
        val = data_temp.loc[data_temp['mamh'] == key, 'nganh'].values[0]
        if val in ['nganh_BB', 'nganh_BMAV']:
            continue
        else:
            features[key] = torch.cat((features[key], features_supervision[val]), 1)
    data = [{
            'mh1': i, 
            'mh2': j, 
            'similarity': cosine_similarity(features[i], features[j]).item()
        } \
        for i in features.keys() \
        for j in features.keys()]
    attention_monhoc = pd.DataFrame(data)
    attention_matrix = attention_monhoc.pivot_table(index=["mh1"], columns=["mh2"], values="similarity")
    if save_attn_matrix != None:
        attention_matrix.to_csv(f'./data/{save_attn_matrix}.csv')
    return attention_matrix