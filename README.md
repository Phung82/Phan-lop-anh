ĐỀ TÀI:
-------
TÌM HIỂU NGÔN NGỮ PYTHON, THƯ VIỆN PYTORCH VÀ VIẾT ỨNG DỤNG PHÂN LỚP HÌNH ẢNH (BIRD, CAT, FROG, HORSE)
-------
MỞ ĐẦU
-------
1.	Lý do chọn đề tài
- Deep Learning là một thuật toán dựa trên một số ý tưởng não bộ tới việc tiếp thu nhiều tầng biểu đạt, cả cụ thể lẫn trừu tượng, qua đó làm rõ nghĩa của các loại dữ liệu. Deep Learning được ứng dụng nhiều trong nhận diện hình ảnh, nhận diện giọng nói, xử lý ngôn ngữ tự nhiên.
- Hiện nay rất nhiều các bài toán nhận dạng, phân loại sử dụng deep learning để giải quyết do deep learning có thể giải quyết các bài toán với số lượng, kích thước đầu vào lớn với hiệu năng cũng như độ chính xác vượt trội so với các phương pháp phân lớp truyền thống.
- Convolutional Neutal Network (CNNs – Mạng nơ-ron tích chập) là một trong những mô hình Deep Learning tiên tiến giúp cho chúng ta xây dựng được nhưng hệ thống thông minh với độ chính xác cao. Trong bài báo cáo này, chúng em tập trung nghiên cứu về “Mạng Nơ-ron nhân tạo” cũng như ý tưởng phân lớp ảnh dựa trên mô hình CNNs (Image Classification). Và áp dụng để xây dựng ứng dụng phân lớp ảnh “bird”, “cat”, “frog” và “horse”.
2.	Cấu trúc đồ án
- Chương 1: Tìm hiểu ngôn ngữ lập trình Python
- Chương 2: Tìm hiểu mạng nơ-ron và thư viện Pytorch
- Chương 3: Xây dựng ứng dụng
------------------------------------------------------------------------------

CHƯƠNG 1 VÀ CHƯƠNG 2 ĐƯỢC TRÌNH BÀI TRONG BÁO CÁO
-------
CHƯƠNG 3: XÂY DỰNG ỨNG DỤNG
-------
3.1 Nêu bài toán
    - Nhiệm vụ của bài toán phân lớp là cần tìm một mô hình phân lớp để có thể xác định được dữ liệu được truyền vào là thuộc phân lớp nào trong 4 lớp “Bird”, “Cat”, “Frog”, “Horse”.
 
<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/01.jpg" width="600" height="400"/>
<p>Hình 3.1 Sơ đồ mô phỏng 4 lớp “Bird”, “Cat”, “Frog”, “Horse”</p>

3.2 Chuẩn bị dữ liệu
    -   Bộ dữ liệu CIFAR-10 bao gồm 60000 hình ảnh màu 32x32 trong 10 lớp, với 6000 hình ảnh cho mỗi lớp. Có 50000 hình ảnh dùng để đào tạo và 10000 hình ảnh dùng cho việc kiểm tra.
    -   Tập dữ liệu thô được chia thành 5 lô đào tạo và một lô thử nghiệm, mỗi lô có 10000 hình ảnh. Lô thử nghiệm chứa chính xác 1000 hình ảnh được chọn ngẫu nhiên từ mỗi lớp. Các lô đào tạo chứa các hình ảnh còn lại theo thứ tự ngẫu nhiên, nhưng một số lô đào tạo có thể chứa nhiều hình ảnh từ lớp này hơn lớp khác. Giữa chúng, các lô đào tạo chứa chính xác 5000 hình ảnh từ mỗi lớp.
    -   Để chuẩn bị dữ liệu cho việc xây dựng ứng dùng thì, từ bộ dữ liệu CIFAR10 thô cần phải được chuyển đổi về bộ dữ liệu hình ảnh và được phân theo từng lớp với 2 nhóm dùng để huấn luyện và kiểm tra.
    -   Ở đây 4 bộ dữ liệu cần dung trong 10 bộ dữ liệu của CIFAR10 là: “Bird”, “Cat”, “Frog”, “Horse”.
 
<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/02.png" width="600" height="400" />
<p>Hình 3.4 Cây thư mục của bộ dữ liệu</p>

<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/03.png" width="600" height="400" />
<p>Hình 3.5 Hình ảnh dùng cho việc kiểm tra của bộ dữ liệu về “Bird”</p>

3.3.	Phương pháp lựa chọn đề tài
    -   CNN (Convolutional Neural Network) là một Structure rất phổ biến và quen thuộc trong Deep Learning. CNN được ứng dụng nhiều trong Computer Vision, Recommender System, Natural Language Processing, ...
    -   Với Convolutional Neural Network, đây là một deep neural network architectures. Hiểu đơn giản, nó cũng chính là một dạng Artificial Neural Network, một Multiplayer Perceptron nhưng mang thêm 1 vài cải tiến, đó là Convolution và Pooling.
    
<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/04.png" width="800" height="250" />
<p>Hình 3.6 Mô hình Convolutional Neural Network</p>

3.4.	Giao diện và các chức năng của ứng dụng
 -------

<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/05.png" width="500" height="400" />
<p>Hình 3.7 Giao diện màn hình chính của ứng dụng</p>
 
 
<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/06.png" width="900" height="350" />
<p>Hình 3.8 Mô hình mạng nơ-ron CNN</p>


<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/07.png" width="420" height="400" />
<p>Hình 3.9 Kết quả thông số huấn luyện</p>

<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/08.png" width="600" height="400" />
<p>Hình 3.11 Giao diện diện huấn luyện mô hình mạng</p>

<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/09.jpg" width="400" height="400"/>
<p>Hình 3.12 Kết quả phân lớp “cat” </p>

<img   src="https://github.com/Phung82/Phan-lop-anh/blob/main/Screenshots/10.jpg" width="400" height="400"/>
<p>Hình 3.12 Kết quả phân lớp “horse”</p>


3.5.	Đánh giá
    -   Ưu điểm
            +   Ứng dụng phân loại hình ảnh dựa trên phương pháp CNNs đã đạt được một số điểm như sau:
            +   Nắm được vấn đề cơ bản về deep learning và mạng nơ-ron.
            +   Sử dụng tốt các thư viện hỗ trợ như Pytorch, tkinter, pillow để xử lý hình ảnh và dữ liệu, cũng như sử dụng ngôn ngữ lập trình python.
            +   Nắm được các đặt điểm của bài toán phân lớp.
    -   Khuyết điểm
            +   Tốc độ huấn luyện tập dữ liệu còn hạn chế.
            +   Cấu trúc ứng dụng còn đơn giản.
3.6.	Hướng phát triển của bài toán
    -   Nâng cao hiệu quả của chương trình, mở rộng số lượng các lớp nhận dạng nhiều hơn với tập dữ liệu CIFAR100.
    -   Phát triển chương trình thành module phần cứng. Có khả năng tương thích với các thiết bị quan sát như camera…
    -   Nghiên cứu theo hướng một ứng dụng cụ thể như: nhận diện phương tiện giao thông, nhận dạng các loại đồ vật, hàng hóa.

-----------------------------------------------------------------------------------------------------
TÀI LIỆU THAM KHẢO
-------

    -   [ 1 ]	Trang chủ của Pytorch: https://pytorch.org/
    -   [ 2 ]	Tài liệu hỗ trợ của Pytorch: https://pytorch.org/tutorials/
    -   [ 3 ]	Blog Machine Learning Cơ Bản của T.s Vũ Hữu Tiệp: https://machinelearningcoban.com/
    -   [ 4 ]	MNIST Handwritten Digit Recognition in PyTorch:    https://nextjournal.com/
    -   [ 5 ]	Y. LeCun and Y. Bengio.“Convolutional networks for images, speech, and time series.” In M. A. Arbib, editor, The Handbook of Brain Theory and Neural Networks. MIT Press, 1995.
    -   [ 6 ]	Kenji Doi “Convert CIFAR-10 and CIFAR-100 datasets into PNG images”
    -   [ 7 ]	“Convert CIFAR-10 and CIFAR-100 datasets into PNG images” by Dan Clark, KDnuggets.

