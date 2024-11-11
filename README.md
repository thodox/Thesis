# Thesis: Research on Adversarial Robustness of Machine Learning-Based Intrusion Detection Systems


Research, design, and develop intrusion detection systems using Machine Learning, Deep Learning, and Ensemble Learning.
Utilize Multimodal Learning in combination with GANs to generate adversarial traffic. Additionally, apply adversarial attack algorithms such as FGSM, ZOO, DeepFool, and C&W to create adversarial traffic.
This adversarial traffic is then tested on the previously developed intrusion detection systems (Transferability).
Train the intrusion detection systems with defense strategies and test different adversarial samples on the trained systems to evaluate the models 'Robustness’.


## Setting:
1.	Ngôn ngữ lập trình: Python 3.9
2.	Các thư viện cần thiết: pandas , numpy , scikit-learn,  tensorflow,  matplotlib ,seaborn.
3.	Xử lí dữ liệu: 
-	Dữ liệu từ tập CICIDS 2017 (2,830,743 bản ghi, 79 đặc trưng) được chuẩn hóa qua các bước:
-	Loại bỏ 10 đặc trưng không cần thiết (trùng lặp, không ảnh hưởng quá trình huấn luyện).
-	Thay thế các giá trị vô hạn bằng NaN và loại bỏ các bản ghi chứa NaN.
-	Chuyển đổi các cột Flow Bytes/s và Flow Packets/s sang dạng số.
-	Mã hóa nhãn: gồm 0(bình thường), 1( độc hại).
-	Chuẩn hóa dữ liệu với MinMaxScaler và chia thành hai tập train/test (80/20)
4.	Huấn luyện mô hình trên dữ liệu gốc 
-	GNB, DT (độ sâu 3), LR: Sử dụng thư viện hỗ trợ trong Python.
-	DNN: 6 lớp Dense (512 → 1), Dropout chống overfitting, 243,537 tham số, train 20 epoch.
-	LSTM: 4 lớp LSTM (128 → 512), 2 triệu tham số, train 20 epoch.
-	Ensemble ML: Kết hợp GNB, DT, LR (hard-voting).
-	Ensemble DL: Kết hợp DNN và LSTM (soft-voting).
5.	Tạo mẫu tấn công đối kháng
-	Tấn công vào mô hình học sâu là DNN đã được huấn luyện trên dữ liệu gốc.
-	 FGSM: tạo mẫu dựa trên độ nhiễu thêm vào epsilon.
-	ZOO: sử dụng tối ưu hóa bậc không để xấp xỉ gradient và tạo mẫu đối kháng.
-	Deepfool: tìm mẫu đối kháng gần nhất bằng cách lặp qua và dịch chuyển ranh giới  các dự đoán tuyến tính hóa xung quanh điểm dữ liệu ban đầu.
-	CW: tạo mẫu đối kháng thông qua quá trình tối ưu hóa nhưng phải tối thiểu hóa khoảng cách giữa mẫu gốc và mẫu đối kháng mà vẫn khiến cho mô hình phân loại sai. 
6.	Huấn luyện phòng thủ đối kháng
-	Nó giống như bước 4, chỉ thay đổi là trộn thêm các mẫu tấn công đối kháng vào trong quá trình huấn luyện.


## Kết quả:
HIỆU SUẤT CỦA CÁC DETECTOR TRƯỚC DỮ LIỆU CICIDS 2017				
				
Detector	Accuracy	Precision	Recall	F1
DT	99.85	99.58	99.52	99.55
KNN	99.89	99.53	99.77	99.65
LR	98.21	94.36	94.58	94.47
DNN	99.58	99.41	97.96	98.68
LSTM	99.73	99.49	98.86	99.17
Ensemble ML	99.9	99.64	99.73	99.68
Ensemble DL	99.68	99.6	98.41	99
![image](https://github.com/user-attachments/assets/ed73f7b1-a42d-4077-9ac5-864f57600464)
