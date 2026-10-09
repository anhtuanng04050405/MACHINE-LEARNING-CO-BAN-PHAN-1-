**MACHINE LEARNING CƠ BẢN**

**Phần 1: Khái niệm và code pipeline machine learning cơ bản**

**1. Phân biệt Supervised Learning và Unsupervised Learning**

`	`1. 1. Supervised Learning (học có giám sát) là gì?

`	`**Định nghĩa:** Học có giám sát là huấn luyện mô hình với dự liệu đã có nhãn.

`	`Trong học có giám sát (supervised learning), việc xác định đầu vào và đầu ra (nhãn) của mô hình là bước cực kì quan trọng.

**+ Đầu vào (Input - Features):** Là các đặc trưng (features) mô tả dữ liệu.

**+ Đầu ra (Output - Label):** Là giá trị mô hình cần dự đoán, được gọi là nhãn (label).

`	`1. 2. Unsupervised Learning (học không có giám sát) là gì?

`	`**Định nghĩa:** Học không giám gát là dữ liệu không có nhán (Unlabeled Data). Mô hình (model) tự tìm ra cấu trúc ẩn bên trong.

`	`**Ví dụ:** 

\+ Bạn có 1 đống ảnh mèo, chó, gấu, ... Nhưng không biết ảnh nào là gì.

\+ Mô hình (model) tự động nhóm chúng thành từng cụm dựa vào đặc điểm chung.

`	`**Các bài toán chính của Học không có giám sát:**
**\
`	`+ Phân cụm (Clustering): Nhóm các đối tượng tương tự nhau.

`	`+ Giảm chiều dữ liệu (Dimension Reduction): Đơn giản hóa dữ liệu mà vẫn giữ thông tin quan trọng.

`	`+ Phát hiện bất thường: Phát hiện gian lận thẻ tín dụng, phát hiện lỗi trong máy móc.

`	`1. 3. So sánh:

|**Tiêu chí**|**Supervised Learning**|**Unsupervised Learning**|
| :- | :- | :- |
|Có nhãn dữ liệu|Có|Không|
|Mục tiêu|Dự đoán đầu ra|Tìm cấu trúc/ cụm|
|Ví dụ phổ biến|<p>+ Phân loại</p><p>+ Hồi quy</p>|<p>+ Phân cụm</p><p>+ Phát hiện bất thường</p>|

**2. Pipeline cơ bản trong một bài toán Machine Learning**

**Bước 1:** Thu thập dữ liệu (Data Collection)

Dữ liệu có thể từ nhiều nguồn như file CSV, cơ sở dữ liệu, API hoặc tự mình thu thập từ web.

**Bước 2:** Tiền xử lý dữ liệu (Data Preprocessing)

**+ Làm sạch & Chuẩn hóa dữ liệu thô:** Xử lý thiếu, loại bỏ nhiễu, chuẩn hóa (ví dụ: StandardScaler)

**+ Chia dữ liệu thành 3 phần quan trọng:**\
`	`**1) Tập huấn luyện (Train Set):** Dùng để mô hình học.

`	`**2) Tập đánh giá (Validation/ Dev Set):** Dùng để tinh chỉnh mô hình trong quá trình huấn luyện (Chọn siêu tham số tốt nhất)

`	`**3) Tập kiểm tra (Test Set):** Dùng để đánh giá khách quan hiệu suất cuối cùng của mô hình (sau khi đã tinh chỉnh)

**Lưu ý quan trọng:** ***Tuyệt đối không dùng Test Set để tinh chỉnh mô hình***

**Bước 3:** Huấn luyện mô hình (Model Training)

\+ Chọn thuật toán ML phù hợp. Ví dụ: Decision Tree, KNN, Logistic Regression.

\+ Huấn luyện trên tập Train.

\+ Mô hình học các mối quan hệ và đặc trưng từ dữ liệu

**Bước 4:** Đánh giá mô hình (Model Evaluation)

\+ Dùng tập Test để đo lường hiệu suất.

\+ Các chỉ số:\
`	`1) Độ chính xác (Accuracy)

`	`2) Ma trận nhầm lẫn (Confusion Matrix)

\+ Cần chú ý:

\1) Underfitting: Mô hình học không đủ\.

\2) Overfitting: Mô hình học thuộc lòng dữ liệu\.

**3. Code Pipeline Machine Learning Cơ Bản**

`	`3. 1. Thư viện Scikit-learn trong Python

`	`Scikit-learn (Sklearn) là thư viện dành cho các thuật toán học máy được viết dựa trên ngôn ngữ Python. Thư viện cung cấp một tập các công cụ xử lý các bài toán machine learning và statistical modeling gồm: classification (phân loại), regression (hồi quy), clustering (phân cụm), và dimensionality reduction (giảm chiều).

Sau đây là một số thuật toán được xây dựng bởi thư viện scikit-learn:

`	`+ Clustering: Nhóm thuật toán Phân cụm dữ liệu không gán nhãn. Ví dụ thuật toán Kmeans.

`	`+ Cross Validation: Kiểm thử chéo, đánh giá độ hiệu quả của thuật toán học giám sát sử dụng dữ liệu kiểm thử (validation data) trong quá trình huấn luyện mô hình.

`	`+ Datasets: Gồm nhóm các Bộ dữ liệu được tích hợp sẵn trong thư viện. Hầu như các bộ dữ liệu đã được chuẩn hóa và mang lại hiệu suất cao trong quá trình huấn luyện như iris, digit, ...

`	`+ Dimensionality Reduction: Mục đích của thuật toán này là để Giảm số lượng thuộc tính quan trọng của dữ liệu bằng các phương pháp như tổng hợp, biểu diễn dữ liệu và lựa chọn đặc trưng. Ví dụ thuật toán PCA.

`	`+ Ensemble methods: Các phương pháp tập hợp sử dụng nhiều thuật toán học tập để có được hiệu suất dự đoán tốt hơn so với bất kỳ thuật toán học cấu thành nào.

`	`+ Feature extraction: Trích chọn đặc trưng. Lựa chọn các đặc trưng có ý nghĩa trong việc huấn luyện mô hình học giám sát.

`	`+ Parameter Tuning: Tinh chỉnh tham số. Các thuật toán phục vụ việc lựa chọn tham số phù hợp để tối ưu hóa mô hình.

`	`+ Manifold Learning: Các thuật toán học tổng hợp và phân tích dữ liệu đa chiều phức tạp.

`	`+ Supervised Models: Học giám sát. Mảng lớn các thuật toán học máy hiện nay. Ví dụ như linear models, discriminate analysis, naive bayes, lazy methods, neural networks, support vector machines và dicision trees.

3\. 2. Ví dụ

`	`Chúng ta sử dụng cây quyết định Decision tree phân loại để mô hình hóa bộ dữ liệu hoa iris.

`	`Bộ dữ liệu này được cung cấp dưới dạng tập dữ liệu mẫu với thư viện và được tải.

**Import thư viện:** pip install scikit-learn

**Source code:** 

![](Aspose.Words.19c79ee9-1862-49b2-b405-5fca229b99b5.001.png)

**Ví dụ:** Với test\_size=0.8 và random\_state=50 thì độ chính xác là 83,33%. Trong đó:

`	`+ Loại 1: Mô hình đoán đúng 39 cây

`	`+ Loại 2: Mô hình đoán đúng 27 cây loại 2 và đoán nhầm 15 cây loại 2 sang loại 3

`	`+ Loại 3: Mô hình đoán đúng 34 cây loại 3 và đoán nhầm 5 cây loại 3 sang loại 2
