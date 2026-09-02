# Nhân ma trận & Broadcasting (bug shape #1 của người mới)


"""

- Bài toán nhân ma trận:
    - X: (batch_size, in_features)
    - W: (out_features, in_features)
    - b: (out_features,)
    - Y: (batch_size, out_features)

- Quy tắc Broadcasting:
    - So chiều từ phải sang trái, hai chiều hợp lệ khi bằng nhau HOẶC một trong hai bằng 1.
    
    - Ví dụ: Dưới đây là ý nghĩa của kích thước (5, 2, 3):

        5: Mảng lớn ngoài cùng có 5 khối (hoặc 5 ma trận 2D).

        2: Trong mỗi khối đó, có 2 hàng (thay vì 1 hàng như trước).

        3: Trong mỗi hàng, có 3 cột (chứa 3 phần tử).

        Tổng cộng, mảng này có 30 phần tử (5 x 2 x 3 = 30).
        
        array([[[1., 1., 1.],
        [1., 1., 1.]],

       [[1., 1., 1.],
        [1., 1., 1.]],

       [[1., 1., 1.],
        [1., 1., 1.]],

       [[1., 1., 1.],
        [1., 1., 1.]],

       [[1., 1., 1.],
        [1., 1., 1.]]])

"""


import numpy as np

batch_size, in_features, out_features = 4, 3, 2

X = np.random.randn(batch_size, in_features)   #(4, 3)
W = np.random.randn(out_features, in_features) #(2, 3)
b = np.random.randn(out_features)              # (2,)

Y = X @ W.T + b                                   #(4, 2)
print(X.shape, W.shape, b.shape, Y.shape)
#(4, 3) (2, 3) (2,) (4, 2)
# QUY TẮC BROADCASTING: so chiều từ PHẢI sang TRÁI,
# hai chiều hợp lệ khi bằng nhau HOẶC một trong hai bằng 1.
A = np.ones((5, 1, 3))
B = np.ones((   4, 3))
# (5, 1, 3) (4, 3)
print(A)
print(B)
print((A + B)) # (5, 4, 3)
print((A + B).shape)  # (5, 4, 3)  <- chiều 1 được "kéo giãn" thành 4
# (5, 4, 3)
