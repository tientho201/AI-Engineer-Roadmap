import numpy as np
x = np.arange(24).reshape(2, 3, 4)
print("x: ", x)
"""
 [[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
"""

print("shape: ", x.shape, "ndim: ", x.ndim, "dtype: ", x.dtype) # shape:  (2, 3, 4) ndim:  3 dtype:  int64

print("reshape(6, 4): ", x.reshape(6, 4))
"""
reshape(6, 4):  [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]
"""

print("reshape(2, -1): ", x.reshape(2, -1))
"""
reshape(2, -1):  [[ 0  1  2  3  4  5  6  7  8  9 10 11]
 [12 13 14 15 16 17 18 19 20 21 22 23]]
"""

print("transpose(0, 2, 1): ", x.transpose(0, 2, 1))
"""
transpose(0, 2, 1):  [[[ 0  4  8]
  [ 1  5  9]
  [ 2  6 10]
  [ 3  7 11]]

 [[12 16 20]
  [13 17 21]
  [14 18 22]
  [15 19 23]]]
"""

print("newaxis: ", x[np.newaxis, ...])
"""
newaxis:  [[[[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]]

  [[12 13 14 15]
   [16 17 18 19]
   [20 21 22 23]]]]
"""
print("squeeze: ", x.squeeze())
"""
squeeze:  [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]
"""
print("concatenate: ", np.concatenate([x, x], axis=0))
"""
concatenate:  [[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]

 [[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
"""
print("stack: ", np.stack([x, x], axis=0))
"""
stack:  [[[[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]]

  [[12 13 14 15]
   [16 17 18 19]
   [20 21 22 23]]]]
"""
print("sum: ", x.sum(axis=1))
"""
sum:  [[12 15 18]
 [48 51 54]]
"""
print("argmax: ", x.argmax(axis=-1))
"""
argmax:  [[3 3 3]
 [3 3 3]]
"""
print("where: ", np.where(x > 10, x, 0))
"""
where:  [[[ 0  0  0  0]
  [ 0  0  0  0]
  [ 0  0  0  0]]

 [[ 0  0  0  0]
  [ 0  0  0  0]
  [ 0  0  0  0]]]
"""
print("x[x > 10]: ", x[x > 10])
"""
x[x > 10]:  [[11 12 13 14]
 [19 20 21 22]]
"""