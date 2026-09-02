import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 51)
train_loss = 2.5 * np.exp(-epochs / 12) + 0.1 + np.random.rand(50) * 0.03

"""
np.exp(-1/12), np.exp(-2/12), ..., np.exp(-50/12)
Điều này mô phỏng hiện tượng loss giảm khi mô hình học tốt hơn.
2.5 * ... làm loss ban đầu lớn hơn.
+ 0.1 đặt mức loss tối thiểu, để loss không giảm về 0 tuyệt đối.
np.random.rand(50) tạo 50 số ngẫu nhiên trong khoảng [0, 1).
np.random.rand(50) * 0.03 thêm nhiễu nhỏ vào train loss, giúp đường biểu đồ trông tự nhiên hơn.
"""

val_loss   = 2.5 * np.exp(-epochs / 15) + 0.25 + np.random.rand(50) * 0.05

"""
Dòng này mô phỏng loss của tập validation.
Nó tương tự train_loss, nhưng có vài khác biệt:
epochs / 15 làm validation loss giảm chậm hơn train loss.
+ 0.25 khiến validation loss cao hơn train loss.
np.random.rand(50) * 0.05 thêm nhiễu lớn hơn một chút so với train loss.
Điều này phản ánh thực tế: validation loss thường dao động nhiều hơn và cao hơn train loss.
"""

val_loss[30:] += np.linspace(0, 0.25, 20)      # bắt đầu overfit từ epoch 30

"""
Dòng này mô phỏng hiện tượng overfit.
Từ epoch 30 trở đi, validation loss tăng lên rất nhanh.
np.linspace(0, 0.25, 20) tạo 20 số từ 0 đến 0.25, tăng dần.
val_loss[30:] += ... thêm vào validation loss để tăng lên.
Điều này phản ánh thực tế: khi mô hình học quá tốt, nó sẽ không hoạt động tốt trên tập validation.
"""

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(epochs, train_loss, label="train")
ax[0].plot(epochs, val_loss, label="validation")
ax[0].axvline(31, ls="--", c="red", label="bắt đầu overfit")
ax[0].set(xlabel="Epoch", ylabel="Loss", title="Learning Curve")
ax[0].legend(); ax[0].grid(alpha=.3)

# Ma trận nhầm lẫn
cm = np.array([[45, 3, 2], [5, 40, 5], [1, 4, 45]])
im = ax[1].imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax[1].text(j, i, cm[i, j], ha="center", va="center")
ax[1].set(xlabel="Dự đoán", ylabel="Thực tế", title="Confusion Matrix")
plt.colorbar(im, ax=ax[1])

plt.tight_layout()
plt.savefig("phase1_plots.png", dpi=120)
plt.show()