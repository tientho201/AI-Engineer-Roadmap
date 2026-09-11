import timm , torch, time, numpy as np
import onnxruntime as ort

model = timm.create_model("convnextv2_tiny", pretrained=True, num_classes=10).eval()
dummy = torch.randn(1, 3, 224, 224)

# 1) Xuất ONNX
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)

# 2) So sánh tốc độ
def bench(fn, n=100, warmup=10):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1000

sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
x_np = dummy.numpy()

with torch.no_grad():
    print(f"PyTorch     : {bench(lambda: model(dummy)):.2f} ms")
print(f"ONNX Runtime: {bench(lambda: sess.run(None, {'input': x_np})):.2f} ms")

# 3) Kiểm tra kết quả khớp nhau (BẮT BUỘC sau mọi lần convert!)
with torch.no_grad():
    torch_out = model(dummy).numpy()
onnx_out = sess.run(None, {"input": x_np})[0]
print("Sai khác tối đa:", np.abs(torch_out - onnx_out).max())   # nên < 1e-4