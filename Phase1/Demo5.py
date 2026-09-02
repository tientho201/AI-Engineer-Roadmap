class Value:
    """Scalar có khả năng tự tính gradient — bản rút gọn của torch.Tensor."""
    def __init__(self, data, _children=(), _op = ""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}), grad={self.grad}"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  -> gradient đi thẳng qua
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a  <- ĐÂY LÀ CHAIN RULE
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out
    
    def __tanh_activation(self, data):
        import math
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        return t  
    
    def tanh(self):
        t = self.__tanh_activation(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2  <- ĐÂY LÀ CHAIN RULE
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        # Sắp xếp topo: phải tính gradient node sau trước node trước
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0                 # dL/dL = 1
        for node in reversed(topo):
            node._backward()
            
x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
b = Value(6.8813735870195432)

n = x1 * w1 + x2 * w2 + b
o = n.tanh()
o.backward()

print(f"output = {o.data:.4f}")
print(f"dL/dw1 = {w1.grad:.4f}  (kỳ vọng ≈ 1.0000)")
print(f"dL/dx1 = {x1.grad:.4f}  (kỳ vọng ≈ -1.5000)")