import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, label='{self.label}')\n{self.data}"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Route gradients, reducing along axis 0 if NumPy broadcasting occurred
            if self.data.shape == out.data.shape:
                self.grad += out.grad
            else:
                self.grad += np.sum(out.grad, axis=0, keepdims=True)
                
            if other.data.shape == out.data.shape:
                other.grad += out.grad
            else:
                other.grad += np.sum(out.grad, axis=0, keepdims=True)
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            # Apply chain rule for matrix multiplication via transposes
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        
        def _backward():
            # Local derivative of tanh(x) is 1 - tanh^2(x)
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        
        def _backward():
            # Local derivative of ReLU is 1 for x > 0, else 0
            self.grad += (self.data > 0) * out.grad
            
        out._backward = _backward
        return out
    
    def sum(self):
        out = Tensor(np.sum(self.data), (self,), 'sum')
        
        def _backward():
            # Distribute the scalar gradient equally to all elements
            self.grad += np.ones_like(self.data) * out.grad
            
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        
        def _backward():
            # Route gradients and handle dimensional broadcasting
            if self.data.shape == out.data.shape:
                self.grad += out.grad
            else:
                self.grad += np.sum(out.grad, axis=0, keepdims=True)
                
            if other.data.shape == out.data.shape:
                other.grad -= out.grad 
            else:
                other.grad -= np.sum(out.grad, axis=0, keepdims=True)
            
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Apply product rule and handle dimensional broadcasting
            if self.data.shape == out.data.shape:
                self.grad += other.data * out.grad
            else:
                self.grad += np.sum(other.data * out.grad, axis=0, keepdims=True)
                
            if other.data.shape == out.data.shape:
                other.grad += self.data * out.grad
            else:
                other.grad += np.sum(self.data * out.grad, axis=0, keepdims=True)
            
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Seed the output gradient
        self.grad = np.ones_like(self.data)
        
        # Traverse the DAG in reverse topological order
        for node in reversed(topo):
            node._backward()