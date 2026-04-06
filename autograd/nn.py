from engine import Tensor
import numpy as np

class Linear:
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nin, nout) * 0.1, label='W')
        self.b = Tensor(np.random.randn(1, nout) * 0.1, label='b')
        
    def __call__(self, x):
        # Forward pass: (batch_size, nin) @ (nin, nout) + (1, nout)
        return (x @ self.W) + self.b
    
    def parameters(self):
        return [self.W, self.b]


class VectorizedMLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply non-linearity to all hidden layers, leaving the output layer linear
            if i < len(self.layers) - 1:
                x = x.tanh() 
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]