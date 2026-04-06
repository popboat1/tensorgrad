import numpy as np
from tensorgrad.engine import Tensor
from tensorgrad.utils import draw_tensor

X = Tensor(np.random.randn(3, 2), label='X_input')
W = Tensor(np.random.randn(2, 4), label='Weights')
b = Tensor(np.random.randn(1, 4), label='Bias')

Z = X @ W; Z.label = 'X @ W'
Out = Z + b; Out.label = 'Output'

Out.backward()

graph = draw_tensor(Out, show_data=True)

graph.render("tensor_graph", format="svg", cleanup=True)