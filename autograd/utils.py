import numpy as np
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_tensor(root, show_data=False):
    """
    Visualizes a Tensor computation graph. 
    By default, only shows shapes to keep massive networks readable.
    Set show_data=True to print the actual matrices and gradients.
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        shape_str = str(n.data.shape)
        
        if show_data:
            data_str = np.array2string(n.data, precision=4, suppress_small=True).replace('\n', '\\n')
            grad_str = np.array2string(n.grad, precision=4, suppress_small=True).replace('\n', '\\n')
            
            label = "{%s | shape %s | data:\\n%s | grad:\\n%s}" % (n.label, shape_str, data_str, grad_str)
        else:
            label = "{%s | shape %s}" % (n.label, shape_str)
            
        dot.node(name=uid, label=label, shape='record')
        
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
            
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
    return dot