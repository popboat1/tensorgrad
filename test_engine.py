import numpy as np
import torch
import torch.nn.functional as F
from tensorgrad.engine import Tensor

def test_matrix_math():
    np.random.seed(42)
    x_data = np.random.randn(3, 4)
    w_data = np.random.randn(4, 5)
    
    # --- PyTorch Implementation ---
    x_pt = torch.tensor(x_data, requires_grad=True)
    w_pt = torch.tensor(w_data, requires_grad=True)
    
    # z = (x @ w).relu().sum()
    z_pt = (x_pt @ w_pt).relu().sum()
    z_pt.backward()
    
    # --- TensorGrad Implementation ---
    x_tg = Tensor(x_data)
    w_tg = Tensor(w_data)
    
    z_tg = (x_tg @ w_tg).relu().sum()
    z_tg.backward()
    
    # --- Assertions ---
    np.testing.assert_allclose(z_pt.data.numpy(), z_tg.data, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(x_pt.grad.numpy(), x_tg.grad, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(w_pt.grad.numpy(), w_tg.grad, rtol=1e-4, atol=1e-4)

def test_broadcasting():
    np.random.seed(42)
    x_data = np.random.randn(3, 4)
    b_data = np.random.randn(1, 4)
    
    # PyTorch
    x_pt = torch.tensor(x_data, requires_grad=True)
    b_pt = torch.tensor(b_data, requires_grad=True)
    z_pt = (x_pt + b_pt).sum()
    z_pt.backward()
    
    # TensorGrad
    x_tg = Tensor(x_data)
    b_tg = Tensor(b_data)
    z_tg = (x_tg + b_tg).sum()
    z_tg.backward()
    
    # Assertions
    np.testing.assert_allclose(b_pt.grad.numpy(), b_tg.grad, rtol=1e-4, atol=1e-4)

def test_cross_entropy():
    np.random.seed(42)
    # 3 samples, 10 classes
    logits_data = np.random.randn(3, 10)
    # Random target classes between 0 and 9
    target_data = np.random.randint(0, 10, size=(3,))
    
    # PyTorch
    logits_pt = torch.tensor(logits_data, requires_grad=True)
    target_pt = torch.tensor(target_data, dtype=torch.long)
    loss_pt = F.cross_entropy(logits_pt, target_pt)
    loss_pt.backward()
    
    # TensorGrad
    logits_tg = Tensor(logits_data)
    loss_tg = logits_tg.cross_entropy(target_data)
    loss_tg.backward()
    
    # Assertions
    np.testing.assert_allclose(loss_pt.item(), loss_tg.data.item(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(logits_pt.grad.numpy(), logits_tg.grad, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    print("Running Test...")
    test_matrix_math()
    print("Matrix Math & ReLU Passed")
    test_broadcasting()
    print("Broadcasting Passed")
    test_cross_entropy()
    print("Multiclass Cross-Entropy Passed")
    print("\nAll tests passed! TensorGrad matches PyTorch exactly.")