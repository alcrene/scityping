import numpy as np
import torch
from scityping.torch import TorchTensor, TorchModule, TorchGenerator
from scityping.torch import torch_module_state_encoder, torch_module_state_decoder

def test_torch():
    # Scalars
    s_u8   = torch.tensor(np.uint8(5))
    s_i32  = torch.tensor(np.int32(5))
    s_f64  = torch.tensor(np.float64(5))
    s_bool = torch.tensor(True)
    # Vectors
    v_u8   = torch.tensor(np.uint8  ([5, 8, 9]))
    v_i32  = torch.tensor(np.int32  ([5, 8, 9]))
    v_f64  = torch.tensor(np.float64([5, 8, 9]))
    v_bool = torch.tensor([True, False, True])
    # Matrices
    m_u8   = torch.outer(v_u8, v_u8)
    m_i32  = torch.outer(v_i32, v_i32)
    m_f64  = torch.outer(v_f64, v_f64)
    m_bool = torch.outer(v_bool, v_bool)

    for a in [s_u8, s_i32, s_f64, s_bool,
              v_u8, v_i32, v_f64, v_bool,
              m_u8, m_i32, m_f64, m_bool]:
        b = TorchTensor.validate(TorchTensor.reduce(a))
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert (a == b).all()


    # TODO: Test TorchModule, TorchGenerator
    #       Test TorchTensor & TorchGenerator with GPU tensors, if GPU is available
