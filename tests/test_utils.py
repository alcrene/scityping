import pytest
import numpy as np
from scityping.utils import TypeRegistry

def test_TypeRegistry():
    A = type("A", (), {})
    B = type("B", (), {})

    # Test that `token_sets` attribute is kept in sync
    reg = TypeRegistry(Generator=A)
    reg[np.random.Generator]=A
    reg["torch.Generator"]=B
    assert len(reg._key_tokens) == 3
    del reg["Generator"]
    assert len(reg._key_tokens) == 2
    assert frozenset(("numpy", "random", "_generator", "generator")) in reg._key_tokens

    assert TypeRegistry(Generator=A).get("numpy.random.Generator") is A
    assert TypeRegistry(Generator=A).get(np.random.Generator) is A
    assert TypeRegistry(Generator=A).get("torch.Generator") is A
    assert TypeRegistry({"numpy.Generator":A}).get("torch.Generator") is A
    assert TypeRegistry({"numpy.Generator":A,
                         "torch.Generator":B}).get("torch.Generator") is B
    assert TypeRegistry({"Generator":A,
                         "torch.Generator":B}).get("torch.Generator") is B
    assert TypeRegistry({"torch.Generator":B,                          # Test for logic errors that would make result
                         "Generator":A}).get("torch.Generator") is B   # dependent on the order of registry entries 
    assert TypeRegistry(Generator=A).get("mypkg.Generator") is A

    # Errors
    # TODO: Caplog, error: replaced key
    # with pytest.raises(ValueError):  # Duplicate keys
    #     TypeRegistry(Generator=A, generator=A)

    with pytest.raises(ValueError):  # Ambiguous match
        TypeRegistry({"Generator":A,
                      "torch.Generator":B})["mypkg.Generator"]
    with pytest.raises(KeyError):  # No matching entry
        TypeRegistry(Juniper=A)[complex]
    with pytest.raises(KeyError):
        TypeRegistry({"numpy.Generator": A})["Generator.numpy"]

    with pytest.raises(KeyError):  # The last token must match
        TypeRegistry({"numpy.Generator": A})["numpy"]