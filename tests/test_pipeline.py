import numpy as np
import pandas as pd
import pytest
from ucimlrepo import fetch_ucirepo

def test_data_loading():
    """
    Testa se o dataset de Bike Sharing (ID 275) é carregado corretamente.
    """
    try:
        ds = fetch_ucirepo(id=275)
        X = ds.data.features
        y = ds.data.targets['cnt']
        
        assert X.shape[0] == 17379, f"Esperado 17379 instâncias, obtido {X.shape[0]}"
        assert X.shape[1] == 13, f"Esperado 13 features, obtido {X.shape[1]}"
        assert len(y) == 17379, "Shape da variável-alvo inconsistente com as features"
    except Exception as e:
        pytest.fail(f"Erro ao carregar os dados da UCI: {e}")

def test_log1p_inverse():
    """
    Testa se a transformação log1p e sua reversa expm1 são consistentes.
    """
    original = np.array([0, 10, 100, 1000], dtype=float)
    transformed = np.log1p(original)
    reverted = np.expm1(transformed)
    
    np.testing.assert_allclose(original, reverted, rtol=1e-5)
