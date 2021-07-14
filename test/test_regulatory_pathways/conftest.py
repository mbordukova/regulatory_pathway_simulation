from scipy import stats

import pytest

from model import SystemConfiguration


@pytest.fixture(scope='function')
def system_configuration():
    system_configuration = SystemConfiguration(
        regulatory_protein_binding_rate=0.5,
        regulatory_protein_dissolution_rate=0.75,
        mrna_production_rate=0.125,
        mrna_degeneration_rate=0.0625,
    )
    return system_configuration


@pytest.fixture(scope='function')
def promoter_state_distribution():
    return stats.bernoulli(p=0.75)


@pytest.fixture(scope='function')
def mrna_concentration_distribution():
    return stats.beta(a=0.25, b=0.75)
