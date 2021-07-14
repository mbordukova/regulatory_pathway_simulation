from unittest import mock

import numpy as np

import pytest

from regulatory_pathways.main import run_simulation
from regulatory_pathways.model import EnsembleState, SystemState
from regulatory_pathways.simulation import EnsembleEvolution


@pytest.fixture(scope='function')
def mock_sampler():
    with mock.patch('main.Sampler.get_bernoulli_sample') as mock_get_bernoulli_sample:
        mock_get_bernoulli_sample.side_effect = [0, 1, 1, 1]
        yield


def test_run_simulation(mock_sampler):
    bernoulli_distribution = 'bernoulli(0.75)'
    beta_distribution = 'beta(0.25, 0.75)'
    regulatory_protein_binding_rate = 0.5
    regulatory_protein_dissolution_rate = 0.75
    mrna_production_rate = 0.125
    mrna_degeneration_rate = 0.0625

    ensemble_evolution = run_simulation(
        dna_distribution=bernoulli_distribution,
        mrna_concentration_distribution=beta_distribution,
        regulatory_protein_binding_rate=regulatory_protein_binding_rate,
        regulatory_protein_dissolution_rate=regulatory_protein_dissolution_rate,
        mrna_production_rate=mrna_production_rate,
        mrna_degeneration_rate=mrna_degeneration_rate,
        num_samples=2,
        num_steps=3,
        step_size=0.25,
        random_seed=0,
    )

    # manually verified values
    # NOTE this test mimics exactly the same flow as in test_experiment_run_creates_paths_for_ensemble
    expected_ensemble_evolution = EnsembleEvolution(
        ensemble_states=[EnsembleState(
            system_state_samples=[
                SystemState(1, 0.33942501326209373), SystemState(1, 0.05455125471258647)
            ]
        ),
            EnsembleState(
                system_state_samples=[
                    SystemState(1, 0.3653714974298735), SystemState(0, 0.053698891357702304)
                ]
            ),
            EnsembleState(
                system_state_samples=[
                    SystemState(0, 0.35966256778253175), SystemState(1, 0.0841098461802382)
                ]
            )
        ],
        time=np.array([0.0, 0.25, 0.5]),
    )

    assert ensemble_evolution == expected_ensemble_evolution
