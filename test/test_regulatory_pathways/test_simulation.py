import numpy as np

from unittest import mock

from model import SystemState, EnsembleState, StateEvolution
from sampler import Sampler
from simulation import Experiment, generate_ensemble, EnsembleEvolution


def test_experiment_generates_correct_time_axis(system_configuration):
    sampler = Sampler()

    state_evolution = StateEvolution(system_configuration=system_configuration, sampler=sampler)

    experiment = Experiment(num_steps=5, step_size=1, state_evolution=state_evolution)

    np.testing.assert_array_almost_equal(experiment.time_axis, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_generate_samples(promoter_state_distribution, mrna_concentration_distribution):
    ensemble = generate_ensemble(
        promoter_state_distribution,
        mrna_concentration_distribution,
        num_samples=10,
        random_state=np.random.RandomState(seed=0),
    )

    expected_promoter_state_values = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1]
    expected_mrna_state_values = [
        0.33942501326209373,
        0.05455125471258647,
        0.028668064520627888,
        0.000658784822938769,
        0.5335263671238721,
        0.00035473780166545433,
        0.14827079072007254,
        0.08420245965199201,
        0.21875465704527183,
        0.04795217022785702,
    ]

    expected_system_states = [
        SystemState(dna_state, mrna_state)
        for dna_state, mrna_state in zip(expected_promoter_state_values, expected_mrna_state_values)
    ]

    expected_ensemble_state = EnsembleState(expected_system_states)

    assert ensemble == expected_ensemble_state


def test_experiment_run_creates_paths_for_ensemble(
        system_configuration,
        promoter_state_distribution,
        mrna_concentration_distribution,
):
    ensemble = generate_ensemble(
        promoter_state_distribution, mrna_concentration_distribution, 2, np.random.RandomState(seed=0)
    )

    sampler = Sampler()

    sampler.get_bernoulli_sample = mock.Mock(side_effect=[0, 1, 1, 1])

    state_evolution = StateEvolution(system_configuration=system_configuration, sampler=sampler)

    experiment = Experiment(num_steps=3, step_size=0.25, state_evolution=state_evolution)

    ensemble_evolution = experiment.run(ensemble)

    # manually verified values
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
