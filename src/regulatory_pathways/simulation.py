from dataclasses import dataclass

from typing import List

import numpy as np
import scipy.stats

from regulatory_pathways.model import SystemState, EnsembleState, StateEvolution


@dataclass(init=True, frozen=True)
class EnsembleEvolution:
    time: np.ndarray
    ensemble_states: List[EnsembleState]

    @property
    def mrna_paths(self):
        num_states = self.ensemble_states[0].num_samples
        paths = [[] for _ in range(num_states)]

        for state in self.ensemble_states:
            for j in range(num_states):
                paths[j].append(state.system_state_samples[j].mrna_state.value)

        return self.time, paths

    @property
    def mrna_average_path(self):
        _, paths = self.mrna_paths
        paths = np.array(paths)
        average_path = paths.mean(axis=0)
        return self.time, average_path

    @property
    def mrna_histogram_path(self):
        histograms = []

        for state in self.ensemble_states:
            values = [system_state.mrna_state.value for system_state in state.system_state_samples]
            histograms.append(values)

        return histograms

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return self.ensemble_states == other.ensemble_states and \
               np.allclose(self.time, other.time)


class Experiment:
    def __init__(self, num_steps: int, step_size: float, state_evolution: StateEvolution):
        self.state_evolution: StateEvolution = state_evolution
        self.time_axis: np.ndarray = np.linspace(0, (num_steps - 1) * step_size, num=num_steps, endpoint=True)

    def run(self, ensemble: EnsembleState) -> EnsembleEvolution:
        ensemble_evolution = [ensemble]

        for t_next, t_prev in zip(self.time_axis[1:], self.time_axis[:-1]):
            step = t_next - t_prev

            ensemble = EnsembleState(
                [
                    self.state_evolution.get_next_state(system_state, step)
                    for system_state in ensemble.system_state_samples
                ]
            )

            ensemble_evolution.append(ensemble)

        return EnsembleEvolution(ensemble_states=ensemble_evolution, time=self.time_axis)


def generate_ensemble(
        promoter_state_distribution: scipy.stats.rv_discrete,
        mrna_concentration_distribution: scipy.stats.rv_continuous,
        num_samples: int = 1_000,
        random_state: np.random.RandomState = np.random.RandomState(seed=0)
) -> EnsembleState:
    """

    Parameters
    ----------
    promoter_state_distribution : scipy.stats.rv_discrete
    mrna_concentration_distribution : scipy.stats.rv_continuous
    num_samples : int
    random_state : np.random.RandomState

    Returns
    -------
    ensemble_state : EnsembleState
        Ensemble of ``num_samples`` samples drawn from the given distributions.
    """
    samples = [
        SystemState(
            dna_state_value=promoter_state_distribution.rvs(random_state=random_state),
            mrna_state_value=100 * mrna_concentration_distribution.rvs(random_state=random_state),
        ) for _ in range(num_samples)
    ]
    return EnsembleState(samples)
