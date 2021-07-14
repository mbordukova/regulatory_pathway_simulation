import re

import numpy as np

from scipy import stats

from regulatory_pathways.model import SystemConfiguration, StateEvolution
from regulatory_pathways.sampler import Sampler
from regulatory_pathways.simulation import generate_ensemble, Experiment, EnsembleEvolution


BERNOULLI_DISTRIBUTION_REGEX = r'^bernoulli\((?P<bernoulli_parameter>\d+\.\d+)\)$'
BETA_DISTRIBUTION_REGEX = r'^beta\((?P<a>\d+\.\d+), (?P<b>\d+\.\d+)\)$'


def run_simulation(
        dna_distribution: str,
        mrna_concentration_distribution: str,
        regulatory_protein_binding_rate: float,
        regulatory_protein_dissolution_rate: float,
        mrna_production_rate: float,
        mrna_degeneration_rate: float,
        num_samples: int = 100,
        num_steps: int = 100,
        step_size: float = 1,
        random_seed: int = 0,
) -> EnsembleEvolution:
    random_state = np.random.RandomState(seed=random_seed)

    bernoulli_parameter_match = re.search(BERNOULLI_DISTRIBUTION_REGEX, dna_distribution)

    if not bernoulli_parameter_match:
        raise ValueError(f'Invalid input for DNA distribution: {dna_distribution}')

    try:
        bernoulli_parameter = float(bernoulli_parameter_match.group('bernoulli_parameter'))
        assert 0.0 <= bernoulli_parameter <= 1.0
    except:
        raise ValueError(
            f'Invalid format for Bernoulli parameter: {bernoulli_parameter_match.group("bernoulli_parameter")}'
        )

    beta_parameter_match = re.search(BETA_DISTRIBUTION_REGEX, mrna_concentration_distribution)

    if not beta_parameter_match:
        raise ValueError(f'Invalid input for mRNA distribution: {mrna_concentration_distribution}')

    try:
        a = float(beta_parameter_match.group('a'))
        b = float(beta_parameter_match.group('b'))
        assert a >= 0.0
        assert b >= 0.0
    except:
        raise ValueError(
            f'Invalid format for Beta parameters: {beta_parameter_match.group("a")}, {beta_parameter_match.group("b")}'
        )

    promoter_state_distribution = stats.bernoulli(p=bernoulli_parameter)
    mrna_concentration_distribution = stats.beta(a=a, b=b)

    system_configuration = SystemConfiguration(
        regulatory_protein_binding_rate=regulatory_protein_binding_rate,
        regulatory_protein_dissolution_rate=regulatory_protein_dissolution_rate,
        mrna_production_rate=mrna_production_rate,
        mrna_degeneration_rate=mrna_degeneration_rate,
    )

    sampler = Sampler()

    ensemble = generate_ensemble(
        promoter_state_distribution, mrna_concentration_distribution, num_samples, random_state
    )

    state_evolution = StateEvolution(system_configuration=system_configuration, sampler=sampler)

    experiment = Experiment(num_steps=num_steps, step_size=step_size, state_evolution=state_evolution)

    ensemble_evolution = experiment.run(ensemble)

    return ensemble_evolution

