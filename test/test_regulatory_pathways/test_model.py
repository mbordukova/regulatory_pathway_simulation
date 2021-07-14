from unittest import mock

from model import DNAState, SystemState, StateEvolution, MRNAState
from sampler import Sampler


def test_state_evolution_dna_flips_from_0_to_1(system_configuration):
    sampler = Sampler()
    sampler.get_bernoulli_sample = mock.Mock(return_value=1)

    system_state = SystemState(dna_state_value=0, mrna_state_value=0.25)

    state_evolution = StateEvolution(sampler, system_configuration)

    next_state = state_evolution.get_next_state(system_state, step=0.25)

    # flipped 0 -> 1
    assert next_state.dna_state == DNAState(value=1)

    sampler.get_bernoulli_sample.assert_called_once_with(bernoulli_parameter=0.125)

    # z_1 = z_0 + step * (next_dna_state * beta - gamma * z)
    # z_0 = 0.25 + 0.25 * (1 * 0.125 - 0.0625 * 0.25) = 0.27734375
    assert next_state.mrna_state == MRNAState(value=0.27734375)


def test_state_evolution_dna_flips_from_1_to_0(system_configuration):
    sampler = Sampler()
    sampler.get_bernoulli_sample = mock.Mock(return_value=1)

    system_state = SystemState(dna_state_value=1, mrna_state_value=0.25)

    state_evolution = StateEvolution(sampler, system_configuration)

    next_state = state_evolution.get_next_state(system_state, step=0.25)

    # flipped 1 -> 0
    assert next_state.dna_state == DNAState(value=0)

    sampler.get_bernoulli_sample.assert_called_once_with(bernoulli_parameter=0.1875)

    # z_1 = z_0 + step * (next_dna_state * beta - gamma * z)
    # z_0 = 0.25 + 0.25 * (0 * 0.125 - 0.0625 * 0.25) = 0.24609375
    assert next_state.mrna_state == MRNAState(value=0.24609375)


def test_state_evolution_dna_doesnt_flip_from_0(system_configuration):
    sampler = Sampler()
    sampler.get_bernoulli_sample = mock.Mock(return_value=0)

    system_state = SystemState(dna_state_value=0, mrna_state_value=0.25)

    state_evolution = StateEvolution(sampler, system_configuration)

    next_state = state_evolution.get_next_state(system_state, step=0.25)

    # didn't flip 0 -> 1
    assert next_state.dna_state == DNAState(value=0)

    sampler.get_bernoulli_sample.assert_called_once_with(bernoulli_parameter=0.125)

    # z_1 = z_0 + step * (next_dna_state * beta - gamma * z)
    # z_0 = 0.25 + 0.25 * (0 * 0.125 - 0.0625 * 0.25) = 0.24609375
    assert next_state.mrna_state == MRNAState(value=0.24609375)


def test_state_evolution_dna_doesnt_flip_from_1(system_configuration):
    sampler = Sampler()
    sampler.get_bernoulli_sample = mock.Mock(return_value=0)

    system_state = SystemState(dna_state_value=1, mrna_state_value=0.25)

    state_evolution = StateEvolution(sampler, system_configuration)

    next_state = state_evolution.get_next_state(system_state, step=0.25)

    # didn't flipped 1 -> 0
    assert next_state.dna_state == DNAState(value=1)

    sampler.get_bernoulli_sample.assert_called_once_with(bernoulli_parameter=0.1875)

    # z_1 = z_0 + step * (next_dna_state * beta - gamma * z)
    # z_0 = 0.25 + 0.25 * (1 * 0.125 - 0.0625 * 0.25) = 0.27734375
    assert next_state.mrna_state == MRNAState(value=0.27734375)