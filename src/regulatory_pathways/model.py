from typing import Any, ClassVar, List

from dataclasses import dataclass

from regulatory_pathways.sampler import Sampler


@dataclass(init=True, frozen=True)
class SystemConfiguration:
    regulatory_protein_binding_rate: float
    regulatory_protein_dissolution_rate: float
    mrna_production_rate: float
    mrna_degeneration_rate: float


@dataclass(init=True, frozen=True)
class CompoundState:
    compound_name: ClassVar[str]
    state_name: ClassVar[str]
    state_unit: ClassVar[str]
    value: Any = None


class DNAState(CompoundState):
    compound_name = 'dna'
    state_name = 'is_bound'


class MRNAState(CompoundState):
    compound_name = 'mrna'
    state_name = 'concentration'
    state_unit = 'm^-3'


@dataclass
class SystemState:
    dna_state: DNAState
    mrna_state: MRNAState

    def __init__(self, dna_state_value: int, mrna_state_value: float):
        self.dna_state = DNAState(value=dna_state_value)
        self.mrna_state = MRNAState(value=mrna_state_value)


@dataclass
class EnsembleState:
    system_state_samples: List[SystemState]

    @property
    def num_samples(self):
        return len(self.system_state_samples)


class StateEvolution:
    def __init__(self, sampler: Sampler, system_configuration: SystemConfiguration):
        self.sampler = sampler
        self.system_configuration = system_configuration

    def get_next_state(self, state: SystemState, step: float) -> SystemState:
        next_dna_state_value = self._get_next_dna_state_value(state.dna_state, step)
        next_mrna_state_value = self._get_next_mrna_state_value(state.mrna_state, next_dna_state_value, step)

        return SystemState(
            dna_state_value=next_dna_state_value, mrna_state_value=next_mrna_state_value
        )

    def _get_next_dna_state_value(self, state: DNAState, step: float) -> int:
        if state.value == 0:
            probability_to_flip = step * self.system_configuration.regulatory_protein_binding_rate
        elif state.value == 1:
            probability_to_flip = step * self.system_configuration.regulatory_protein_dissolution_rate
        else:
            raise ValueError('Unknown DNA state value.')

        will_flip = self.sampler.get_bernoulli_sample(bernoulli_parameter=probability_to_flip)

        if will_flip:
            next_state_value = 1 - state.value
        else:
            next_state_value = state.value

        return next_state_value

    def _get_next_mrna_state_value(self, mrna_state: MRNAState, dna_state_value: int, step: float) -> float:
        return mrna_state.value + step * (
                dna_state_value * self.system_configuration.mrna_production_rate
                - self.system_configuration.mrna_degeneration_rate * mrna_state.value
        )
