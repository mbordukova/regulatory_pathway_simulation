"""
Tests for sampler.
"""
from unittest import mock

import pytest

from sampler import Sampler


@pytest.mark.parametrize('bernoulli_outcome', [0, 1])
def test_sampler_returns_outcome_of_random_state_binomial_passed_to_it(bernoulli_outcome):
    random_state = mock.Mock()
    mock_binomial = mock.Mock(return_value=bernoulli_outcome)
    random_state.binomial = mock_binomial

    sampler = Sampler(random_state=random_state)
    bernoulli_parameter = 0.75

    assert sampler.get_bernoulli_sample(bernoulli_parameter=bernoulli_parameter) == bernoulli_outcome

    mock_binomial.assert_called_once_with(n=1, p=bernoulli_parameter, size=1)
