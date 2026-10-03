"""@file test_switch_oracle.py
@brief The oracle keeps every player's simulated time under the float32-clock budget (audit P2.1)."""
import pytest

from switch_oracle import PLAYER_SIM_BUDGET_S, STARTUP_EPISODE_S, player_chunks


def test_no_warmup_keeps_one_player_per_stage():
    assert player_chunks(12, 5400.0) == [list(range(12))]


@pytest.mark.parametrize("warmup", [0.0, 2000.0, 8608.0])
def test_every_player_stays_under_the_budget(warmup):
    chunks = player_chunks(12, 5400.0 + warmup)
    assert sum(chunks, []) == list(range(12))
    for chunk in chunks:
        assert STARTUP_EPISODE_S + len(chunk) * (5400.0 + warmup) <= PLAYER_SIM_BUDGET_S
    assert PLAYER_SIM_BUDGET_S < 2 ** 17


def test_max_warmup_needs_two_players():
    assert [len(c) for c in player_chunks(12, 5400.0 + 8608.0)] == [8, 4]


def test_an_episode_longer_than_the_budget_is_refused():
    with pytest.raises(ValueError):
        player_chunks(1, 130_000.0)
