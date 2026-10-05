

def test_training_mix_draws_c_per_episode_without_changing_jobs():
    import dataclasses
    from scenarios.randomized import DEFAULT_PARAMS, randomized_scenario
    mixed = dataclasses.replace(DEFAULT_PARAMS, due_date_allowance_range=(1.75, 2.5))
    cs = [randomized_scenario(s, mixed)["_meta"]["due_date_allowance"] for s in range(20)]
    assert all(1.75 <= c <= 2.5 for c in cs) and len(set(cs)) > 10
    a = randomized_scenario(3, mixed)
    fixed = dataclasses.replace(DEFAULT_PARAMS, due_date_allowance=a["_meta"]["due_date_allowance"])
    assert a["jobs"] == randomized_scenario(3, fixed)["jobs"]
    assert a == randomized_scenario(3, mixed)                    # still a pure function of the seed


def test_load_mix_picks_a_profile_and_base_equals_defaults():
    import dataclasses
    import pytest
    from scenarios.randomized import DEFAULT_PARAMS, randomized_scenario
    base_only = dataclasses.replace(DEFAULT_PARAMS, load_mix=({"name": "base"},))
    assert randomized_scenario(5, base_only)["jobs"] == randomized_scenario(5, DEFAULT_PARAMS)["jobs"]
    mix = dataclasses.replace(DEFAULT_PARAMS, load_mix=(
        {"name": "base"}, {"name": "utilhi", "utilization": [1.0, 1.8], "lull_utilization": [0.4, 0.7]}))
    names = {randomized_scenario(s, mix)["_meta"]["load_profile"] for s in range(20)}
    assert names == {"base", "utilhi"}
    hi = next(s for s in range(20) if randomized_scenario(s, mix)["_meta"]["load_profile"] == "utilhi")
    assert randomized_scenario(hi, mix)["_meta"]["params"]["utilization"] == [1.0, 1.8]
    with pytest.raises(ValueError):
        randomized_scenario(0, dataclasses.replace(DEFAULT_PARAMS, load_mix=({"name": "x", "agv_count": 4},)))


def test_regime_blocks_redraw_per_block_and_respect_boundaries():
    import dataclasses
    from scenarios.randomized import DEFAULT_PARAMS, randomized_scenario
    mix = ({"name": "base"}, {"name": "utilhi", "utilization": [1.0, 1.8], "lull_utilization": [0.4, 0.7]})
    p = dataclasses.replace(DEFAULT_PARAMS, regime_block_seconds=5400.0, horizon_seconds=30600.0,
                            due_date_allowance_range=(1.75, 2.5), load_mix=mix)
    sc = randomized_scenario(7, p)
    blocks = sc["_meta"]["blocks"]
    assert len(blocks) == 6 and len({b["due_date_allowance"] for b in blocks}) == 6
    assert all(int(ph["start"] // 5400) == int((ph["end"] - 1e-6) // 5400) for ph in sc["_phases"])
    for j in sc["jobs"]:
        b = blocks[int(j["arrivalTime"] // 5400)]
        work = sum(min(op["duration"]) for op in j["operations"])
        assert abs(j["dueDate"] - (j["arrivalTime"] + b["due_date_allowance"] * work)) < 0.02
    longer = randomized_scenario(7, dataclasses.replace(p, horizon_seconds=40000.0))
    assert longer["_meta"]["blocks"][:6] == blocks                 # adding blocks keeps the earlier ones


def test_regime_blocks_off_leaves_the_instance_unchanged():
    import dataclasses
    from scenarios.randomized import DEFAULT_PARAMS, randomized_scenario
    p = dataclasses.replace(DEFAULT_PARAMS, due_date_allowance=2.0)
    sc = randomized_scenario(4, p)
    assert "blocks" not in sc["_meta"] and all("block" not in ph for ph in sc["_phases"])
