import sys, json, numpy as np, torch
sys.path.insert(0, "env")
from env_wrappers.twin_env import VectorizedTwinEnv
from scenarios import REGISTRY
from config import EncoderConfig, FusionConfig, ActorCriticConfig, JOB_HEAD_RULES, MACHINE_HEAD_RULES
from models.network import SchedulingNetwork
from models.actor_critic import prior_log_probs
PARAMS = json.loads('{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}')
gen = REGISTRY["randomized"](21600.0, random_warmup=True, warmup_dispatching_rule=None, agv_move_speed=None,
    agv_handshake_duration=None, machine_flexibility=0.0, secondary_time_multiplier=1.0, params_overrides=PARAMS)
venv = VectorizedTwinEnv(8, "results/dev-popart-smoke/floor7/des_floor.json", reward_spec="env/config/rewards/tardiness.json",
    train_seed=0, scenario_generator=gen, obs_caps=(15,256), slot_seconds=900, gamma_per_second=1-1/10800)
obs, _ = venv.reset()
obs_list=[obs]
for i in range(7):
    obs, r, d, t, info = venv.step(np.tile([[2,1]], (8,1)))
    obs_list.append(obs)
O = {k: torch.tensor(np.concatenate([o[k] for o in obs_list]), dtype=torch.float32) for k in obs}
B = O["action_mask"].shape[0]
for prior in ("plain", "offset", "scale"):
    torch.manual_seed(0)
    net = SchedulingNetwork(EncoderConfig(), FusionConfig(), ActorCriticConfig())
    plp = prior_log_probs("MDD-TECT", 0.8, (JOB_HEAD_RULES, MACHINE_HEAD_RULES))
    if prior == "offset": net.actor_critic.actor.set_prior_offset(plp)
    if prior == "scale": net.actor_critic.actor.init_prior(plp)
    net.eval()
    with torch.no_grad(): a, lp, v = net.act(O)
    net.train()
    lp2, v2, ent, kl = net.evaluate(O, a, plp)
    print(prior, "max|ratio-1| at start", float((lp2-lp).exp().sub(1).abs().max()),
          "logit std across states", float(net(O)[0].std(0).mean()))
    adv = torch.randn(B)
    trunk = list(net.fusion.parameters())
    pg = -(torch.exp(lp2 - lp) * adv).mean()
    g_pg = torch.autograd.grad(pg, trunk, retain_graph=True)
    g_v = torch.autograd.grad(0.5 * ((v2 - (v2.detach() + torch.randn(B))) ** 2).mean(), trunk, retain_graph=True)
    n = lambda g: float(torch.sqrt(sum((x**2).sum() for x in g)))
    print("   |grad pg on fusion trunk| %.3g   |grad value on trunk| %.3g" % (n(g_pg), n(g_v)))
