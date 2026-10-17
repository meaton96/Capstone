"""
@file eval_wrap.py
@brief rq4-agvfail-v5 (2026-10-14): run a frozen evaluate.py with mlagents' Unity timeout raised to 1,800 s.
@details B2 seed 32 has an 8,399 s random warm-up. Unity simulates it internally, with no exchange with Python, and on
         a cluster node that took about 5.5 min, more than the 300 s timeout_wait hard-coded in the frozen code's
         env_wrappers.unity_env. Python then raised UnityTimeOutException before the first decision (tasks l3000 / pa /
         s32 and none / pn / s32, twice; eval-transfer-unity-v5's ppo s32 on 10-12). Raising the wait only changes how
         long Python waits; the code, the instances and the actions are unchanged.
@par Usage
@code{.sh}
python results/rq4-agvfail-v5-rules/eval_wrap.py CODE_DIR [evaluate.py arguments...]
@endcode
"""
import sys

TIMEOUT_WAIT = 1800


def main():
    code = sys.argv[1]
    sys.path.insert(0, code)
    import mlagents_envs.environment as me
    orig = me.UnityEnvironment.__init__

    def init(self, *a, **kw):
        kw["timeout_wait"] = max(int(kw.get("timeout_wait") or 0), TIMEOUT_WAIT)
        orig(self, *a, **kw)

    me.UnityEnvironment.__init__ = init
    sys.argv = [code + "/evaluate.py"] + sys.argv[2:]
    import evaluate
    evaluate.main()


if __name__ == "__main__":
    main()
