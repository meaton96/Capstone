"""
Thin wrapper around `mlagents-learn` that silences the
"Unknown side channel data received" warning.

That warning fires once per completed episode (EpisodeTelemetryChannel flushes one
message to Python at episode end; bare mlagents-learn has no listener registered for
it, so mlagents_envs logs it as unrecognized). Under RLDecisionDrainMode, episodes can
complete many times per second, which floods the terminal and buries the actual
Step/Training/Mean Reward lines. mlagents-learn's own --debug flag only toggles
DEBUG vs INFO and can't raise the threshold above WARNING, so this filters the one
noisy message directly instead.

Usage: identical to mlagents-learn, just swap the command:
    python mlagents_learn_quiet.py smoke_test.yaml --env=linux_server/capstone.x86_64 \\
        --run-id=run02 --no-graphics --force --env-args -rldecisiondrain true
"""
import logging
import sys


class _DropUnknownSideChannel(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Unknown side channel data received" not in record.getMessage()


# Pre-create the logger mlagents_envs' side channel manager will later fetch by the
# same name (logging.getLogger is a singleton per name) and attach a filter to it.
# Filters survive any later logger.setLevel() call mlagents-learn makes internally,
# unlike a level change, which is why this works regardless of --debug.
logging.getLogger("mlagents_envs.side_channel.side_channel_manager").addFilter(
    _DropUnknownSideChannel()
)

from mlagents.trainers.learn import main  # noqa: E402  (must come after filter setup)

if __name__ == "__main__":
    sys.exit(main())
