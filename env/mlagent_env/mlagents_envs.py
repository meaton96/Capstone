from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

class UnitySchedulingEnv:

    def __init__(self, file_name=None, time_scale=20.0):
        self.engine_channel = EngineConfigurationChannel()
        # file_name=None connects to the Unity Editor
        self.env = UnityEnvironment(
            file_name=file_name,
            side_channels=[self.engine_channel],
        )
        # Speed up simulation — no need for real-time visuals during training
        self.engine_channel.set_configuration_parameters(time_scale=time_scale)

        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

    def reset(self):
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        obs = self._extract_obs(decision_steps)
        return obs

    def step(self, action: int):
        from mlagents_envs.base_env import ActionTuple
        import numpy as np

        action_tuple = ActionTuple(
            discrete=np.array([[action]], dtype=np.int32)
        )
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            obs = self._extract_obs(terminal_steps)
            reward = terminal_steps.reward[0]
            done = True
        else:
            obs = self._extract_obs(decision_steps)
            reward = decision_steps.reward[0]
            done = False

        return obs, reward, done, {}

    def _extract_obs(self, steps):
        """Convert ML-Agents obs into your obs_dict format."""
        raw = steps.obs[0][0]  # first sensor, first agent
        # Slice raw into your expected observation components:
        # factory_grid, sched_matrix, global_scalars, etc.
        # This mapping must match CollectObservations order
        return {"raw_obs": raw}  # expand this to match your encoder inputs

    def close(self):
        self.env.close()