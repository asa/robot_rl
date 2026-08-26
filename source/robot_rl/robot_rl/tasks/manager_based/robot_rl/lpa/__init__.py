# TINH LPA environments (bead tinh-wf62).

import gymnasium as gym

# Reuse the G1 agent (PPO runner) configs — they are robot-agnostic.
from ..g1 import agents

_registered = False

if not _registered:
    gym.register(
        id="LPA-walking-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-skill",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFSkillEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-graphturn",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFGraphTurnEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-ramp",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRampEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-rough-retrieve2",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV4EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
        },
    )

    gym.register(
        id="LPA-walking-clf-rough-retrieve",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV3EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
        },
    )

    gym.register(
        id="LPA-walking-clf-rough-priced",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV2EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
        },
    )

    _registered = True

    gym.register(
        id="LPA-walking-clf-rough",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

gym.register(
        id="LPA-walking-clf-hist",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFHistEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

gym.register(
        id="LPA-walking-clf-hist-clear4",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFHistClear4EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )
