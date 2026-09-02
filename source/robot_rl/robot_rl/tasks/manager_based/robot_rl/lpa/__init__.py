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
        id="LPA-walking-clf-graph1",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFGraph1EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalkarm",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalkArmEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgLowEnt",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalkclr",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalkClrEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgLowEnt",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalksym",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalkSymEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgLowEnt",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalkgait",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalkGaitEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgLowEnt",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalkgait2",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalkGait2EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgLowEnt",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalkclear",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalkClearEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgLowEnt",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalkp9",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalkP9EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgLowEnt",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalk3",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            # SAME env cfg as cladwalk2 -- the only change is the
            # RUNNER, so a difference between the two runs is the
            # entropy coefficient and nothing else.
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalk2EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgLowEnt",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalk2",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalk2EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-cladwalk",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFCladWalkEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-clad4",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFClad4EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-clad3",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFClad3EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-clad2",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFClad2EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="LPA-walking-clf-clad1",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFClad1EnvCfg",
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
        id="LPA-walking-clf-rough-honest-gait",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV5EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
        },
    )

    gym.register(
        id="LPA-walking-clf-arm-swing-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFArmSwingPlayEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
        },
    )

    gym.register(
        id="LPA-walking-clf-rough-lowent",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV10EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48LowEnt",
        },
    )

    gym.register(
        id="LPA-walking-clf-rough-elbow-depth-tuned",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV10EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
        },
    )

    gym.register(
        id="LPA-walking-clf-rough-elbow-depth",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV9EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
        },
    )

    gym.register(
        id="LPA-walking-clf-rough-elbow-tuck",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV8EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
        },
    )

    gym.register(
        id="LPA-walking-clf-rough-contact-priced",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV7EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
        },
    )

    gym.register(
        id="LPA-walking-clf-rough-arm-swing",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lpa_walking_clf_env_cfg:LpaWalkingCLFRoughV6EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfgH48",
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
