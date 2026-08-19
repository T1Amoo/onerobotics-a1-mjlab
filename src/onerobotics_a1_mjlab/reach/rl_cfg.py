"""RSL-RL PPO configuration for OneRobotics A1 pose reaching."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def a1_reach_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return the public PPO baseline, without implying policy convergence."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=8,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="onerobotics_a1_reach",
    logger="tensorboard",
    upload_model=False,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=3_000,
    clip_actions=6.0,
  )
