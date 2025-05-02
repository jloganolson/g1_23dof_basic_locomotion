from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import jax.numpy as jp
from mujoco.mjx._src import math

import numpy as np
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src import collision


from mujoco_playground._src import mjx_env
import g1_constants as consts

KEYFRAME_NAME = "home"

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        restricted_joint_range=False,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
     reward_config=config_dict.create(
        scales=config_dict.create(
          height=-0.5,
          orientation=0.5,
          dof_pos_limits=-0.5,
          pose=-0.1,
          alive=0.,
          termination=-100.0,
          stay_still=-0.1,

          # Energy related rewards.
          torques=0,
          action_rate=-0.01,
          energy=-0.003,
          dof_acc=-2.5e-7
        ),
        base_height_target=0.55,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      mask_arms=True,
    )

class G1Env(mjx_env.MjxEnv):
  """Base class for G1 environments."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._xml_path = "../g1_description/scene_mjx_alt.xml"
    self._mj_model =  mujoco.MjModel.from_xml_path(self._xml_path)

    self._mj_model.opt.timestep = self.sim_dt

    if self._config.restricted_joint_range:
      self._mj_model.jnt_range[1:] = consts.RESTRICTED_JOINT_RANGE
      self._mj_model.actuator_ctrlrange[:] = consts.RESTRICTED_JOINT_RANGE

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)

    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe(KEYFRAME_NAME).qpos)
    self._default_pose = jp.array(self._mj_model.keyframe(KEYFRAME_NAME).qpos[7:])

    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])

    self._left_hand_geom_id = self._mj_model.geom("left_hand_collision").id
    self._right_hand_geom_id = self._mj_model.geom("right_hand_collision").id
    self._left_foot_geom_id = self._mj_model.geom("left_foot").id
    self._right_foot_geom_id = self._mj_model.geom("right_foot").id
    self._left_shin_geom_id = self._mj_model.geom("left_shin").id
    self._right_shin_geom_id = self._mj_model.geom("right_shin").id
    self._left_thigh_geom_id = self._mj_model.geom("left_thigh").id
    self._right_thigh_geom_id = self._mj_model.geom("right_thigh").id

    self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id

    self._desired_up_vec = jp.array([0.0, 1.0, 0.0])
   # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor


  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (23,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
    }

    metrics = {
        **{f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()},
    }

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name

    obs = self._get_obs(data, info, contact)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    action_effect = action * self._config.action_scale
    # Create a mask to zero out action for the last 10 joints (arms) if config is set.
    # Assuming mjx_model.nu is 23.
    arm_mask = jp.ones_like(action_effect).at[-10:].set(0.0)
    masked_action_effect = jp.where(
        self._config.mask_arms, action_effect * arm_mask, action_effect
    )
    motor_targets = self._default_pose + masked_action_effect
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    # contact_filt = contact | state.info["last_contact"]
    
    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    unscaled_rewards = self._get_reward(
        data, action, state.info, done
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in unscaled_rewards.items()
    }
    reward = sum(rewards.values()) * self.dt


    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    
    # state.info["last_contact"] = contact
    
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    
    
    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
    
    
  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data, "pelvis")[-1] < 0.0
    contact_termination = collision.geoms_colliding(
        data,
        self._right_foot_geom_id,
        self._left_foot_geom_id,
    )
    contact_termination |= collision.geoms_colliding(
        data,
        self._left_foot_geom_id,
        self._right_shin_geom_id,
    )
    contact_termination |= collision.geoms_colliding(
        data,
        self._right_foot_geom_id,
        self._left_shin_geom_id,
    )
    return (
        fall_termination
        | contact_termination
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
    )

  def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
      gyro = self.get_gyro(data, "pelvis")
      info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_gyro = (
          gyro
          + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
          * self._config.noise_config.level
          * self._config.noise_config.scales.gyro
      )

      gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
      info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_gravity = (
          gravity
          + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
          * self._config.noise_config.level
          * self._config.noise_config.scales.gravity
      )

      joint_angles = data.qpos[7:]
      info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_joint_angles = (
          joint_angles
          + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
          * self._config.noise_config.level
          * self._config.noise_config.scales.joint_pos
      )

      joint_vel = data.qvel[6:]
      info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_joint_vel = (
          joint_vel
          + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
          * self._config.noise_config.level
          * self._config.noise_config.scales.joint_vel
      )


      state = jp.hstack([
          # noisy_gyro,  # 3
          noisy_gravity,  # 3
          noisy_joint_angles - self._default_pose,  # 23
          noisy_joint_vel,  # 23
          info["last_act"],  # 23
      ])

      accelerometer = self.get_accelerometer(data, "pelvis")
      root_height = data.qpos[2]

      privileged_state = jp.hstack([
          state,
          gyro,  # 3
          accelerometer,  # 3
          gravity,  # 3
          # linvel,  # 3
          # global_angvel,  # 3
          joint_angles - self._default_pose,
          joint_vel,
          root_height,  # 1
          data.actuator_force,  # 23
          contact,  # 2
          # feet_vel,  # 4*3
      ])

      return {
          "state": state,
          "privileged_state": privileged_state,
      }
    
  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    up = data.site_xmat[self._pelvis_imu_site_id] @ jp.array([0.0, 0.0, 1.0])
    joint_torques = data.actuator_force
    # torso_height = data.site_xpos[self._imu_site_id][2]
    return {
        "height": self._cost_base_height(data.qpos[2]),
        "orientation": self._reward_orientation(
            up, self._desired_up_vec
        ),
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
        "stay_still": self._cost_stay_still(data.qvel[:6]),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "action_rate": self._cost_action_rate(action, info),
        "torques": self._cost_torques(joint_torques),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),

    }
  

  def _reward_orientation(
      self, imu_up_vec: jax.Array, world_up_vec: jax.Array
  ) -> jax.Array:
    cos_dist = jp.dot(imu_up_vec, world_up_vec)
    normalized = 0.5 * cos_dist + 0.5
    # return jp.square(normalized)
    return normalized
  

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    shortfall = self._config.reward_config.base_height_target - base_height
    penalty_base = jp.clip(shortfall, a_min=0.0)
    return  jp.square(penalty_base)
 

  def _cost_stay_still(self, qvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qvel[:2])) + jp.square(qvel[5])

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    # return jp.sum(jp.square(qpos - self._default_pose))
    return jp.sum(jp.abs(qpos - self._default_pose))
 
  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
        return jp.sum(jp.square(act - info["last_act"]))
  
  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def get_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GRAVITY_SENSOR}_{frame}"
      )

  def get_global_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_LINVEL_SENSOR}_{frame}"
    )

  def get_global_angvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_ANGVEL_SENSOR}_{frame}"
    )

  def get_local_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.LOCAL_LINVEL_SENSOR}_{frame}"
    )

  def get_accelerometer(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.ACCELEROMETER_SENSOR}_{frame}"
    )

  def get_gyro(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GYRO_SENSOR}_{frame}"
    )

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model