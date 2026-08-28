#To train Cara using Reinforcement Learning (RL), a script that defines the Environment, Observations (what #Cara "sees"), and Rewards (what Cara "wants").

 
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

@configclass
class CaraLocomotionCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        # 1. THE SCENE: Spawn Cara (your USD file) and the floor
        self.scene.robot = CARA_CFG.replace(prim_path="{ENV_REGEX}/Robot")
        self.scene.terrain = TerrainImporterCfg(prim_path="/World/ground")

        # 2. OBSERVATIONS: What the Orin Nano "perceives" in sim
        self.observations.policy.joint_pos = ObsTerm(func=get_joint_pos) # 20 angles
        self.observations.policy.joint_vel = ObsTerm(func=get_joint_vel) # 20 speeds
        self.observations.policy.base_lin_vel = ObsTerm(func=get_base_lin_vel) # For balance

        # 3. ACTIONS: The 20 target positions for your PCA9685 boards
        self.actions.joint_pos = JointPositionActionCfg(asset_name="robot", joint_names=[".*"])

        # 4. REWARDS: The "Brain" of the operation
        self.rewards.track_lin_vel = RewTerm(func=track_lin_vel_exp, weight=1.0) # Walk forward!
	self.rewards.expressive_waddle = RewTerm(func=expressive_waddle_reward, weight=0.2)
	self.rewards.track_lin_vel = RewTerm(func=track_lin_vel_exp, weight=1.0)
	self.rewards.homeostasis = RewTerm(func=thermal_penalty, weight=-0.4)
	
    def expressive_waddle_reward(env):
    	# 1. Encourage Torso Roll (The "Waddle" side-to-side)
   	 # We reward the robot for having a rhythmic roll velocity
   	 torso_roll_vel = env.scene.robot.data.body_ang_vel_w[:, 0, 0] # Torso link, X-axis
   	 waddle_reward = torch.abs(torso_roll_vel)
    
    	# 2. Shoulder-Hip Sync (Expressive arms)
   	 # We reward arm movement that scales with leg movement
    	arm_vel = torch.norm(env.scene.robot.data.joint_vel[:, SHOULDER_INDICES], dim=1)
    	leg_vel = torch.norm(env.scene.robot.data.joint_vel[:, HIP_INDICES], dim=1)
  	sync_reward = arm_vel * leg_vel # Arms move more when legs move more
    
   	 return waddle_reward + (0.5 * sync_reward)
