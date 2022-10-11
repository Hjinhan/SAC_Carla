 
import ray
import carla
# import gym
import numpy as np 
 
from carla_base import CarlaEnv
 
class ParallelEnv(object):
    def __init__(self, config): 
        
        ray.shutdown()
        ray.init()

        self.env_list = [
            CarlaRemoteEnv.remote(config=config, carla_port=carla_port,tm_port = config.carla_tm_ports[index])
            for index,carla_port in  enumerate(config.carla_ports)
        ]
        self.env_num = config.num_parallel_envs
        self.episode_reward_list = [0] * self.env_num
        self.episode_steps_list = [0] * self.env_num
        self._max_episode_steps = config.max_episode_steps
        self.total_steps = 0
   
    def reset(self):
        obs_list = [env.reset.remote() for env in self.env_list]
        obs_list = [ray.get(obs) for obs in obs_list]
        self.obs_list = np.array(obs_list)
        return self.obs_list

    def step(self, action_list):
        return_list = [
            self.env_list[i].step.remote(action_list[i]) for i in range(self.env_num)
        ]
        return_list = ray.get(return_list)     # 
        return_list = np.array(return_list, dtype=object)
        self.next_obs_list = return_list[:, 0]      # 这样出来的最外一层会是列表
        self.reward_list = return_list[:, 1]
        self.done_list = return_list[:, 2]
        self.info_list = return_list[:, 3]

        return self.next_obs_list, self.reward_list, self.done_list, self.info_list

    def get_obs(self):
        for i in range(self.env_num):
            self.total_steps += 1
            self.episode_steps_list[i] += 1
            self.episode_reward_list[i] += self.reward_list[i]

            self.obs_list[i] = self.next_obs_list[i]
            if self.done_list[i] or self.episode_steps_list[
                    i] >= self._max_episode_steps:
                # tensorboard.add_scalar('train/episode_reward_env{}'.format(i),
                #                        self.episode_reward_list[i],
                #                        self.total_steps)
                # logger.info('Train env {} done, Reward: {}'.format(
                #     i, self.episode_reward_list[i]))

                self.episode_steps_list[i] = 0
                self.episode_reward_list[i] = 0
                obs_list_i = self.env_list[i].reset.remote()
                self.obs_list[i] = ray.get(obs_list_i)
                self.obs_list[i] = np.array(self.obs_list[i])
        return self.obs_list
 
@ray.remote
class CarlaRemoteEnv(object):
    def __init__(self, config, carla_port, tm_port):

        class ActionSpace(object):
            def __init__(self,
                         action_space=None,
                         low=None,
                         high=None,
                         shape=None,
                         n=None):
                self.action_space = action_space
                self.low = low
                self.high = high
                self.shape = shape
                self.n = n    # 表示有几个动作
 
            def sample(self):
                return self.action_space.sample()
         
        self.config = config
        self.env = CarlaEnv(config,
                         carla_port,tm_port)   # 每个并行的环境的 carla_port 是不一样的

        self.low_bound = self.env.action_space.low[0]
        self.high_bound = self.env.action_space.high[0]     # 这里出来的只是一个数值
    
        self._max_episode_steps = self.config.max_episode_steps
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)
            
    def reset(self):
        obs = self.env.reset()
        return obs
 
    def step(self, model_output_act):
        """
        Args:
            model_output_act(np.array): The values must be in in [-1, 1].
        """
        assert np.all(((model_output_act<=1.0 + 1e-3), (model_output_act>=-1.0 - 1e-3))), \
            'the action should be in range [-1.0, 1.0]'
        mapped_action = self.low_bound + (model_output_act - (-1.0)) * (
            (self.high_bound - self.low_bound) / 2.0)
        mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)

        return self.env.step(mapped_action)


#------------------------------------------------------- 暂时不使用*----
class LocalEnv(object):
    def __init__(self, config):
        """Map action space [-1, 1] of model output to new action space
        [low_bound, high_bound].
        """
  
        self.env = CarlaEnv(
                 config,
                 config.eval_carla_port, config.eval_carla_tm_port,   # 每个并行的环境的 carla_port 是不一样
                   )
        self._max_episode_steps = config.max_episode_steps
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        # print('local  action_dim',self.action_dim)

        self.low_bound = self.env.action_space.low[0]
        self.high_bound = self.env.action_space.high[0]     # 这里出来的只是一个数值

    def reset(self):
        obs, _ = self.env.reset()
        return obs


    def step(self, model_output_act):
        """
        Args:
            model_output_act(np.array): The values must be in in [-1, 1].
        """
        assert np.all(((model_output_act<=1.0 + 1e-3), (model_output_act>=-1.0 - 1e-3))), \
            'the action should be in range [-1.0, 1.0]'
        mapped_action = self.low_bound + (model_output_act - (-1.0)) * (
            (self.high_bound - self.low_bound) / 2.0)
        mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)

        return self.env.step(mapped_action)