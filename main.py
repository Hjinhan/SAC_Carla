
import sys
sys.path.append('/home/hjh/code/carla_0.9.11/PythonAPI/carla')
sys.path.append('/home/hjh/code/carla_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
 

import os
import time
import numpy as np
import logging
from tensorboardX import SummaryWriter
 
import carla
import carla_base

from carla_base import CarlaEnv
from replay_memory import ReplayMemory
from env_wrapper import ParallelEnv, LocalEnv
from agent_base import SAC_Model, SAC_Alg, SAC_Agent  # Choose base wrt which deep-learning framework you are using
from config import Config
 

class TrainPipeline(object):
    def __init__(self,config):
        self.config = config
        
        # Parallel environments for training
        self.parallel_envs = ParallelEnv(self.config)

        # env for eval
        # self.eval_env = LocalEnv(self.config)

        # Initialize model, algorithm, agent
        self.model     = SAC_Model(self.config.obs_dim, self.config.action_dim)
        self.algorithm = SAC_Alg( self.model, self.config)
        self.agent     = SAC_Agent(self.algorithm, self.config)

        # Initialize replay_memory
        self.men = ReplayMemory(self.config)

        # tensorboard record
        record_summarywriter_path = os.path.join(self.config.record_path,"runs")
        self.writer = SummaryWriter(record_summarywriter_path)
        
        # logging record
        os.makedirs(os.path.join(self.config.record_path, "output_logger/"), exist_ok=True)
        logging.basicConfig(filename = os.path.join(self.config.record_path, "output_logger/")+"/sac_obsType_" 
                                            + config.observations_type +"_train"+".log", level = logging.INFO)
        logging.info("-----------------Carla_SAC-------------------")


    def train(self):
        total_steps = 0
        last_save_steps = 0
        test_flag = 0
        avg_reward = 0

        obs_list = self.parallel_envs.reset()

        print("total_steps: ",total_steps)
        while total_steps < self.config.train_total_steps:
            # Train episode
            if self.men.size() < self.config.warmup_steps:   # 刚开始时先进行一下预热，往经验池填充一些随即样本
                action_list = [
                    np.random.uniform(-1, 1, size=self.config.action_dim)
                    for _ in range(self.config.num_parallel_envs)
                ]
            else:
                action_list = self.agent.sample(obs_list)   # 并行获取动作
            next_obs_list, reward_list, done_list, info_list = self.parallel_envs.step(
                action_list)

            # Store data in replay memory
            for i in range(self.config.num_parallel_envs):
                self.men.append(obs_list[i], action_list[i], reward_list[i],
                        next_obs_list[i], done_list[i])
    
            obs_list = self.parallel_envs.get_obs()
            total_steps = self.parallel_envs.total_steps
            # Train agent after collecting sufficient data
            if self.men.size() >= self.config.warmup_steps:
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = self.men.sample_batch(
                    self.config.batch_size)
                critic_loss, actor_loss = self.agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                            batch_terminal)
                print('\n total_steps: {}, critic_loss: {}, actor_loss: {}'.format(total_steps, critic_loss, actor_loss))
                logging.info('\n total_steps: {}, critic_loss: {}, actor_loss: {}'.format(total_steps, critic_loss, actor_loss))
    
            # Save agent
            if total_steps > int(self.config.save_interval_steps) and total_steps > last_save_steps + int(1e4):
                self.agent.save(self.config.record_path+'/step_{}_reward_{}.model'.format(total_steps, avg_reward))
                last_save_steps = total_steps

            # # Evaluate episode
            if (total_steps + 1) // self.config.test_interval_steps >= test_flag:
                while (total_steps + 1) // self.config.test_interval_steps >= test_flag:
                    test_flag += 1
                avg_reward = self.run_evaluate_episodes()
                self.writer.add_scalars('episode_reward', {'episode_reward':avg_reward}, total_steps)
 
                logging.info(
                    '\nTotal steps {}, Evaluation over {} episodes, Average reward: {}'
                    .format(total_steps, self.config.eval_episodes, avg_reward))

                print('\nTotal steps {}, Evaluation over {} episodes, Average reward: {}'.
                    format(total_steps, self.config.eval_episodes, avg_reward))


    # Runs policy for 3 episodes by default and returns average reward
    def run_evaluate_episodes(self):  # 3局的reward之和再做平均
        avg_reward = 0.
        for k in range(self.config.eval_episodes):
            obs = self.parallel_envs.reset()
            done = False
            steps = 0
            while not done and steps < self.config.max_episode_steps:
                steps += 1
  
                action = self.agent.predict(obs[0])   # obs跟 train里面的obs获取方式不一样，train里面是通过get_obs 函数获取
                obs, reward, done, _ = self.parallel_envs.step(action)
                avg_reward += reward[0]
        avg_reward /= self.config.eval_episodes
        return avg_reward

def env_test(config):

    env = CarlaEnv(config,
                 config.test_carla_port,config.test_carla_tm_port )   # 每个并行的环境的 carla_port 是不一样的
    print("env.observation_space:",env.observation_space)
    print("env.action_space:",env.action_space)
    print("env.action_space.high:",env.action_space.high)
 
    env.reset()
    steps = 0
    done = False
    while not done and steps < env.max_episode_steps:
        steps += 1
        next_obs, reward, done, info = env.step([1, 0])
        print(next_obs)
    env.close()



if __name__ == "__main__":

    start = time.time()
    config = Config()

    training_pipeline = TrainPipeline(config)
    training_pipeline.train()
    
    # env_test(config)


    end = time.time()
    print("run time:%.4fs" % (end - start))
    

