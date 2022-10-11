import torch
import math

    
class Config:
    def __init__(self):     # 
         
        self.seed = 10   # numpy和torch的随机数种子

    #----------环境相应参数------------

        self.max_episode_steps=500
        self.render= False
        self.carla_port= 2000
        self.changing_weather_speed= 0.1
        self.frame_skip= 1
        self.observations_type= 'state'
        self.traffic = True
        self.vehicle_name = 'tesla.cybertruck'
        self.map_name = 'Town05'
        self.autopilot = True

        self.num_parallel_envs = 1
        self.carla_ports= [2000]
        self.carla_tm_ports = [3000]

        self.test_carla_port= 2000
        self.test_carla_tm_port= 3000


    #----------------------------------
        self.obs_dim = 9
        self.action_dim = 2
        
        self.train_total_steps = 5e10
        self.test_interval_steps = 2e3
        self.save_interval_steps = 4e4
        self.warmup_steps = 4e2
        self.eval_episodes = 3
        self.memory_size = 8e5
        self.batch_size = 256*4
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4

    #-------------------------------------
        self.record_path = "./result"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


