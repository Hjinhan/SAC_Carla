
import numpy as np

__all__ = ['ReplayMemory']
 
 
class ReplayMemory(object):
    def __init__(self, config):
        """ create a replay memory for off-policy RL or offline RL.

        Args:
            max_size (int): max size of replay memory
            obs_dim (list or tuple): observation shape
            act_dim (list or tuple): action shape
        """
        self.max_size = int(config.memory_size)
        self.obs_dim = config.obs_dim
        self.act_dim = config.action_dim

        self.obs = np.zeros((self.max_size, self.obs_dim), dtype='float32')
        if self.act_dim == 0:  # Discrete control environment
            self.action = np.zeros((self.max_size, ), dtype='int32')
        else:  # Continuous control environment
            self.action = np.zeros((self.max_size, self.act_dim), dtype='float32')
        self.reward = np.zeros((self.max_size, ), dtype='float32')
        self.terminal = np.zeros((self.max_size, ), dtype='bool')
        self.next_obs = np.zeros((self.max_size, self.obs_dim), dtype='float32')
 
        self._curr_size = 0
        self._curr_pos = 0

    def sample_batch(self, batch_size):
        """ sample a batch from replay memory

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def make_index(self, batch_size):
        """ sample a batch of indexes

        Args:
            batch_size (int): batch size

        Returns:
            batch of indexes
        """
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        return batch_idx

    def sample_batch_by_index(self, batch_idx):
        """ sample a batch from replay memory by indexes

        Args:
            batch_idx (list or np.array): batch indexes

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def append(self, obs, act, reward, next_obs, terminal):
        """ add an experience sample at the end of replay memory

        Args:
            obs (float32): observation, shape of obs_dim
            act (int32 in Continuous control environment, float32 in Continuous control environment): action, shape of act_dim
            reward (float32): reward
            next_obs (float32): next observation, shape of obs_dim
            terminal (bool): terminal of an episode or not
        """
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obs[self._curr_pos] = obs
        self.action[self._curr_pos] = act
        self.reward[self._curr_pos] = reward
        self.next_obs[self._curr_pos] = next_obs
        self.terminal[self._curr_pos] = terminal
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        """ get current size of replay memory.
        """
        return self._curr_size

    def __len__(self):
        return self._curr_size

   
