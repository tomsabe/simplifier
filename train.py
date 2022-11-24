"""
RL agent to play sentence simplification game

Credits:
Deep Reinforcement Learning Hands-On by Maxim Lapan provided a starting point

"""

# ISSUES
# Should consider iterating through full episodes
#   & updating nets after a fixed number of episodes
#   based on step rewards and final rewards of episode

import argparse
import time
import gym
import gyms
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
#from tensorboardX import SummaryWriter
import rl_model
import jsonlines
import matplotlib.pyplot as plt

HYPERS = {
          "gamma":0.99,
          "batch_size":32,
          "learning_rate":1e-4,
          "epsilon_decay":-1e-4, # had -1e-4 earlier, sort of worked
          "solved_reward":10
}

class Agent:
    """RL agent that interacts with environment"""

    def __init__(self, env):
        self.env = env
        self.buffer = pd.DataFrame(columns=['state','action','reward','is_done','next_state'])
        self.game_number = 0
        self._reset()
    
    def _reset(self):
        self.state, info = env.reset()
        self.game_steps = []
        self.game_begin = info
        self.game_number += 1
        self.total_reward = 0.0
    
    def _sample_buffer(self, batch_size):
        num = min(len(self.buffer),batch_size)
        return self.buffer.sample(n=num)

    def _display_env_info(self, info, begin=False):
      if begin:
        d_info = self.game_begin
      else:
        d_info = info
      print(d_info.get('text'))
      print(f"Simple score = {d_info.get('simple_score')}")
      print(f"Semantic score = {d_info.get('semantic_score')}")
      print(f"Total score = {d_info.get('total_score')}")
      return d_info

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        """Take a step and store in buffer"""
#        print(f"Playing step, epsilon = {epsilon}.")
        #If random < epsilon, and we have enough in batch then explore
        if np.random.random() < epsilon:
            action = env.action_space.sample()
            self.game_steps.append(str(action))
        #Otherwise, perform the 'best' learned action
        else:
            state_a = self.state
            state_v = torch.as_tensor(state_a,dtype=torch.float32, device=device)
            q_vals_v = net(state_v) #forward pass in model
            inds = torch.max(q_vals_v,0).indices #get best action
            action = inds.item()
            self.game_steps.append(f"(L){action}")
        #Perform one action in the environment
        new_state, reward, is_done, _, info = self.env.step(action)
        #Accumulate the reward
        self.total_reward += reward
#        print(f"Step reward was +{reward}, game reward = {self.total_reward}\n")
        #Log the result in the buffer
        l = len(self.buffer)
        self.buffer.loc[l] = [self.state,action,reward,is_done,new_state]
        #Advance to the new state
        self.state = new_state
        #If game is over, reset and return cumulative reward
        if is_done:
          done_reward = int(self.total_reward*100)/100
          d = {
              "game_num":self.game_number,
          }
          print("\033c")
          print(f"START OF GAME {self.game_number}.\n")
          d['begin'] = self._display_env_info(info,begin=True)
          print(f"Game steps: {self.game_steps}")
          d['steps'] = self.game_steps
          print("\n\nEND OF GAME:\n")
          d['end'] = self._display_env_info(info)
          print(f"Done_reward = {done_reward}.\n\n\n")
          d['reward']=done_reward
          with jsonlines.open('logfile.jsonl',mode='a') as writer:
            writer.write(d)
          self._reset()
          return done_reward
        else:
          return None
    
def batch_loss(batch, net, tgt_net, device="cpu"):
    """Calculate losses on batch"""
#    print("Here is the batch:")
#    print(batch)
    states = np.array(batch['state'].tolist(),dtype='float32')
#    print(states)
    actions = batch['action'].tolist()
    rewards = batch['reward'].tolist()
    dones = batch['is_done'].tolist()
    next_states = np.array(batch['next_state'].tolist(),dtype='float32')
    #Convert data into tensors
    states_v = torch.as_tensor(states,dtype=torch.float32, device=device)
    next_states_v = torch.as_tensor(next_states,dtype=torch.float32, device=device)
    actions_v = torch.as_tensor(actions,dtype=torch.int64,device=device)
    rewards_v = torch.as_tensor(rewards,dtype=torch.float32,device=device)
    done_mask = torch.as_tensor(dones,dtype=torch.bool, device=device)
    #Pass observations to first model (used to calculate gradients)
    # Gather Q-values for the taken action
    # (gather is performed on dimension 1 which holds actions)
    state_action_values = torch.gather(net(states_v),1,actions_v.unsqueeze(-1))
    #Apply target network to next states
    # calculate maximum Q-value along the action dimension
    next_state_values = torch.max(tgt_net(next_states_v),1).indices
    #If there is transition in batch to last step, set nsv = 0
    next_state_values[done_mask] = 0.0
    #Detach nsv from the computation graph
    # this way backprop will not affect prediction for next state
    # We want to preserve nsv for use in Bellman equation
    #  to calculate reference Q-values
    next_state_values = next_state_values.detach()
    #Calculate the Bellman approximation value
    expected_state_action_values = torch.reshape(
                                                next_state_values * HYPERS['gamma'] \
                                                + rewards_v,
                                                (32,1)
                                                )
    #Return MSE loss
#    print(f"SAV:\n{state_action_values}\nESAV:\n{expected_state_action_values}")
    return nn.MSELoss()(state_action_values,
                            expected_state_action_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true")
    parser.add_argument("--net")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    env = gym.make("gyms/TextWorld-v0")
    #Make new net with correct input size
    obs_shape = env.observation_space.shape
    net = rl_model.RL(obs_shape,
                      env.action_space.n).to(device)
    tgt_net = rl_model.RL(obs_shape,
                      env.action_space.n).to(device)
    #by default start at epsilon = 1
    epsilon = 1.0
    #if a saved network was specified, load it
    if args.net:
      net.load_state_dict(torch.load(args.net))
      net.eval()
      tgt_net.load_state_dict(torch.load(args.net))
      tgt_net.eval()
      #also, take down epsilon to 0.5
      epsilon = 0.5
    #else: Should I be Xavier initializing my networks? 
      #--> https://github.com/bentrevett/pytorch-rl/blob/master/1%20-%20Vanilla%20Policy%20Gradient%20(REINFORCE)%20%5BCartPole%5D.ipynb    
    #Start tensorboard
#    writer = SummaryWriter(comment="gyms/TextWorld-v0")
    #Prepare for training loop
    agent = Agent(env)
    optimizer = optim.Adam(net.parameters(),lr=HYPERS['learning_rate'])
    total_rewards = []
    move = 0
    ts_move = 0
    ts = time.time()
    best_m_reward = None
    #Begin training loop
    while True: 
      move += 1
      #Play a step and get reward
      reward = agent.play_step(net, epsilon, device=device)
      if reward is not None:  # if this is final step in episode
        #Update rewards metrics
        total_rewards.append(reward)
        n_games = len(total_rewards)
        m_reward = np.mean(total_rewards[-100:])
        #Calculate game speed
        game_time = (time.time() - ts)
        ts = time.time()
        #Log progress
        print("%d: moves %d games, avg game score (last 100) %.3f, "
              "epsilon %.2f, speed %.2f seconds/game" % (
                move, n_games, m_reward, epsilon, game_time))
#        writer.add_scalar("avg_score_100", m_reward, move)
#        writer.add_scalar("game_score", reward, move)
        #If at least 100 games done:  
        if(n_games>100):
          #Check if average score (last 100 games) is a new record .. if so, save model
          if (best_m_reward is None) or (m_reward > best_m_reward):
            torch.save(net.state_dict(),"model_save/TextWorld-avgscore_%.0f.dat" % m_reward)
            plt.plot(total_rewards, 'o-r')
            plt.ylabel('Reward')
            plt.savefig('TextWorld-avgscore_%.0f.pdf' % m_reward)
            if best_m_reward is not None: 
              print("Best reward updated %.3f -> %.3f" % (best_m_reward, m_reward))
            best_m_reward = m_reward
          #If average is high enough, report game solved
          if m_reward > HYPERS['solved_reward']:
            print("Solved in %d moves" % move)
            break
      #Now we are done with the "if" for final step
      #For every step: 
      #Update the target network each 128 moves 
      if move % 128 == 0: 
        tgt_net.load_state_dict(net.state_dict())
      #Zero the gradients, get a batch, calculate loss, minimize loss
      #Should I do this every 32 moves? 64 moves? 128 moves?
      optimizer.zero_grad()
      batch = agent._sample_buffer(HYPERS["batch_size"])
      if len(batch)==HYPERS["batch_size"]:
        loss_t = batch_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
      #update epslion
      epsilon += HYPERS['epsilon_decay']

