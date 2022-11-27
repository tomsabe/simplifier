"""
RL agent to play sentence simplification game

Credits:
Deep Reinforcement Learning Hands-On by Maxim Lapan provided a starting point
Additional code and ideas adopted from cleanrl python project, dqn.py reference code
---> https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

"""

# ISSUES
# Test wandb and video capture
# Consider replacing replay buffer with stable baselines
#   (may make better use of GPU)
# Add epsilon to parama; stop decr epslion at zero

import argparse
import os
import sys
import random
import time
from distutils.util import strtobool

import gym
import gyms
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import jsonlines
import matplotlib.pyplot as plt

import rl_model

HYPERS = {
          "gamma":0.99,
          "batch_size":32,
          "learning_rate":1e-4,
          "epsilon_decay":-1e-4, # had -1e-4 earlier, sort of worked
          "solved_reward":10
}

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--env-id", type=str, default="gyms/TextWorld-v0",
                      help="the id of the environment")
  parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                      help="the name of this experiment")
  parser.add_argument("--seed", type=int, default=1,
                      help="seed of the experiment")
  parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                      help="if toggled, `torch.backends.cudnn.deterministic=False`")
  parser.add_argument("--cuda", default=False,action="store_true")
  parser.add_argument("--net", help="name of previously saved net to use")
  parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                      help="if toggled, this experiment will be tracked with Weights and Biases")
  parser.add_argument("--wandb-project-name", type=str, default="text_simplification",
                      help="the wandb's project name")
  parser.add_argument("--wandb-entity", type=str, default=None,
                      help="the entity (team) of wandb's project")
  parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                      help="whether to capture videos of the agent performances (check out `videos` folder)")
  args = parser.parse_args()
  return args

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
#          with jsonlines.open('logfile.jsonl',mode='a') as writer:
#            writer.write(d)
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
    args = parse_args()
    #Experiment tracking, following clearnrl reference 
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
#            monitor_gym=True,   relies on depricated method
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
#    writer.add_text(
#        "hyperparameters",
#        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in HYPERS])),
#    )

    # set seeds as recommended in cleanrl
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    #Set cuda flag
    device = torch.device("cuda" if args.cuda else "cpu")

    #Set up the gym environment
    env = gym.make(f"{args.env_id}")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.capture_video:
      env = gym.wrappers.RecordVideo(env,f"videos/{run_name}")
#    env.seed(args.seed) - appears to be outdated
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    #Create a policy network and target network
    # Input dimensions match observation space shape
    # Output dimensions match action space shape
    obs_shape = env.observation_space.shape
    net = rl_model.RL(obs_shape,
                      env.action_space.n).to(device)
    tgt_net = rl_model.RL(obs_shape,
                      env.action_space.n).to(device)

    #by default start at epsilon = 1
    epsilon = 1.0

    #if a previously saved network was specified, then load it
    if args.net:
      net.load_state_dict(torch.load(args.net))
      net.eval()
      tgt_net.load_state_dict(torch.load(args.net))
      tgt_net.eval()
      #when using saved network, take down epsilon to 0.5
      epsilon = 0.5

    #otherwise .. should I be Xavier initializing my networks? 
      #--> https://github.com/bentrevett/pytorch-rl/blob/master/1%20-%20Vanilla%20Policy%20Gradient%20(REINFORCE)%20%5BCartPole%5D.ipynb
    
    #create an agent who will act in the environment
    agent = Agent(env)
    
    #prepare for the training loop
    optimizer = optim.Adam(net.parameters(),lr=HYPERS['learning_rate'])
    game_rewards = []
    move = 0
    ts_move = 0
    ts = time.time()
    best_m_reward = None
    
    #run the training loop
    while True: 
      move += 1
      #play one step and get reward
      final_reward = agent.play_step(net, epsilon, device=device)
      if final_reward is not None:  # if this is final step in episode
        #update rewards metrics
        game_rewards.append(final_reward)
        num_games = len(game_rewards)
        avg_game_reward = np.mean(game_rewards[-100:])
        #calculate game speed
        game_time = (time.time() - ts)
        ts = time.time()
        #log progress
        print("%d: moves %d games, avg game score (last 100) %.3f, "
              "epsilon %.2f, speed %.2f seconds/game" % (
                move, num_games, avg_game_reward, epsilon, game_time))
        writer.add_scalar("charts/final_reward", final_reward, move)
        writer.add_scalar("charts/epslion",epsilon,move)
        #If at least 100 games done:  
        if(num_games>100):
          writer.add_scalar("charts/100_game_average", avg_game_reward, move)
          #Check if average score (last 100 games) is a new record .. if so, save model
          if (best_m_reward is None) or (avg_game_reward > best_m_reward):
            torch.save(net.state_dict(),"model_save/TextWorld-avgscore_%.0f.dat" % avg_game_reward)
#            plt.plot(game_rewards, 'o-r')
#            plt.ylabel('Reward')
#            plt.savefig('TextWorld-avgscore_%.0f.pdf' % avg_game_reward)
            if best_m_reward is not None: 
              print("Best reward updated %.3f -> %.3f" % (best_m_reward, avg_game_reward))
            best_m_reward = avg_game_reward
          #If average is high enough, report game solved
          if avg_game_reward > HYPERS['solved_reward']:
            print("Solved in %d moves" % move)
            env.close()
            writer.close()
            sys.exit(0)
      #(now we are done with the "if" for 'final step')

      #sample batches and update network
      optimizer.zero_grad()
      batch = agent._sample_buffer(HYPERS["batch_size"])
      if len(batch)==HYPERS["batch_size"]:
        loss_t = batch_loss(batch, net, tgt_net, device=device)
        writer.add_scalar("losses/td_loss",loss_t.item(), move)
        loss_t.backward()
        optimizer.step()

      #periodically update the target network 
      if move % 128 == 0: 
        tgt_net.load_state_dict(net.state_dict())

      #decrease epslion
      if epsilon > 0:
        epsilon += HYPERS['epsilon_decay']

