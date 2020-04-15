from ddpg_agent import Agent
import numpy as np
import torch

class MADDPG():
    def __init__(self, num_agents, state_size, action_size, random_seed):
        self.agents = []
        for _ in range(0, num_agents):
            agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)
            self.agents.append(agent)
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
    
    
    def act(self, states):
        actions = []
        for i, state in enumerate(states):
            state = np.array([state])
            action = self.agents[i].act(state)
            action = np.clip(np.squeeze(action), -1, 1)
            actions.append(action)
        
        return np.array(actions)
            
    def step(self, states, actions, rewards, next_states, dones):
        for i, state in enumerate(states):
            state = np.array([state])
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            self.agents[i].step(state, action, reward, next_state, done)
    
    def save(self):
        for i, agent in enumerate(self.agents):    
            torch.save(agent.actor_local.state_dict(), 'agent' + str(i+1) + '_actor_local.pth')
            torch.save(agent.critic_local.state_dict(), 'agent' + str(i+1) + '_critic_local.pth')
        

    def load(self):
        for i, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('agent' + str(i+1) + '_actor_local.pth'))
            agent.critic_local.load_state_dict(torch.load('agent' + str(i+1) + '_critic_local.pth'))

