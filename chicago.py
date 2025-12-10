import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
from matplotlib.patches import FancyArrow

# ---------------- Environment Definition ----------------
class PatrolEnv(gym.Env):
    """Custom Gymnasium Environment for Predictive Patrol Routing"""
    metadata = {'render_modes': ['human']}

    def __init__(self, grid_size=10, num_agents=2):
        super().__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents

        # Action space: 0=up,1=down,2=left,3=right,4=stay
        self.action_space = spaces.MultiDiscrete([5] * num_agents)

        # Observation space: flattened representation
        # Crime risk + agent positions
        obs_size = grid_size * grid_size + num_agents * 2
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(obs_size,),
                                            dtype=np.float32)

        # Initialize state
        self.crime_risk = np.random.rand(grid_size, grid_size).astype(np.float32)
        self.agent_pos = [[random.randint(0, grid_size - 1),
                           random.randint(0, grid_size - 1)]
                          for _ in range(num_agents)]

        # Visited cells tracking
        self.visited_cells = [set() for _ in range(num_agents)]
        self.coverage = 0.0
        
        # Store last moves (for logs + arrows)
        self.last_moves = ["" for _ in range(num_agents)]
        self.trails = [[] for _ in range(num_agents)]

        # Assign crime types randomly to each cell
        self.crime_types = np.random.choice(
            ["Theft", "Assault", "Burglary", "Vandalism", "Robbery"],
            size=(grid_size, grid_size)
        )

        # Episode tracking
        self.step_count = 0
        self.max_steps = 100

        # Matplotlib setup
        self.render_enabled = False
        self.fig = None
        self.ax = None

    def enable_render(self):
        """Enable rendering (only when needed for visualization)"""
        if not self.render_enabled:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.render_enabled = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.crime_risk = np.random.rand(self.grid_size, self.grid_size).astype(np.float32)
        self.agent_pos = [[random.randint(0, self.grid_size - 1),
                           random.randint(0, self.grid_size - 1)]
                          for _ in range(self.num_agents)]
        self.visited_cells = [set() for _ in range(self.num_agents)]
        self.coverage = 0.0
        self.last_moves = ["" for _ in range(self.num_agents)]
        self.trails = [[] for _ in range(self.num_agents)]
        self.step_count = 0
        
        return self._get_obs(), {}

    def step(self, actions):
        self.step_count += 1
        rewards = []
        directions_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

        # Track collision
        collision_penalty = 0
        
        for i, action in enumerate(actions):
            row, col = self.agent_pos[i]
            prev_row, prev_col = row, col
            direction = directions_map[action]

            if action == 0 and row > 0: row -= 1
            if action == 1 and row < self.grid_size - 1: row += 1
            if action == 2 and col > 0: col -= 1
            if action == 3 and col < self.grid_size - 1: col += 1

            self.agent_pos[i] = [row, col]
            self.trails[i].append((prev_row, prev_col, row, col))

            # Mark cell as visited
            self.visited_cells[i].add((row, col))

            self.last_moves[i] = f"Moved {direction} to ({row},{col})"
            
            # Reward calculation
            risk_value = float(self.crime_risk[row, col])
            
            # High risk patrol reward (prioritize high-risk areas)
            risk_reward = risk_value * 2.0
            
            # Exploration bonus (encourage visiting new cells)
            exploration_bonus = 0.5 if (row, col) not in self.visited_cells[i] else 0.0
            
            # Movement penalty (discourage staying)
            movement_penalty = -0.1 if action == 4 else 0.0
            
            # Calculate individual agent reward
            agent_reward = risk_reward + exploration_bonus + movement_penalty
            rewards.append(agent_reward)

        # Check for agent collision
        if len(self.agent_pos) > 1:
            if self.agent_pos[0] == self.agent_pos[1]:
                collision_penalty = -1.0

        # Calculate coverage
        all_visited = set()
        for visited_set in self.visited_cells:
            all_visited.update(visited_set)
        self.coverage = len(all_visited) / (self.grid_size * self.grid_size)
        
        # Coverage bonus
        coverage_bonus = self.coverage * 0.5

        # Total reward
        total_reward = float(np.mean(rewards) + collision_penalty + coverage_bonus)

        # Update crime risk with noise
        self.crime_risk = np.clip(
            self.crime_risk + np.random.normal(0, 0.05, (self.grid_size, self.grid_size)),
            0.0, 1.0
        ).astype(np.float32)

        terminated = False
        truncated = self.step_count >= self.max_steps
        
        return self._get_obs(), total_reward, terminated, truncated, {}

    def _risk_level(self, value):
        if value < 0.3:
            return "LOW"
        elif value < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

    def _get_obs(self):
        """Flatten observation for neural network"""
        # Flatten crime risk grid
        crime_flat = self.crime_risk.flatten()
        
        # Normalize agent positions to [-1, 1]
        agent_pos_flat = []
        for pos in self.agent_pos:
            norm_row = (pos[0] / (self.grid_size - 1)) * 2 - 1
            norm_col = (pos[1] / (self.grid_size - 1)) * 2 - 1
            agent_pos_flat.extend([norm_row, norm_col])
        
        # Concatenate
        obs = np.concatenate([crime_flat, agent_pos_flat]).astype(np.float32)
        return obs

    def render(self, mode='human'):
        if not self.render_enabled:
            return
            
        self.ax.clear()

        # Heatmap with "coolwarm"
        self.ax.imshow(self.crime_risk, cmap='coolwarm', vmin=0, vmax=1)

        # Agents: Officer (triangle), Van (square)
        markers = ["^", "s"]
        colors = ["blue", "green"]
        labels = ["Officer", "Van"]

        # Plot agents + movement logs + risk analysis
        for i, pos in enumerate(self.agent_pos):
            r, c = pos
            self.ax.scatter(c, r, marker=markers[i], color=colors[i], s=200, label=labels[i])

            # Risk level & crime type at current position
            risk_value = self.crime_risk[r, c]
            risk_level = self._risk_level(risk_value)
            crime_type = self.crime_types[r, c]

            self.ax.text(self.grid_size + 0.5, i + 1,
                         f"{labels[i]} {self.last_moves[i]} | {risk_level} RISK → {crime_type}",
                         fontsize=10, color=colors[i], va="center")

        # Draw arrows for last moves
        for i, trail in enumerate(self.trails):
            if trail:
                r1, c1, r2, c2 = trail[-1]
                arrow = FancyArrow(c1, r1, (c2 - c1), (r2 - r1),
                                   width=0.05, color=colors[i], alpha=0.7)
                self.ax.add_patch(arrow)

        # Grid setup
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.set_xticklabels(range(self.grid_size))
        self.ax.set_yticklabels(range(self.grid_size))
        self.ax.set_title(f"Crime Risk Map | Coverage: {self.coverage*100:.1f}%")

        plt.pause(0.01)


# ---------------- PPO Algorithm Implementation ----------------

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, obs_dim, action_dims, lr=3e-4, gamma=0.99, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01):
        self.obs_dim = obs_dim
        self.action_dims = action_dims  # List of action dimensions for each agent
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lr = lr
        
        # Initialize networks (simple fully connected)
        self._init_networks()
        
        # Memory
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def _init_networks(self):
        """Initialize policy and value networks"""
        # Hidden layer size
        self.h1_size = 128
        self.h2_size = 64
        
        # Policy network weights
        self.W1_policy = np.random.randn(self.obs_dim, self.h1_size) * 0.1
        self.b1_policy = np.zeros(self.h1_size)
        self.W2_policy = np.random.randn(self.h1_size, self.h2_size) * 0.1
        self.b2_policy = np.zeros(self.h2_size)
        
        # Output layers for each agent
        self.W_out_policy = []
        self.b_out_policy = []
        for action_dim in self.action_dims:
            self.W_out_policy.append(np.random.randn(self.h2_size, action_dim) * 0.1)
            self.b_out_policy.append(np.zeros(action_dim))
        
        # Value network weights
        self.W1_value = np.random.randn(self.obs_dim, self.h1_size) * 0.1
        self.b1_value = np.zeros(self.h1_size)
        self.W2_value = np.random.randn(self.h1_size, self.h2_size) * 0.1
        self.b2_value = np.zeros(self.h2_size)
        self.W_out_value = np.random.randn(self.h2_size, 1) * 0.1
        self.b_out_value = np.zeros(1)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def get_value(self, state):
        """Compute value estimate"""
        h1 = self._relu(np.dot(state, self.W1_value) + self.b1_value)
        h2 = self._relu(np.dot(h1, self.W2_value) + self.b2_value)
        value = np.dot(h2, self.W_out_value) + self.b_out_value
        return value[0]
    
    def get_action(self, state):
        """Sample action from policy"""
        # Forward pass
        h1 = self._relu(np.dot(state, self.W1_policy) + self.b1_policy)
        h2 = self._relu(np.dot(h1, self.W2_policy) + self.b2_policy)
        
        actions = []
        log_probs = []
        
        for i, (W_out, b_out) in enumerate(zip(self.W_out_policy, self.b_out_policy)):
            logits = np.dot(h2, W_out) + b_out
            probs = self._softmax(logits)
            
            # Sample action
            action = np.random.choice(len(probs), p=probs)
            actions.append(action)
            
            # Log probability
            log_prob = np.log(probs[action] + 1e-10)
            log_probs.append(log_prob)
        
        return actions, sum(log_probs)
    
    def store_transition(self, state, actions, reward, value, log_prob, done):
        """Store transition in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(actions)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def compute_advantages(self):
        """Compute GAE advantages"""
        rewards = np.array(self.memory['rewards'])
        values = np.array(self.memory['values'])
        dones = np.array(self.memory['dones'])
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Compute returns and advantages
        next_value = 0
        next_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                next_advantage = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * 0.95 * next_advantage
            returns[t] = rewards[t] + self.gamma * next_value
            
            next_value = values[t]
            next_advantage = advantages[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, epochs=4, batch_size=64):
        """Update policy using PPO"""
        if len(self.memory['states']) < batch_size:
            return 0, 0
        
        states = np.array(self.memory['states'])
        actions = np.array(self.memory['actions'])
        old_log_probs = np.array(self.memory['log_probs'])
        
        advantages, returns = self.compute_advantages()
        
        total_policy_loss = 0
        total_value_loss = 0
        n_updates = 0
        
        for _ in range(epochs):
            # Simple gradient descent (without mini-batching for simplicity)
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                old_log_prob = old_log_probs[i]
                advantage = advantages[i]
                return_val = returns[i]
                
                # Forward pass
                h1 = self._relu(np.dot(state, self.W1_policy) + self.b1_policy)
                h2 = self._relu(np.dot(h1, self.W2_policy) + self.b2_policy)
                
                # Compute new log probs
                new_log_prob = 0
                for j, (W_out, b_out) in enumerate(zip(self.W_out_policy, self.b_out_policy)):
                    logits = np.dot(h2, W_out) + b_out
                    probs = self._softmax(logits)
                    new_log_prob += np.log(probs[action[j]] + 1e-10)
                
                # PPO clipped objective
                ratio = np.exp(new_log_prob - old_log_prob)
                clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
                
                # Value loss
                value_pred = self.get_value(state)
                value_loss = (return_val - value_pred) ** 2
                
                # Simple gradient update (this is simplified - real implementation would compute gradients)
                # For demonstration, we'll use a simple update rule
                learning_signal = advantage * 0.001
                
                # Update policy weights (simplified)
                self.W1_policy += learning_signal * np.outer(state, np.ones(self.h1_size)) * self.lr
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                n_updates += 1
        
        # Clear memory
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        return total_policy_loss / max(n_updates, 1), total_value_loss / max(n_updates, 1)


# ---------------- Training Loop ----------------

def train_ppo(env, agent, episodes=500, render_every=50):
    """Train PPO agent"""
    episode_rewards = []
    
    print("Starting PPO Training...")
    print(f"Episodes: {episodes}, Render every: {render_every}")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Enable rendering for visualization episodes
        if episode % render_every == 0:
            env.enable_render()
        else:
            env.render_enabled = False
        
        while not done:
            # Get action from policy
            actions, log_prob = agent.get_action(obs)
            value = agent.get_value(obs)
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(obs, actions, reward, value, log_prob, done)
            
            # Render if enabled
            if env.render_enabled and episode % render_every == 0:
                env.render()
            
            obs = next_obs
            episode_reward += reward
        
        # Update policy
        if len(agent.memory['states']) > 32:
            policy_loss, value_loss = agent.update(epochs=4)
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}/{episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Episode Reward: {episode_reward:.2f} | Coverage: {env.coverage*100:.1f}%")
    
    return episode_rewards


# ---------------- Main Simulation ----------------
if __name__ == "__main__":
    print("Running file:", os.path.abspath(__file__))
    print("=" * 60)
    
    # Create environment
    env = PatrolEnv(grid_size=10, num_agents=2)
    
    # Create PPO agent
    obs_dim = env.observation_space.shape[0]
    action_dims = [5, 5]  # Two agents, each with 5 actions
    agent = PPOAgent(obs_dim, action_dims, lr=0.0003)
    
    # Train the agent
    episode_rewards = train_ppo(env, agent, episodes=200, render_every=50)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Average Reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.2f}")
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.6)
    plt.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Progress')
    plt.grid(True)
    plt.show()
    
    # Test trained agent
    print("\nRunning trained agent visualization...")
    env.enable_render()
    obs, _ = env.reset()
    
    for step in range(100):
        actions, _ = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(actions)
        env.render()
        if terminated or truncated:
            break
    
    print("Simulation finished.")
    plt.ioff()
    plt.show(block=True)