# Predictive-Patrol-Routing-Using-RL
ParkSmart-RL: PPO-based intelligent parking slot selection using reinforcement learning. Agent optimizes cost, distance &amp; availability in a dynamic 12-slot lot. Real-time Pygame visualization with vehicle path tracing, reward heatmap &amp; interactive controls. Built with Stable-Baselines3, Gymnasium and Python.


### Objective  
The ParkSmart-RL project implements an intelligent parking slot selection system using **Proximal Policy Optimization (PPO)** — a state-of-the-art reinforcement learning algorithm. The agent learns to autonomously choose the optimal parking spot in a dynamic 12-slot environment by balancing **cost**, **distance**, and **availability**, simulating real-world smart parking scenarios. This project demonstrates how RL can power decision-making in autonomous vehicles and smart city applications.

### Skills Learned  
- Deep understanding of Reinforcement Learning (RL) and Markov Decision Processes (MDPs)  
- Mastery of Proximal Policy Optimization (PPO) algorithm and policy gradients  
- Designing custom Gymnasium environments for real-world problems  
- Training stable RL agents using Stable-Baselines3  
- Real-time 2D visualization and interactive simulation with Pygame  
- Reward shaping and exploration-exploitation trade-offs  

### Tools & Technologies Used  
- **Python 3.13** – Core programming language  
- **Stable-Baselines3** – PPO implementation  
- **Gymnasium** – RL environment framework  
- **Pygame** – Real-time visualization engine  
- **NumPy** – State processing and normalization  

### Key Features  
- 12-slot dynamic parking environment with random cost, distance & occupancy  
- PPO agent trained for 15,000 timesteps  
- Interactive Pygame visualization (900×500) with:  
  → Vehicle movement & path tracing  
  → Reward-based heatmap (red = bad, green = good)  
  → Real-time slot info (cost, distance, reward)  
- User controls:  
  `SPACE` → Manual PPO decision  
  `A` → Auto mode (continuous autonomous parking)  
  `R` → Reset environment  
  `ESC` → Exit  

### Demo Screenshots  

**Ref 1: Parking Environment & Trained Agent in Action**  
(https://github.com/vaibhavv2004/ParkSmart-RL/blob/main/Screenshot%202025-10-13%20115026.png)
*Agent autonomously selects high-reward (green) slots*


**Ref 2: Interactive Controls & Info Panel**  
(https://github.com/vaibhavv2004/ParkSmart-RL/blob/main/Screenshot%202025-10-13%20115654.png)
*Episode count, last reward, and selected slot details*

Reporting
Finally, all findings were documented, including detailed descriptions of vulnerabilities, exploits, and remediation recommendations.

**Ref 3: Final Project Report**
https://github.com/vaibhavv2004/ParkSmart-RL/blob/main/PROJECT%20REPORT.docx

**Ref 4: Coding part**
https://github.com/vaibhavv2004/ParkSmart-RL/blob/main/RL_ParkSmart.py
