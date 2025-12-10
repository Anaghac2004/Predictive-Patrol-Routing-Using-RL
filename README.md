# Predictive-Patrol-Routing-Using-RL
A Reinforcement Learning (PPO-based) patrol optimization system designed to improve police route planning using spatio-temporal crime forecasting.

### Objective  
This project develops a reinforcement learning framework to optimize police patrol routes across a 10×10 urban grid. Crime intensities are simulated and enhanced using Kernel Density Estimation (KDE) to model spatio-temporal risk. A Proximal Policy Optimization (PPO) agent is trained to learn patrol paths that prioritize high-risk areas by considering crime severity, forecasted risk, and coverage efficiency.
Built using Gymnasium and Stable-Baselines3, the system demonstrates how AI can support proactive policing through adaptive route planning.

### Skills Learned  
- Reinforcement Learning foundations (states, actions, rewards, episodic interaction)
- Proximal Policy Optimization (PPO) and Actor–Critic methods
- Custom environment design in OpenAI Gymnasium
- Crime risk simulation & KDE-based forecasting
- Visualization using Matplotlib (heatmaps, patrol path animations)
- Reward shaping, exploration strategies, and PPO hyperparameter tuning
- Performance evaluation using reward curves & grid coverage metrics

### Tools & Technologies Used  
- Python 3.x
- Gymnasium – Custom RL environment
- Stable-Baselines3 (PPO) – Agent training
- NumPy, SciPy – KDE, noise simulation, state encoding
- Matplotlib – Heatmaps & agent movement visualization
- Jupyter Notebook / VS Code – Experimentation & debugging

### Key Features  
- 10×10 dynamic crime grid with temporal Gaussian noise
- Severity-aware reward system (robbery > assault > burglary > vandalism > theft)
- PPO agent trained for 200 episodes
- Coverage tracking, heatmap visualization, path animation
- Flattened risk map + normalized agent position as observations
- Five-action movement: up, down, left, right, stay
- Performance:
     - Mean Reward (last 50 episodes): 110.75
     - Max Coverage: 37–60% depending on episode

### Demo Screenshots  

**Ref 1: Crime Risk Grid & Trained PPO Agent in Action**  
- https://github.com/Anaghac2004/Predictive-Patrol-Routing-Using-RL/blob/main/Screenshot%202025-10-10%20101034.png
- https://github.com/Anaghac2004/Predictive-Patrol-Routing-Using-RL/blob/main/Screenshot%202025-10-10%20101103.png
- https://github.com/Anaghac2004/Predictive-Patrol-Routing-Using-RL/blob/main/Screenshot%202025-10-10%20101206.png
*PPO agent navigating the 10×10 crime-risk grid, prioritizing high-risk (red) zones based on KDE-predicted intensities and severity-weighted rewards.*


**Ref 2: Interactive Controls & Info Panel**  
(https://github.com/vaibhavv2004/ParkSmart-RL/blob/main/Screenshot%202025-10-13%20115654.png)
*Episode count, last reward, and selected slot details*

Reporting
Finally, all findings were documented, including detailed descriptions of vulnerabilities, exploits, and remediation recommendations.

**Ref 3: Final Project Report**
https://github.com/Anaghac2004/Predictive-Patrol-Routing-Using-RL/blob/main/CB.PS.I5DAS22004-RL_REPORT.doc

**Ref 4: Coding part**
https://github.com/Anaghac2004/Predictive-Patrol-Routing-Using-RL/blob/main/chicago.py
