# 🚦 Adaptive Traffic Signal Control Using Deep Reinforcement Learning for Smart City Traffic Optimization

##  Overview
This repository presents an intelligent traffic signal control system powered by **Deep Reinforcement Learning (DRL)** to optimize traffic flow in smart cities. Traditional fixed-time and actuated signal systems struggle to adapt to dynamic congestion patterns. By leveraging DRL, this project enables real-time traffic signal adjustments that improve mobility, reduce waiting time, and minimize overall congestion.

The system is implemented in **Python** using the **CityFlow** traffic simulator and the **traffic-signal-control dataset**.

---

##  Objectives
- Develop an adaptive DRL-based traffic signal control framework.
- Reduce **average waiting time, queue length, and travel time**.
- Compare performance against traditional fixed-time traffic systems.
- Demonstrate system scalability for **smart city applications**.

---

## Key Features
✔ Deep Reinforcement Learning (DQN/PPO/DDPG)  
✔ Realistic simulation using CityFlow dataset  
✔ Custom traffic state and reward design  
✔ Performance visualization and comparison  
✔ Scalable architecture for multi-intersection control  

---


## **Planned Directory Overview**

### **data/**
Contains input configuration files for the traffic simulation:
- **roadnet.json** – Defines the structure of the traffic network.
- **flow.json** – Describes the vehicle flow entering the network.
- **signal_plan_template.txt** – Template for initializing traffic signal timing.

### **src/**
Core source code for the reinforcement learning environment and agents:
- **environment.py** – Custom wrapper around the CityFlow simulator.
- **agent_dqn.py** – Deep Q-Network (DQN) agent implementation.
- **agent_ppo.py** – Proximal Policy Optimization (PPO) agent (optional).
- **train.py** – Script for training RL agents.
- **evaluate.py** – Script for testing and visualizing trained models.

### **results/**
Stores outputs generated during training and evaluation:
- **plots/** – Performance visualizations (e.g., average rewards).
- **logs/** – Saved training logs for analysis and reproducibility.

### **Other Files**
- **README.md** – Project documentation.
- **requirements.txt** – Python dependencies required to run the project.



---

## Dataset
This project uses datasets from the **CityFlow Traffic Signal Control Dataset**:

🔗 https://traffic-signal-control.github.io/dataset.html

Each dataset includes:
- **roadnet.json** – Defines road topology and intersections
- **flow.json** – Vehicle traffic patterns
- **signal_plan_template.txt** – Initial signal phase configurations

---

## Technologies Used
| Component          | Technology        |
|-------------------|------------------|
| Programming       | Python 3.10       |
| Deep RL Framework | PyTorch / TensorFlow |
| Traffic Simulator | CityFlow          |
| Visualization     | Matplotlib, Seaborn |
| Environment Mgmt  | Anaconda / Virtualenv |

---

##  Getting Started
City flow is only compartible with Python 3.10 and below so you dont need a high version of python

### Installation
```bash
git clone https://github.com/MoyeGl/Autonomous-Driving-in-Smart-Cities-DL-for-Traffic-Optimization
cd Autonomous-Driving-in-Smart-Cities-DL-for-Traffic-Optimization
pip install -r requirements.txt
run main.py
