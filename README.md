# RL_project

## Overview
This project implements reinforcement learning for managing energy storage in a data center environment. The main goal is to train and validate an agent to optimize energy transactions based on price data.

## Project Structure
```
RL_project/
├── __pycache__/
├── analysis&baseline.ipynb   # Data analysis and baseline comparisons
├── env.py                    # Data center environment definition
├── plots.ipynb              # Visualization notebooks
├── README.md
├── requirements.txt
├── TQN/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── tqn_main.py         # Main training script
│   ├── tqn_utility.py      # Utility functions
│   ├── tqn_validate.py     # Validation script
│   ├── tqn.py             # TQN agent implementation
│   └── tune.py            # Hyperparameter tuning
├── train.xlsx              # Training dataset
├── trained_agent_best.pkl  # Pre-trained model
└── validate.xlsx          # Validation dataset
```

## Setup

### Requirements
Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Environment
To run the environment with a random agent:
```bash
python main.py --path train.xlsx
```

### Training the Agent
To train the agent:
```bash
python TQN/tqn_main.py
```

### Validating the Agent
To validate the trained agent:
```bash
python TQN/tqn_validate.py
```

## Environment Details
The `DataCenterEnv` class in `env.py` defines the environment with the following key methods:

- `__init__(self, path_to_test_data)`: Initializes the environment with provided dataset
- `step(self, action)`: Executes an action and returns next state, reward, and termination flag
- `observation(self)`: Returns the current state
- `reset(self)`: Resets the environment to initial state

## Analysis Tools
- `analysis&baseline.ipynb`: Contains data analysis and baseline comparisons
- `plots.ipynb`: Contains visualization code