# RL_project

## Overview
This project implements reinforcement learning for managing energy storage in a data center environment. The main goal is to train and validate an Tabular Q Learning agent to optimize energy transactions based on price data.

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

### Training the Agent
To train the agent:
```bash
python TQN/tqn_main.py
```

### Validating the Agent
To validate the trained agent (currently it's running the best agent):
```bash
python TQN/tqn_validate.py
```

## Analysis Tools
- `analysis&baseline.ipynb`: Contains data analysis and baseline models
- `plots.ipynb`: Contains visualization code