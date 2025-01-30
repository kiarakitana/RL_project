# RL_project

## Overview
This project is focused on reinforcement learning for managing energy storage in a data center environment. The main components of the project include training and validating an agent to optimize energy transactions based on given data.

## Project Structure
RL_project/
├── __pycache__/
├── analysis&baseline.ipynb
├── env.py
├── main.py
├── plots.ipynb
├── README.md
├── requirements.txt
├── TQN/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── TQN/tqn_main.py
│   ├── tqn_utility.py
│   ├── TQN/tqn_validate.py
│   ├── tqn.py
│   ├── tune.py
├── train.xlsx
├── trained_agent_best.pkl
├── validate.xlsx


### Key Files and Directories

- `env.py`: Contains the `DataCenterEnv` class, which defines the environment for the reinforcement learning agent.
- `main.py`: Script to run the environment with a random agent for testing purposes.
- `TQN/`: Directory containing various scripts related to training and validating the agent.
  - `tqn_main.py`: Main script for training the agent.
  - `tqn_validate.py`: Script for validating the trained agent.
  - `tqn_utility.py`: Utility functions used in training and validation.
  - `tqn.py`: Defines the TQN agent.
  - `tune.py`: Script for hyperparameter tuning.
- `analysis&baseline.ipynb`: Jupyter notebook for data analysis and baseline comparisons.
- `plots.ipynb`: Jupyter notebook for generating plots.
- `train.xlsx`: Training dataset.
- `validate.xlsx`: Validation dataset.
- `trained_agent_best.pkl`: Serialized trained agent.

## Setup

### Requirements
Install the required Python packages using `requirements.txt`:
```sh
pip install -r requirements.txt

Running the Code
Running the Environment
To run the environment with a random agent:
python main.py --path train.xlsx

Training the Agent
To train the agent, run:
python TQN/tqn_main.py

Validating the Agent
To validate the trained agent, run:
python TQN/tqn_validate.py

Environment Details:
The DataCenterEnv class in env.py defines the environment. Key methods include:

__init__(self, path_to_test_data): Initializes the environment with the provided dataset.
step(self, action): Executes a step in the environment based on the given action.
observation(self): Returns the current state of the environment.
reset(self): Resets the environment to the initial state.

Notebooks:
analysis&baseline.ipynb: Contains data analysis and baseline comparisons.
plots.ipynb: Contains code for generating plots.
