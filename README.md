<img src="https://apre.it/wp-content/uploads/2021/01/logo_uni-pisa.png" width="150" />  

# Meta Learning for Online Continual Learning (Meta-OCL) 

### Thesis Project in Continual Learning (CL)

[@Unipisa](@unipisa) - _Computer Science Department_   
_M.Sc. Computer Science - Artificial Intelligence_

**Author:** Irene Pisani [^1]



## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation-and-usage)
- [Usage](#usage)
- [Project Structure](#project-structure)



## Introduction 

This repository contains the source code and documentation for my thesis project titled **"[Your Thesis Title]"**. The goal of this project is to [briefly describe the goal of the project]. 

This thesis focuses on [brief overview of the problem you are solving], and aims to provide [a summary of your solution or contribution].

**Objectives, goal, and features** - This are the objective:
- [Objective 1] 
- [Objective 2]
- [Objective 3]

**Tools and frameworks** - The project is built using the following technologies:
- Programming Language: [e.g., Python, Java]
- Libraries/Frameworks: [e.g., TensorFlow, Flask, SciPy]
- Tools: [e.g., Docker, Git, Jupyter Notebooks]
- [Any other important tools/techniques used]



## Installation and Usage

Follow these instruction to set up the project on local machine:

1. Clone this repository:
   ```
   git clone https://github.com/AlbinSou/ocl_survey.git
   ```
2. Create a new environment with python 3.10
   ```
   conda create -n ocl_survey python=3.10
   conda activate ocl_survey
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set your PYTHONPATH as the root of the project
   ```
   conda env config vars set PYTHONPATH=/home/.../ocl_survey
   ```
5. Create a `deploy config` in order to specity where the results should be stored, the datasets fetched and data logged. Add a new `deploy config` or change the content of `config/deploy/default.yaml`.

6.  Test the environment by launching one of the script contained in `experiment/` directory.


## Usage 

To run the project, follow these instructions:

* **[How to start the project]**
   Launch main.py to test the environment as follow:
   ```
   cd experiments/
   python main.py strategy=er experiment=split_cifar100
   ```
* **[How to run experiments or analysis]**
  To launch an experiment, start from the default config file and change the part that needs to change
   ```
   python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel
   ```
   Before running the script, you can display the full config with "-c job" option
   ```
   python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel -c job
   ```
   It's also possible to override more fine-grained arguments
   ```
   python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel strategy.alpha=0.7 optimizer.lr=0.05
   ```
   Finally, to use the parameters found by the hyperparameter search, use
   ```
   python main.py strategy=er_ace experiment=split_cifar100 +best_configs=split_cifar100/er_ace
   ```

* **[How to interpret or visualize results]** Results will be saved in the directory specified in results.yaml. Under the following structure:`<results_dir>/<strategy_name>_<benchmark_name>/<seed>/`.

* **[How to run hyperparameter selection experiments]**
  Modify the strategy specific search parameters, search range etc ... inside main_hp_tuning.py then run
   ```
   python main_hp_tuning.py strategy=er_ace experiment=split_cifar100
   ```
* **[How to run experiments with gradients analysis]**
* **[How to ...]**


## Project Structure

Here’s a quick overview of the files and directories:

```
meta-ocl/
├── avalanche.git/           # AVALANCHE - EXTERNAL LIBRARY
├── config/                  # HYDRA - CONFIGURATION FILES
│   ├── benchmark/              # Config for benchmarks
│   ├── best_configs/           # Config for best configs found by main_hp_tuning.py
│   ├── deploy/                 # Config for deploy (machine specific results and data path)
│   ├── evaluation/             # Config to manage evaluation frequency and parrallelism
│   ├── experiment/             # Config to manage general experiment settings
│   ├── model/                  # Config for models
│   ├── optimizer/              # Config for optimizer
│   ├── scheduler/              # Config for scheduler
│   └── strategy/               # Config for CL strategy
├── experiments              # PROJECT EXPERIMENTS 
│   ├── main_hp_tuning.py       # Hyperparameter optimization
│   ├── main.py                 # Launch single experiments
│   ├── gradient_insight.py     # [NEW] Gradients analysis 
│   └── spaces.py               # Additional script
├── notebooks/                  # Additional utilities
├── results/                 # EXPERIMENTAL RESULTS
├── scripts/                    # Scripts for additional functions
│   └── get_results.py          # Collect results from multiple seeds
├── src/                     # MAIN - SOURCE CODE                      
│   ├── factories/              # Scripts for benchmark, method, and model creation
│   ├── strategies/             # Scripts for additional strategies or plugins
│   └── toolkit/                # Scripts for additional functions
└── test/                       # Additional test 
├── LICENSE                     # Licence
├── READM.me                    # Project description
├── enviromental.yaml           # Environment
└── requirements.txt            # List of dependencies
                  
```

## Credits, Licence and Ackwnoledge

Portions of the code used in this project are derived from [Name of the Project](), a project that is still actively maintained. Full credit goes to the original authors for their contributions, upon which this work is based.  

Code is available under MIT license.[^2] See [LICENSE](LICENSE) for the full license.

---

[^1]: Contact me at my [istitutional email address](mail-to:i.pisani1@studenti.unipi.it). 
[^2]: A short and simple permissive license with conditions only requiring preservation of copyright and license notices.
