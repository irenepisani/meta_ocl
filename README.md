<img src="https://apre.it/wp-content/uploads/2021/01/logo_uni-pisa.png" width="150" />  

# Meta Learning for Online Continual Learning (Meta-OCL) 

### Thesis Project in Continual Learning (CL)

[@Unipisa](@unipisa) - _Computer Science Department_   
_M.Sc. Computer Science - Artificial Intelligence_

**Author:** Irene Pisani [^1]

---

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

Follow these steps to set up the project locally:

1. Clone this repository:

   \`\`\`bash
   git clone https://github.com/yourusername/your-repo-name.git
   \`\`\`

2. Navigate to the project directory:

   \`\`\`bash
   cd your-repo-name
   \`\`\`

3. Install the required dependencies:

   \`\`\`bash
   pip install -r requirements.txt  # If using Python
   \`\`\`

4. [Any other setup instructions]



To run the project, follow these instructions:

1. [Step 1: Explain how to start the project]
2. [Step 2: How to run tests or analysis]
3. [Step 3: How to interpret or visualize the results]

Example command:

\`\`\`bash
python main.py --input data/input_file.csv --output results/output_file.csv
\`\`\`
## Usage

## Project Structure

Here’s a quick overview of the files and directories:

```

│
├── data/                   # Dataset files (not included in repo, if private)
├── src/                    # Source code
│   ├── main.py             # Main script to run the project
│   ├── utils.py            # Utility functions
│   └── model/              # Model scripts
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests for the project
├── results/                # Results from analysis or experiments
├── README.md               # Project documentation (this file)
└── requirements.txt        # List of dependencies

meta-ocl/
│
├── data/                   # Dataset files (not included in repo, if private)
├── src/                    # Source code
│   ├── main.py             # Main script to run the project
│   ├── utils.py            # Utility functions
│   └── model/              # Model scripts
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests for the project
├── results/                # Results from analysis or experiments
├── README.md               # Project documentation (this file)
└── requirements.txt        # List of dependencies
```

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
└── test/
                  
```

## Credits, Licence and Ackwnoledge

Portions of the code used in this project are derived from [Name of the Project](), a project that is still actively maintained. Full credit goes to the original authors for their contributions, upon which this work is based.  

Code is available under MIT license.[^2] See [LICENSE](LICENSE) for the full license.

---

[^1]: Contact me at my [istitutional email address](mail-to:i.pisani1@studenti.unipi.it). 
[^2]: A short and simple permissive license with conditions only requiring preservation of copyright and license notices.
