# CABIS : Context-Aware Bug Injection System

🚀 Quick Start
1. Environment Setup

conda create -n cabis python=3.9 -y
conda activate cabis

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies


2. Prepare Data
Place the SolidiFI dataset files in cabis_project/data/solidifi/:


Then preprocess:
bashpython solidifi_preprocessing.py

3. Train CABIS
python train_cabis.py

# Custom configuration
python train_cabis.py --config configs/custom_config.yaml --epochs 150

# Resume from checkpoint
python train_cabis.py --resume cabis_project/models/cabis_epoch_20.pt

4. Inject Vulnerabilities
python inject_vulnerability.py -i contract.sol -o vulnerable.sol -v reentrancy

# Batch processing
python inject_vulnerability.py --batch -i contracts/ -o vulnerable/ -v timestamp_dependence

# With trained model
python inject_vulnerability.py -i contract.sol -o output.sol -v unchecked_call -m cabis_project/models/cabis_best.pt


📁 Project Structure
cabis_project/
├── data/
│   ├── solidifi/         # Raw SolidiFI dataset
│   ├── processed/        # Preprocessed data
│   └── augmented/        # Augmented training data
├── models/               # Saved model checkpoints
├── results/              # Evaluation results
├── logs/                 # Training logs
├── configs/              # Configuration files
└── detection_tools/      # Security analysis tools


🔧 Key Components
Core Modules

cabis_implementation.py - Main CABIS system

CodeEncoder: Multi-level code representation
VulnerabilitySynthesizer: Context-aware bug injection
ExploitabilityVerifier: Validates exploitability


solidifi_preprocessing.py - Data preprocessing

Parses SolidiFI contracts and bug logs
Extracts vulnerability patterns
Creates PyTorch datasets


train_cabis.py - Training script

Adversarial training model
Multi-objective optimization
Checkpoint management


cabis_evaluation.py - Evaluation framework

Diversity metrics
Realism scoring
Detection tool bypass rates


inject_vulnerability.py - CLI tool

Easy vulnerability injection
Batch processing


🧪 Experiments
Run Full Pipeline
bashpython run_cabis_pipeline.py
Individual Evaluations
pythonfrom cabis_evaluation import CABISEvaluator

# Initialize evaluator
evaluator = CABISEvaluator(cabis_model, ['mythril', 'slither'])

# Run evaluation
results = evaluator.evaluate_comprehensive(test_dataset, num_samples=1000)
Metrics Tracked

Diversity Score: Uniqueness of generated patterns
Realism Score: Discriminator + human evaluation
Preservation Score: Functional equivalence
Exploitability Rate: Verified exploitable vulnerabilities
Detection Bypass: Success against security tools
Compilation Rate: Valid Solidity syntax

📊 Experiment Tracking
Weights & Biases
bash# Login to W&B
wandb login

# Training will automatically log to W&B
python train_cabis.py
TensorBoard
bash# View training logs
tensorboard --logdir cabis_project/logs

🛠️ Customization
Configuration
Edit configs/default_config.yaml:
yamlmodel:
  hidden_dim: 768
  num_heads: 12
  num_layers: 6

training:
  batch_size: 32
  learning_rate: 2e-4
  epochs: 150

Update configuration
Add pattern extraction in solidifi_preprocessing.py
Implement injection logic in VulnerabilitySynthesizer
Add detection logic in evaluation
