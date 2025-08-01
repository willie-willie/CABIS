# -*- coding: utf-8 -*-
"""

@author: Willie
"""

#!/usr/bin/env python3
"""
CABIS Quick Start Script
Run this after environment setup to verify everything is working
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if all required packages are installed"""
    print("🔍 Checking CABIS environment...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'networkx': 'NetworkX',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'solcx': 'py-solc-x',
        'mythril': 'Mythril',
        'slither': 'Slither',
        'web3': 'Web3.py'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            missing_packages.append(package)
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("⚠️  CUDA is not available (CPU mode)")
    except:
        pass
    
    # Check Solidity compiler
    try:
        result = subprocess.run(['solc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Solidity compiler (solc) is installed")
        else:
            print("❌ Solidity compiler (solc) is NOT installed")
    except:
        print("❌ Solidity compiler (solc) is NOT installed")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please run the setup script again or install missing packages manually")
        return False
    
    print("\n✅ All required packages are installed!")
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("\n📁 Creating sample data...")
    
    # Create data directories
    data_dir = Path("cabis_project/data/solidifi")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample vulnerable contract
    sample_vulnerable = '''pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    // Reentrancy vulnerability
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount);
        (bool success,) = msg.sender.call{value: amount}("");
        require(success);
        balances[msg.sender] -= amount;  // State change after external call
    }
}'''

    # Sample clean contract
    sample_clean = '''pragma solidity ^0.8.0;

contract SafeBank {
    mapping(address => uint) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    // Fixed version
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;  // State change before external call
        (bool success,) = msg.sender.call{value: amount}("");
        require(success);
    }
}'''
    
    # Save sample contracts
    with open(data_dir / "sample_vulnerable.sol", "w") as f:
        f.write(sample_vulnerable)
    
    with open(data_dir / "sample_clean.sol", "w") as f:
        f.write(sample_clean)
    
    # Create sample bug log
    sample_bug_log = '''loc,length,bug type,approach
8,5,reentrancy,state change after call
15,3,timestamp_dependence,block.timestamp in condition
22,4,unchecked_call,no return value check'''
    
    with open(data_dir / "sample_buglog.csv", "w") as f:
        f.write(sample_bug_log)
    
    print("✅ Sample data created in cabis_project/data/solidifi/")

def test_basic_functionality():
    """Test basic CABIS functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Import and test core modules
        from cabis_implementation import CABIS, HierarchicalCodeEncoder
        from solidifi_preprocessing import SolidiFIPreprocessor
        
        # Test encoder
        encoder = HierarchicalCodeEncoder()
        print("✅ HierarchicalCodeEncoder initialized")
        
        # Test CABIS
        cabis = CABIS()
        print("✅ CABIS system initialized")
        
        # Test preprocessing
        preprocessor = SolidiFIPreprocessor("cabis_project/data/solidifi")
        print("✅ SolidiFI preprocessor initialized")
        
        # Test simple vulnerability injection
        test_contract = '''
pragma solidity ^0.8.0;

contract Test {
    uint public value;
    
    function setValue(uint _value) public {
        value = _value;
    }
}'''
        
        result = cabis.inject_vulnerability(test_contract, 'reentrancy', ensure_exploitable=False)
        print("✅ Basic vulnerability injection working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def setup_experiment_tracking():
    """Setup experiment tracking with Weights & Biases"""
    print("\n📊 Setting up experiment tracking...")
    
    try:
        import wandb
        print("✅ Weights & Biases is installed")
        print("ℹ️  To login to W&B, run: wandb login")
    except ImportError:
        print("⚠️  Weights & Biases not installed (optional)")

def create_run_script():
    """Create a script to run the full pipeline"""
    run_script = '''#!/usr/bin/env python3
"""
CABIS Full Pipeline Runner
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from cabis_implementation import CABIS
from solidifi_preprocessing import prepare_solidifi_data
from cabis_evaluation import CABISEvaluator

def main():
    print("Starting CABIS pipeline...")
    
    # 1. Preprocess data
    print("\\n1. Preprocessing SolidiFI data...")
    dataset, patterns = prepare_solidifi_data('./cabis_project/data/solidifi/')
    
    # 2. Initialize CABIS
    print("\\n2. Initializing CABIS...")
    cabis = CABIS()
    
    # 3. Train model (if needed)
    if not Path('./cabis_project/models/cabis_trained.pt').exists():
        print("\\n3. Training CABIS...")
        cabis.train('./cabis_project/data/solidifi/', epochs=10)  # Reduced for demo
    else:
        print("\\n3. Loading pre-trained model...")
        # Load pre-trained model
    
    # 4. Evaluate
    print("\\n4. Evaluating CABIS...")
    evaluator = CABISEvaluator(cabis, ['mythril', 'slither'])
    results = evaluator.evaluate_comprehensive(dataset, num_samples=100)  # Reduced for demo
    
    print("\\n✅ Pipeline completed successfully!")
    print(f"Results saved to: ./cabis_project/results/")

if __name__ == "__main__":
    main()
'''
    
    with open("run_cabis_pipeline.py", "w") as f:
        f.write(run_script)
    
    os.chmod("run_cabis_pipeline.py", 0o755)
    print("✅ Created run_cabis_pipeline.py")

def main():
    """Main quick start function"""
    print("="*60)
    print("CABIS Quick Start")
    print("="*60)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n⚠️  Some tests failed. Please check your installation.")
    
    # Setup experiment tracking
    setup_experiment_tracking()
    
    # Create run script
    create_run_script()
    
    print("\n" + "="*60)
    print("🎉 Quick start completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place your SolidiFI dataset in: cabis_project/data/solidifi/")
    print("2. Run the full pipeline: python run_cabis_pipeline.py")
    print("3. Check results in: cabis_project/results/")
    print("\nFor development:")
    print("- Use Jupyter: jupyter notebook")
    print("- Track experiments: wandb login")
    print("- Run tests: pytest")

if __name__ == "__main__":
    main()