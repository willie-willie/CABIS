#!/usr/bin/env python3
"""
Complete revised run_cabis_pipeline.py with all fixes
@author: Willie
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import json
import yaml
from datetime import datetime
import time
import logging
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import with try-except for better error handling
try:
    from cabis_implementation import ImprovedCABIS
    from solidifi_preprocessing import prepare_solidifi_data, SolidiFIPreprocessor, VulnerabilityDataset, PatternExtractor, SOLIDIFI_FOLDERS
    from cabis_evaluation import EnhancedCABISEvaluator
    from train_cabis import EnhancedCABISTrainer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are in the same directory")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CABISPipeline:
    """Complete pipeline for CABIS on full SolidiFI dataset"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.start_time = datetime.now()
        self.results = {}
        
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        logger.info("📁 Setting up directory structure...")
        
        paths_config = self.config.get('paths', {})
        dirs = [
            paths_config.get('data_dir', './cabis_project/data/solidifi'),
            paths_config.get('model_dir', './cabis_project/models'),
            paths_config.get('results_dir', './cabis_project/results'),
            paths_config.get('log_dir', './cabis_project/logs'),
            paths_config.get('checkpoint_dir', './cabis_project/checkpoints')
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"   ✅ {dir_path}")
    
    def check_data_availability(self) -> bool:
        """Check if full SolidiFI dataset is available"""
        logger.info("\n📊 Checking SolidiFI dataset availability...")
        
        data_path = Path(self.config['paths']['data_dir'])
        
        if not data_path.exists():
            logger.error(f"❌ Data directory does not exist: {data_path}")
            logger.error("Please create the directory and download the SolidiFI dataset")
            return False
        
        # Check for all 7 vulnerability folders
        solidifi_folders = self.config.get('solidifi_folders', SOLIDIFI_FOLDERS)
        
        expected_folders = list(solidifi_folders.keys())
        found_folders = []
        missing_folders = []
        
        total_sol_files = 0
        total_csv_files = 0
        folder_stats = {}
        
        for folder in expected_folders:
            folder_path = data_path / folder
            if folder_path.exists():
                # Count files with flexible naming patterns
                sol_files = list(folder_path.glob("buggy_*.sol")) + list(folder_path.glob("buggy*.sol"))
                csv_files = (list(folder_path.glob("BugLog_*.csv")) + 
                           list(folder_path.glob("BugLog*.csv")) + 
                           list(folder_path.glob("buglog*.csv")))
                
                sol_count = len(sol_files)
                csv_count = len(csv_files)
                
                total_sol_files += sol_count
                total_csv_files += csv_count
                
                logger.info(f"   ✅ {folder}: {sol_count} .sol, {csv_count} .csv files")
                found_folders.append(folder)
                
                folder_stats[folder] = {
                    'sol_files': sol_count,
                    'csv_files': csv_count,
                    'exists': True
                }
                
                # Show sample files for debugging
                if sol_files:
                    logger.debug(f"      Sample .sol: {[f.name for f in sol_files[:3]]}")
                if csv_files:
                    logger.debug(f"      Sample .csv: {[f.name for f in csv_files[:3]]}")
            else:
                logger.error(f"   ❌ {folder}: Not found!")
                missing_folders.append(folder)
                folder_stats[folder] = {
                    'sol_files': 0,
                    'csv_files': 0,
                    'exists': False
                }
        
        # Check if we have at least some data
        if total_sol_files == 0:
            logger.error("\n❌ No .sol files found in any folder!")
            logger.error("Please check:")
            logger.error("1. The dataset is properly downloaded")
            logger.error("2. The folder structure matches the expected format")
            logger.error(f"3. Files are in: {data_path}")
            return False
        
        if missing_folders:
            logger.warning(f"\n⚠️  Missing folders: {missing_folders}")
            logger.warning("The pipeline will continue with available folders")
        
        # Summary
        logger.info(f"\n📊 Dataset Summary:")
        logger.info(f"   Found {len(found_folders)}/{len(expected_folders)} vulnerability folders")
        logger.info(f"   Total files: {total_sol_files} .sol files, {total_csv_files} .csv files")
        
        if total_csv_files == 0:
            logger.warning("   ⚠️  No CSV files found - will process contracts without bug logs")
        
        self.results['dataset_stats'] = {
            'folders': len(found_folders),
            'total_sol_files': total_sol_files,
            'total_csv_files': total_csv_files,
            'folder_stats': folder_stats
        }
        
        return total_sol_files > 0  # Continue if we have at least some .sol files
    
    def preprocess_data(self) -> Tuple:
        """Preprocess the full SolidiFI dataset"""
        logger.info("\n🔄 Preprocessing SolidiFI dataset...")
        start_time = time.time()
        
        try:
            # First try the standard preprocessing
            dataset, patterns, processed_data = self._standard_preprocessing()
            
            if dataset is None or len(dataset) == 0:
                logger.warning("Standard preprocessing failed, trying recovery mode...")
                dataset, patterns, processed_data = self._recovery_preprocessing()
            
            if dataset is None:
                raise ValueError("Failed to create dataset after all attempts")
            
            # IMPORTANT: Ensure all 7 vulnerability types are present
            logger.info("Ensuring all 7 vulnerability types are represented...")
            self._ensure_all_vulnerability_types(dataset)
            
            # Save preprocessing results
            preprocessing_stats = self._calculate_preprocessing_stats(
                dataset, patterns, processed_data, start_time
            )
            
            self._save_preprocessing_stats(preprocessing_stats)
            self._log_preprocessing_summary(preprocessing_stats)
            
            self.results['preprocessing'] = preprocessing_stats
            
            return dataset, patterns, processed_data
            
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _ensure_all_vulnerability_types(self, dataset):
        """Ensure dataset has all 7 vulnerability types"""
        # Check current distribution
        type_counts = defaultdict(int)
        for i in range(min(1000, len(dataset))):
            try:
                item = dataset[i]
                vuln_type = item.get('vulnerability_type', 'unknown')
                type_counts[vuln_type] += 1
            except:
                pass
        
        logger.info(f"Current distribution: {dict(type_counts)}")
        
        # All expected types
        all_types = ['reentrancy', 'integer_overflow', 'timestamp_dependence', 
                     'unchecked_call', 'tx_origin', 'transaction_order_dependence', 
                     'unhandled_exceptions']
        
        missing_types = [t for t in all_types if type_counts.get(t, 0) == 0]
        
        if missing_types:
            logger.warning(f"Missing vulnerability types: {missing_types}")
            
            # Try to call the dataset's method if it exists
            if hasattr(dataset, 'ensure_all_vulnerability_types'):
                dataset.ensure_all_vulnerability_types()
            else:
                logger.warning("Dataset doesn't have ensure_all_vulnerability_types method")
                logger.warning("Consider updating VulnerabilityDataset class with the fix")
                
                # Create synthetic samples manually
                logger.info("Creating synthetic samples for missing types...")
                if hasattr(dataset, 'pairs'):
                    for vuln_type in missing_types:
                        logger.info(f"Adding synthetic samples for {vuln_type}")
                        for i in range(10):  # Add 10 samples per missing type
                            clean_code, buggy_code = self._create_synthetic_contract(vuln_type)
                            
                            clean = {
                                'code': clean_code,
                                'vulnerability_type': vuln_type,
                                'synthetic': True
                            }
                            
                            buggy = {
                                'code': buggy_code,
                                'vulnerability_type': vuln_type,
                                'synthetic': True,
                                'vulnerability_snippets': []
                            }
                            
                            dataset.pairs.append((clean, buggy))
                    
                    logger.info(f"Added synthetic samples. New dataset size: {len(dataset)}")
    
    def _create_synthetic_contract(self, vuln_type: str) -> Tuple[str, str]:
        """Create synthetic contract pair for missing vulnerability type"""
        # Base clean contract
        clean = """pragma solidity ^0.8.0;
    
    contract Synthetic {
        mapping(address => uint256) balances;
        address owner = msg.sender;
        
        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }
    }"""
        
        # Buggy versions for each type
        buggy_templates = {
            'unchecked_call': clean[:-1] + """
        
        function bug_unchk_send1() public {
            msg.sender.send(1 ether);
        }
    }""",
            'tx_origin': clean[:-1] + """
        
        function bug_txorigin1() public {
            require(tx.origin == owner);
        }
    }""",
            'transaction_order_dependence': clean[:-1] + """
        
        uint256 reward_TOD1;
        function play_TOD1() public {
            if (msg.value > reward_TOD1) {
                reward_TOD1 = msg.value;
            }
        }
    }""",
            'unhandled_exceptions': clean[:-1] + """
        
        function callnotchecked_unchk1(address a) public {
            a.call("");
        }
    }""",
            'reentrancy': clean[:-1] + """
        
        function withdraw_re_ent1() public {
            msg.sender.call{value: balances[msg.sender]}("");
            balances[msg.sender] = 0;
        }
    }""",
            'integer_overflow': clean[:-1] + """
        
        function bug_intou1() public {
            unchecked {
                uint8 x = 255;
                x = x + 1;
            }
        }
    }""",
            'timestamp_dependence': clean[:-1] + """
        
        function bug_tmstmp1() public {
            if (block.timestamp == 1) {
                owner = msg.sender;
            }
        }
    }"""
        }
        
        buggy = buggy_templates.get(vuln_type, clean)
        return clean, buggy
    
    def _standard_preprocessing(self) -> Tuple:
        """Standard preprocessing approach"""
        try:
            preprocessor = SolidiFIPreprocessor(self.config['paths']['data_dir'])
            processed_data = preprocessor.process_dataset()
            
            dataset, patterns = prepare_solidifi_data(self.config['paths']['data_dir'])
            
            return dataset, patterns, processed_data
        except Exception as e:
            logger.error(f"Standard preprocessing error: {str(e)}")
            return None, None, None
    
    def _recovery_preprocessing(self) -> Tuple:
        """Recovery preprocessing when standard fails"""
        logger.info("Attempting recovery preprocessing...")
        
        preprocessor = SolidiFIPreprocessor(self.config['paths']['data_dir'])
        processed_data = preprocessor.process_dataset()
        
        buggy_contracts = processed_data.get('buggy', [])
        clean_contracts = processed_data.get('clean', [])
        
        # Generate clean contracts if needed
        if not clean_contracts and buggy_contracts:
            logger.info(f"Generating clean contracts from {len(buggy_contracts)} buggy contracts...")
            
            for buggy in buggy_contracts[:min(500, len(buggy_contracts))]:
                if isinstance(buggy, dict) and 'code' in buggy and buggy['code']:
                    clean_code = preprocessor._generate_clean_version(buggy)
                    if clean_code:
                        clean_contracts.append({
                            'code': clean_code,
                            'original_buggy': buggy,
                            'vulnerability_type': buggy.get('vulnerability_type', 'unknown'),
                            'source_file': buggy.get('source_file', 'unknown'),
                            'synthetic': True
                        })
            
            logger.info(f"Generated {len(clean_contracts)} clean contracts")
        
        # Create dataset
        if clean_contracts or buggy_contracts:
            dataset = VulnerabilityDataset(clean_contracts, buggy_contracts)
            patterns = PatternExtractor()
            patterns.extract_patterns(dataset)
            
            return dataset, patterns, processed_data
        
        return None, None, processed_data
    
    def _calculate_preprocessing_stats(self, dataset, patterns, processed_data, start_time) -> Dict:
        """Calculate preprocessing statistics"""
        stats = {
            'total_contracts': len(dataset) if dataset else 0,
            'clean_contracts': len(processed_data.get('clean', [])),
            'buggy_contracts': len(processed_data.get('buggy', [])),
            'vulnerability_distribution': {},
            'preprocessing_time': time.time() - start_time,
            'patterns_extracted': {},
            'folder_results': processed_data.get('folder_results', {})
        }
        
        # Calculate vulnerability distribution from dataset
        if dataset and len(dataset) > 0:
            vuln_counts = defaultdict(int)
            sample_size = min(1000, len(dataset))
            
            for i in range(sample_size):
                try:
                    item = dataset[i]
                    vuln_type = item.get('vulnerability_type', 'unknown')
                    vuln_counts[vuln_type] += 1
                except Exception as e:
                    logger.debug(f"Error sampling dataset item {i}: {str(e)}")
            
            stats['vulnerability_distribution'] = dict(vuln_counts)
            stats['sample_size'] = sample_size
        
        # Get patterns statistics
        if hasattr(patterns, 'patterns'):
            stats['patterns_extracted'] = {
                vuln_type: len(pattern_list)
                for vuln_type, pattern_list in patterns.patterns.items()
            }
        
        return stats
    
    def _save_preprocessing_stats(self, stats: Dict):
        """Save preprocessing statistics"""
        results_dir = Path(self.config['paths'].get('results_dir', './cabis_project/results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        stats_path = results_dir / 'preprocessing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Stats saved to: {stats_path}")
    
    def _log_preprocessing_summary(self, stats: Dict):
        """Log preprocessing summary"""
        logger.info(f"✅ Preprocessing completed in {stats['preprocessing_time']:.2f} seconds")
        logger.info(f"   Total contract pairs: {stats['total_contracts']}")
        logger.info(f"   Clean contracts: {stats['clean_contracts']}")
        logger.info(f"   Buggy contracts: {stats['buggy_contracts']}")
        
        if stats['vulnerability_distribution']:
            logger.info("\n   Vulnerability distribution (sampled):")
            total_sampled = sum(stats['vulnerability_distribution'].values())
            for vuln_type, count in sorted(stats['vulnerability_distribution'].items()):
                percentage = (count / total_sampled * 100) if total_sampled > 0 else 0
                logger.info(f"      {vuln_type}: {count} ({percentage:.1f}%)")
    

    def train_model(self, dataset, epochs: Optional[int] = None):
        """Train CABIS model on full dataset with improved error handling"""
        if epochs is None:
            epochs = self.config['training']['epochs']
        
        logger.info(f"\n🏋️ Training CABIS model for {epochs} epochs...")
        
        # Check dataset size
        if len(dataset) == 0:
            logger.error("Cannot train on empty dataset!")
            return False
        
        # Check if model already exists
        model_dir = Path(self.config['paths'].get('model_dir', './cabis_project/models'))
        model_path = model_dir / 'cabis_best.pt'
        
        if model_path.exists():
            response = input("Model already exists. Train new model? (y/n): ")
            if response.lower() != 'y':
                logger.info("Using existing model")
                return True
        
        # Split dataset
        dataset_config = self.config.get('dataset', {})
        train_split = dataset_config.get('train_split', 0.8)
        val_split = dataset_config.get('val_split', 0.1)
        
        train_size = int(train_split * len(dataset))
        val_size = int(val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        # Ensure valid splits
        if train_size == 0:
            train_size = max(1, int(0.8 * len(dataset)))
            val_size = max(1, int(0.1 * len(dataset)))
            test_size = len(dataset) - train_size - val_size
        
        # Check dataset distribution before training
        logger.info("Checking dataset distribution before training...")
        vuln_counts = defaultdict(int)
        sample_size = min(1000, len(dataset))
        
        for i in range(sample_size):
            try:
                item = dataset[i]
                vuln_type = item.get('vulnerability_type', 'unknown')
                vuln_counts[vuln_type] += 1
            except Exception as e:
                logger.debug(f"Error checking item {i}: {str(e)}")
        
        logger.info("Training dataset distribution:")
        for vuln_type in self.config['vulnerability_types']:
            count = vuln_counts.get(vuln_type, 0)
            percentage = (count / sample_size * 100) if sample_size > 0 else 0
            logger.info(f"  {vuln_type}: {count} ({percentage:.1f}%)")
            if count == 0:
                logger.warning(f"  WARNING: No samples found for {vuln_type}!")
        
        try:
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
            logger.info(f"   Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
            
            # Initialize trainer with the fixed train_cabis.py
            trainer = EnhancedCABISTrainer(self.config_path)
            
            # Start training with error handling
            start_time = time.time()
            
            try:
                trainer.train(train_dataset, val_dataset, num_epochs=epochs)
                training_time = time.time() - start_time
                
                logger.info(f"✅ Training completed in {training_time/60:.2f} minutes")
                
                self.results['training'] = {
                    'epochs': epochs,
                    'training_time': training_time,
                    'train_size': train_size,
                    'val_size': val_size,
                    'test_size': test_size,
                    'best_metrics': getattr(trainer, 'best_metrics', {})
                }
                
                return True
                
            except Exception as training_error:
                logger.error(f"❌ Error during training: {str(training_error)}")
                logger.error(f"Training failed after {(time.time() - start_time)/60:.2f} minutes")
                
                # Check if we have any saved checkpoints
                checkpoint_dir = Path(self.config['paths'].get('checkpoint_dir', './cabis_project/checkpoints'))
                checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
                
                if checkpoints:
                    logger.info(f"Found {len(checkpoints)} checkpoint(s)")
                    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                    logger.info(f"Latest checkpoint: {latest_checkpoint}")
                    
                    # Try to save the latest checkpoint as the best model
                    try:
                        import shutil
                        shutil.copy(latest_checkpoint, model_path)
                        logger.info(f"Saved latest checkpoint as best model: {model_path}")
                        
                        # Extract epoch number from checkpoint
                        epoch_num = int(latest_checkpoint.stem.split('_')[-1])
                        
                        self.results['training'] = {
                            'epochs': epoch_num,
                            'training_time': time.time() - start_time,
                            'train_size': train_size,
                            'val_size': val_size,
                            'test_size': test_size,
                            'status': 'partial_completion',
                            'error': str(training_error)
                        }
                        
                        return True  # Partial success
                        
                    except Exception as cp_error:
                        logger.error(f"Failed to save checkpoint as best model: {str(cp_error)}")
                
                # Save error information
                self.results['training'] = {
                    'epochs_attempted': epochs,
                    'training_time': time.time() - start_time,
                    'train_size': train_size,
                    'val_size': val_size,
                    'test_size': test_size,
                    'status': 'failed',
                    'error': str(training_error)
                }
                
                raise  # Re-raise the exception
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Check if we should continue without training
            if model_path.exists():
                logger.info("Found existing model, continuing with evaluation...")
                return True
            else:
                logger.error("No model available for evaluation")
                return False
    
   
    def evaluate_model(self, dataset, num_samples: Optional[int] = None):
        """Comprehensive evaluation of CABIS model"""
        if num_samples is None:
            num_samples = self.config['evaluation'].get('num_samples', 100)
        
        # Ensure we don't try to evaluate more samples than available
        num_samples = min(num_samples, len(dataset))
        
        logger.info(f"\n📈 Evaluating CABIS model on {num_samples} samples...")
        
        try:
            # Initialize CABIS with trained model
            cabis = ImprovedCABIS(self.config)
            
            # Initialize evaluator
            detection_tools = self.config['evaluation'].get('detection_tools', [])
            evaluator = EnhancedCABISEvaluator(cabis, detection_tools, self.config)
            
            # Run evaluation
            start_time = time.time()
            
            # Create test subset
            test_indices = list(range(min(num_samples, len(dataset))))
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            
            evaluation_results = evaluator.evaluate_comprehensive(
                test_dataset,
                num_samples=len(test_dataset)
            )
            
            eval_time = time.time() - start_time
            
            # Add timing info
            evaluation_results['evaluation_time'] = eval_time
            evaluation_results['timestamp'] = datetime.now().isoformat()
            
            # Save results
            self._save_evaluation_results(evaluation_results)
            
            logger.info(f"✅ Evaluation completed in {eval_time/60:.2f} minutes")
            
            self.results['evaluation'] = evaluation_results
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results"""
        results_dir = Path(self.config['paths'].get('results_dir', './cabis_project/results'))
        results_path = results_dir / 'evaluation_results.json'
        
        # Convert numpy types for JSON
        json_results = self._convert_for_json(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"📊 Evaluation results saved to: {results_path}")
    
    def demonstrate_injection(self, processed_data: Optional[Dict] = None):
        """Demonstrate vulnerability injection on sample contracts"""
        logger.info("\n🔧 Demonstrating vulnerability injection...")
        
        try:
            # Initialize CABIS
            cabis = ImprovedCABIS(self.config)
            
            # Create demo directory
            results_dir = Path(self.config['paths'].get('results_dir', './cabis_project/results'))
            demo_dir = results_dir / 'demo_injections'
            demo_dir.mkdir(exist_ok=True)
            
            # Sample contract
            sample_contract = self._get_sample_contract()
            
            # Save original
            with open(demo_dir / 'original.sol', 'w') as f:
                f.write(sample_contract)
            
            # Demonstrate each vulnerability type
            demo_results = {}
            successful_demos = 0
            
            for vuln_type in self.config['vulnerability_types']:
                logger.info(f"\n💉 Injecting {vuln_type}...")
                
                try:
                    result = cabis.inject_vulnerability(
                        sample_contract,
                        vuln_type,
                        ensure_exploitable=True,
                        diversity_level=0.7
                    )
                    
                    # Save result
                    output_file = demo_dir / f'demo_{vuln_type}.sol'
                    with open(output_file, 'w') as f:
                        f.write(result['modified'])
                    
                    # Save metadata
                    metadata_file = demo_dir / f'demo_{vuln_type}_metadata.json'
                    with open(metadata_file, 'w') as f:
                        json.dump({
                            'vulnerability_type': vuln_type,
                            'injection_points': result.get('injection_points', []),
                            'exploit': result.get('exploit'),
                            'compiles': result.get('compiles', False),
                            'metrics': result.get('metrics', {})
                        }, f, indent=2)
                    
                    logger.info(f"   ✅ Success - Saved to: {output_file}")
                    successful_demos += 1
                    
                    demo_results[vuln_type] = {
                        'success': True,
                        'file': str(output_file),
                        'exploit': result.get('exploit')
                    }
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed: {str(e)}")
                    demo_results[vuln_type] = {
                        'success': False,
                        'error': str(e)
                    }
            
            logger.info(f"\n✅ Demonstration complete: {successful_demos}/{len(self.config['vulnerability_types'])} successful")
            
            self.results['demonstration'] = demo_results
            
            # Create README
            self._create_demo_readme(demo_dir, demo_results)
            
        except Exception as e:
            logger.error(f"❌ Demonstration failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _get_sample_contract(self) -> str:
        """Get sample contract for demonstration"""
        return """pragma solidity ^0.8.0;

contract SimpleWallet {
    mapping(address => uint256) public balances;
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    function deposit() public payable {
        require(msg.value > 0, "Deposit amount must be positive");
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }
    
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
    
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }
}"""
    
    def _create_demo_readme(self, demo_dir: Path, demo_results: Dict):
        """Create README for demonstration results"""
        readme_content = [
            "# CABIS Vulnerability Injection Demonstrations",
            "",
            "This directory contains examples of vulnerabilities injected by CABIS.",
            "",
            "## Files",
            "",
            "- `original.sol` - The original clean contract",
        ]
        
        for vuln_type in self.config['vulnerability_types']:
            if demo_results.get(vuln_type, {}).get('success'):
                readme_content.extend([
                    f"- `demo_{vuln_type}.sol` - Contract with {vuln_type} vulnerability",
                    f"- `demo_{vuln_type}_metadata.json` - Injection metadata"
                ])
        
        readme_content.extend([
            "",
            "## Vulnerability Types Demonstrated",
            ""
        ])
        
        for vuln_type in self.config['vulnerability_types']:
            result = demo_results.get(vuln_type, {})
            if result.get('success'):
                readme_content.extend([
                    f"### {vuln_type}",
                    "",
                    f"Status: Successfully injected",  # Removed Unicode character
                ])
                if result.get('exploit'):
                    readme_content.extend([
                        f"Exploit: {result['exploit']}",
                    ])
                readme_content.append("")
        
        # Add failed injections
        failed = [vt for vt in self.config['vulnerability_types'] 
                 if not demo_results.get(vt, {}).get('success')]
        if failed:
            readme_content.extend([
                "## Failed Injections",
                ""
            ])
            for vuln_type in failed:
                error = demo_results.get(vuln_type, {}).get('error', 'Unknown error')
                readme_content.extend([
                    f"- **{vuln_type}**: {error}"
                ])
        
        # Fix: Use UTF-8 encoding when writing
        with open(demo_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(readme_content))
        

    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, defaultdict):
            return dict(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def generate_final_report(self):
        """Generate final comprehensive report"""
        logger.info("\n📊 Generating final report...")
        
        # Calculate total runtime
        total_runtime = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            'pipeline_execution': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_runtime_seconds': total_runtime,
                'total_runtime_minutes': total_runtime / 60
            },
            'configuration': {
                'config_file': self.config_path,
                'vulnerability_types': self.config['vulnerability_types'],
                'dataset_folders': list(self.config.get('solidifi_folders', {}).keys())
            },
            'results': self.results
        }
        
        # Save comprehensive report
        results_dir = Path(self.config['paths'].get('results_dir', './cabis_project/results'))
        report_path = results_dir / 'pipeline_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(self._convert_for_json(report), f, indent=2)
        
        logger.info(f"📄 Final report saved to: {report_path}")
        
        # Generate markdown summary
        self._generate_markdown_summary(report)
    
    def _generate_markdown_summary(self, report: Dict):
        """Generate markdown summary of pipeline execution"""
        md_lines = [
            "# CABIS Pipeline Execution Summary",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Runtime**: {report['pipeline_execution']['total_runtime_minutes']:.1f} minutes",
            "",
            "## Configuration",
            f"- Config file: `{report['configuration']['config_file']}`",
            f"- Vulnerability types: {len(report['configuration']['vulnerability_types'])}",
            ""
        ]
        
        # Dataset statistics
        if 'dataset_stats' in self.results:
            stats = self.results['dataset_stats']
            md_lines.extend([
                "## Dataset Statistics",
                f"- Folders found: {stats['folders']}",
                f"- Total .sol files: {stats['total_sol_files']}",
                f"- Total .csv files: {stats['total_csv_files']}",
                ""
            ])
        
        # Preprocessing results
        if 'preprocessing' in self.results:
            prep = self.results['preprocessing']
            md_lines.extend([
                "## Preprocessing Results",
                f"- Total contract pairs: {prep.get('total_contracts', 0)}",
                f"- Processing time: {prep.get('preprocessing_time', 0):.2f} seconds",
                ""
            ])
            
            if 'vulnerability_distribution' in prep:
                md_lines.append("### Vulnerability Distribution")
                for vuln_type, count in sorted(prep['vulnerability_distribution'].items()):
                    md_lines.append(f"- {vuln_type}: {count}")
                md_lines.append("")
        
        # Training results
        if 'training' in self.results:
            train = self.results['training']
            md_lines.extend([
                "## Training Results",
                f"- Epochs: {train.get('epochs', 'N/A')}",
                f"- Training time: {train.get('training_time', 0)/60:.1f} minutes",
                f"- Train/Val/Test split: {train.get('train_size', 0)}/{train.get('val_size', 0)}/{train.get('test_size', 0)}",
                ""
            ])
        
        # Evaluation results
        if 'evaluation' in self.results:
            eval_res = self.results['evaluation']
            md_lines.extend([
                "## Evaluation Results",
                f"- Evaluation time: {eval_res.get('evaluation_time', 0)/60:.1f} minutes",
            ])
            
            # Add key metrics if available
            for metric in ['diversity_score', 'realism_score', 'preservation_score', 
                          'exploitability_rate', 'compilation_success_rate']:
                if metric in eval_res:
                    md_lines.append(f"- {metric.replace('_', ' ').title()}: {eval_res[metric]:.3f}")
            md_lines.append("")
        
        # Demonstration results
        if 'demonstration' in self.results:
            demo = self.results['demonstration']
            successful = sum(1 for r in demo.values() if r.get('success', False))
            md_lines.extend([
                "## Demonstration Results",
                f"- Successful injections: {successful}/{len(demo)}",
                ""
            ])
        
        # Save markdown summary
        results_dir = Path(self.config['paths'].get('results_dir', './cabis_project/results'))
        md_path = results_dir / 'pipeline_summary.md'
        
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))
        
        logger.info(f"📝 Summary saved to: {md_path}")
    
    def run(self, args):
        """Run the complete pipeline"""
        logger.info("="*80)
        logger.info("CABIS FULL PIPELINE - SolidiFI Complete Dataset")
        logger.info("="*80)
        logger.info(f"Configuration: {self.config_path}")
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Setup
            self.setup_directories()
            
            # Step 2: Check data
            if not self.check_data_availability():
                logger.error("\n❌ Pipeline aborted due to missing data")
                logger.error("Please download the SolidiFI dataset from:")
                logger.error("https://github.com/smartbugs/solidifi-benchmark")
                return False
            
            # Step 3: Preprocess data
            dataset, patterns, processed_data = self.preprocess_data()
            
            # Step 4: Train model (unless skipped)
            if not args.skip_training:
                if not self.train_model(dataset, args.epochs):
                    logger.warning("Training failed, continuing with existing model if available")
            else:
                logger.info("⏭️  Skipping training phase")
            
            # Step 5: Evaluate model (unless skipped)
            if not args.skip_evaluation:
                self.evaluate_model(dataset, args.eval_samples)
            else:
                logger.info("⏭️  Skipping evaluation phase")
            
            # Step 6: Demonstrate injection
            if not args.skip_demo:
                self.demonstrate_injection(processed_data)
            else:
                logger.info("⏭️  Skipping demonstration phase")
            
            # Step 7: Generate final report
            self.generate_final_report()
            
            # Success summary
            total_time = (datetime.now() - self.start_time).total_seconds() / 60
            
            logger.info("\n" + "="*80)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"Total runtime: {total_time:.1f} minutes")
            
            # Show key outputs
            results_dir = self.config['paths'].get('results_dir', './cabis_project/results')
            model_dir = self.config['paths'].get('model_dir', './cabis_project/models')
            
            logger.info(f"\n📁 Results saved to: {results_dir}")
            logger.info("\n🎯 Key outputs:")
            
            outputs = [
                (f"{model_dir}/cabis_best.pt", "Trained model"),
                (f"{results_dir}/evaluation_results.json", "Evaluation report"),
                (f"{results_dir}/demo_injections/", "Demo injections"),
                (f"{results_dir}/pipeline_report.json", "Pipeline report"),
                (f"{results_dir}/pipeline_summary.md", "Summary report")
            ]
            
            for path, desc in outputs:
                if Path(path).exists():
                    logger.info(f"  ✅ {desc}: {path}")
                else:
                    logger.info(f"  ⚠️  {desc}: Not created")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed with error: {str(e)}")
            logger.error("See logs for details")
            
            # Save error report
            self._save_error_report(e)
            
            return False
    
    def _save_error_report(self, error: Exception):
        """Save error report"""
        error_report = {
            'error': str(error),
            'error_type': type(error).__name__,
            'timestamp': datetime.now().isoformat(),
            'partial_results': self._convert_for_json(self.results),
            'traceback': []
        }
        
        # Add traceback
        import traceback
        error_report['traceback'] = traceback.format_exc().split('\n')
        
        results_dir = Path(self.config['paths'].get('results_dir', './cabis_project/results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        error_path = results_dir / 'error_report.json'
        
        try:
            with open(error_path, 'w') as f:
                json.dump(error_report, f, indent=2)
            logger.info(f"Error report saved to: {error_path}")
        except Exception as save_error:
            logger.error(f"Failed to save error report: {save_error}")


def main():
    parser = argparse.ArgumentParser(
        description='Run CABIS full pipeline on complete SolidiFI dataset'
    )
    parser.add_argument('--config', type=str, 
                       default='cabis_project/configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--eval-samples', type=int, default=None,
                       help='Number of samples for evaluation (overrides config)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training phase')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation phase')
    parser.add_argument('--skip-demo', action='store_true',
                       help='Skip demonstration phase')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with minimal samples')
    
    args = parser.parse_args()
    
    # Quick mode settings
    if args.quick:
        logger.info("🚀 Quick mode enabled - using reduced samples")
        if args.epochs is None:
            args.epochs = 2
        if args.eval_samples is None:
            args.eval_samples = 50
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Please create the config file or specify a valid path")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = CABISPipeline(args.config)
    
    # Run pipeline
    success = pipeline.run(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()