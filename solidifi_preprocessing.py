# -*- coding: utf-8 -*-
"""
@author: Willie
"""

import pandas as pd
import re
import json
from typing import List, Dict, Tuple, Optional, Set
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ENHANCED SolidiFI folder mapping with all possible variations
SOLIDIFI_FOLDERS = {
    'Overflow-Underflow': 'integer_overflow',
    'Re-entrancy': 'reentrancy',
    'Timestamp-Dependency': 'timestamp_dependence',
    'TOD': 'transaction_order_dependence',
    'tx.origin': 'tx_origin',
    'Unchecked-Send': 'unchecked_call',
    'Unhandled-Exceptions': 'unhandled_exceptions'
}

# COMPREHENSIVE vulnerability type mapping to handle ALL variations
VULNERABILITY_TYPE_MAPPING = {
    # Integer overflow/underflow variations
    'integer_overflow': 'integer_overflow',
    'integer overflow': 'integer_overflow',
    'Integer Overflow': 'integer_overflow',
    'overflow': 'integer_overflow',
    'underflow': 'integer_overflow',
    'Overflow-Underflow': 'integer_overflow',
    'overflow-underflow': 'integer_overflow',
    'arithmetic': 'integer_overflow',
    'integer_underflow': 'integer_overflow',
    
    # Reentrancy variations
    'reentrancy': 'reentrancy',
    'Reentrancy': 'reentrancy',
    're-entrancy': 'reentrancy',
    'Re-entrancy': 'reentrancy',
    'reentrancy-eth': 'reentrancy',
    'reentrancy-benign': 'reentrancy',
    'reentrancy_eth': 'reentrancy',
    'reentrancy_benign': 'reentrancy',
    
    # Timestamp dependence variations
    'timestamp_dependence': 'timestamp_dependence',
    'timestamp-dependence': 'timestamp_dependence',
    'Timestamp-Dependency': 'timestamp_dependence',
    'timestamp-dependency': 'timestamp_dependence',
    'timestamp_dependency': 'timestamp_dependence',
    'time_manipulation': 'timestamp_dependence',
    'time-manipulation': 'timestamp_dependence',
    'timestamp': 'timestamp_dependence',
    
    # Unchecked call variations
    'unchecked_call': 'unchecked_call',
    'unchecked-call': 'unchecked_call',
    'unchecked_send': 'unchecked_call',
    'unchecked-send': 'unchecked_call',
    'Unchecked-Send': 'unchecked_call',
    'unchecked_low_level_calls': 'unchecked_call',
    'unchecked-low-level-calls': 'unchecked_call',
    'unchecked_return_value': 'unchecked_call',
    
    # TX origin variations
    'tx_origin': 'tx_origin',
    'tx-origin': 'tx_origin',
    'tx.origin': 'tx_origin',
    'txorigin': 'tx_origin',
    'tx origin': 'tx_origin',
    'authorization_through_tx_origin': 'tx_origin',
    'authorization-through-tx-origin': 'tx_origin',
    'tx_origin_authentication': 'tx_origin',
    
    # TOD variations
    'transaction_order_dependence': 'transaction_order_dependence',
    'transaction-order-dependence': 'transaction_order_dependence',
    'transaction_order_dependency': 'transaction_order_dependence',
    'tod': 'transaction_order_dependence',
    'TOD': 'transaction_order_dependence',
    'front_running': 'transaction_order_dependence',
    'front-running': 'transaction_order_dependence',
    'race_condition': 'transaction_order_dependence',
    
    # Unhandled exceptions variations
    'unhandled_exceptions': 'unhandled_exceptions',
    'unhandled-exceptions': 'unhandled_exceptions',
    'Unhandled-Exceptions': 'unhandled_exceptions',
    'unhandled_exception': 'unhandled_exceptions',
    'unhandled-exception': 'unhandled_exceptions',
    'exception_disorder': 'unhandled_exceptions',
    'exception-disorder': 'unhandled_exceptions',
    'exceptions': 'unhandled_exceptions'
}

# Extended vulnerability patterns with more comprehensive detection
VULNERABILITY_PATTERNS = {
    'integer_overflow': ['bug_intou', 'transfer_intou', 'increaseLockTime_intou', '_intou', 
                        'overflow', 'underflow', '+ 1', '- 1', 'uint8', 'uint256'],
    'reentrancy': ['bug_re_ent', '_re_ent', 'withdraw', 'claimReward', 
                   'msg.sender.call', 'call{value:', '.call.value('],
    'timestamp_dependence': ['bug_tmstmp', 'play_tmstmp', 'bugv_tmstmp', '_tmstmp',
                            'block.timestamp', 'now', 'timestamp =='],
    'transaction_order_dependence': ['play_TOD', 'getReward_TOD', 'setReward_TOD', 
                                    'claimReward_TOD', 'winner_TOD', '_TOD', 'reward_TOD'],
    'tx_origin': ['bug_txorigin', 'transferTo_txorigin', 'withdrawAll_txorigin', 
                  'sendto_txorigin', '_txorigin', 'tx.origin', 'require(tx.origin'],
    'unchecked_call': ['bug_unchk_send', 'withdrawLeftOver_unchk', 'sendToWinner_unchk', 
                       'callnotchecked_unchk', '_unchk', '.send(', '.call(', 'bug_unchk'],
    'unhandled_exceptions': ['_unchk', 'callnotchecked', 'unhandled', 'exception',
                            'call(', 'no require', 'unchecked external']
}

class SolidiFIPreprocessor:
    """Enhanced preprocessor for full SolidiFI benchmark dataset (7 folders × 50 files)"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.vulnerability_patterns = defaultdict(list)
        self.clean_contracts = []
        self.buggy_contracts = []
        self.contracts_by_type = defaultdict(list)
        self.all_contracts = []
        self.stats = defaultdict(int)
        self.pattern_statistics = {}
        
        # Initialize pattern statistics for all 7 types
        for vuln_type in ['reentrancy', 'integer_overflow', 'timestamp_dependence', 
                         'unchecked_call', 'tx_origin', 'transaction_order_dependence', 
                         'unhandled_exceptions']:
            self.pattern_statistics[vuln_type] = {
                'functions': defaultdict(int),
                'subtypes': defaultdict(int),
                'total': 0,
                'severities': defaultdict(int)
            }
        
        # Track unmapped types for debugging
        self.unmapped_types = set()
        
    def normalize_vulnerability_type(self, vuln_type: str) -> str:
        """Normalize vulnerability type to standard naming"""
        if not vuln_type:
            return 'unknown'
        
        # Clean the input
        vuln_type = vuln_type.strip().lower().replace('_', ' ').replace('-', ' ')
        
        # Try exact match first
        if vuln_type in VULNERABILITY_TYPE_MAPPING:
            return VULNERABILITY_TYPE_MAPPING[vuln_type]
        
        # Try with underscores
        vuln_type_underscore = vuln_type.replace(' ', '_')
        if vuln_type_underscore in VULNERABILITY_TYPE_MAPPING:
            return VULNERABILITY_TYPE_MAPPING[vuln_type_underscore]
        
        # Try with hyphens
        vuln_type_hyphen = vuln_type.replace(' ', '-')
        if vuln_type_hyphen in VULNERABILITY_TYPE_MAPPING:
            return VULNERABILITY_TYPE_MAPPING[vuln_type_hyphen]
        
        # Try to infer from folder name mapping
        for folder, mapped_type in SOLIDIFI_FOLDERS.items():
            if folder.lower() in vuln_type or vuln_type in folder.lower():
                return mapped_type
        
        # Pattern-based inference
        if any(pattern in vuln_type for pattern in ['overflow', 'underflow', 'intou', 'arithmetic']):
            return 'integer_overflow'
        elif any(pattern in vuln_type for pattern in ['reentran', 're-entr', 're_ent']):
            return 'reentrancy'
        elif any(pattern in vuln_type for pattern in ['timestamp', 'tmstmp', 'time']):
            return 'timestamp_dependence'
        elif any(pattern in vuln_type for pattern in ['unchecked', 'unchk', 'send', 'call']):
            return 'unchecked_call'
        elif any(pattern in vuln_type for pattern in ['tx', 'origin', 'txorigin']):
            return 'tx_origin'
        elif any(pattern in vuln_type for pattern in ['tod', 'order', 'race', 'front']):
            return 'transaction_order_dependence'
        elif any(pattern in vuln_type for pattern in ['exception', 'unhandled', 'handle']):
            return 'unhandled_exceptions'
        
        # Track unmapped types
        self.unmapped_types.add(vuln_type)
        logger.warning(f"Could not map vulnerability type: '{vuln_type}'")
        
        return 'unknown'
    
    def _process_vulnerability_folder(self, folder_path: Path, vuln_type: str, folder_name: str) -> Dict:
        """Process all contracts in a vulnerability type folder with better error handling"""
        # Get all .sol and .csv files with multiple naming patterns
        sol_patterns = ['buggy_*.sol', 'buggy*.sol', '*_buggy.sol', '*.sol']
        csv_patterns = ['BugLog_*.csv', 'BugLog*.csv', 'buglog*.csv', '*.csv']
        
        sol_files = []
        csv_files = []
        
        for pattern in sol_patterns:
            sol_files.extend(folder_path.glob(pattern))
        
        for pattern in csv_patterns:
            csv_files.extend(folder_path.glob(pattern))
        
        # Remove duplicates and sort
        sol_files = sorted(list(set(sol_files)))
        csv_files = sorted(list(set(csv_files)))
        
        logger.info(f"Found {len(sol_files)} .sol files and {len(csv_files)} .csv files")
        
        if len(sol_files) == 0:
            logger.warning(f"No .sol files found in {folder_path}")
            return {'processed': 0, 'errors': 0, 'patterns': 0, 'files_found': 0}
        
        processed_count = 0
        error_count = 0
        patterns_found = 0
        successful_contracts = 0
        
        # Process all .sol files, with or without CSV
        for sol_file in sol_files[:50]:  # Limit to 50 per folder as specified
            try:
                # Find matching CSV if exists
                file_num_match = re.search(r'(\d+)', sol_file.stem)
                csv_file = None
                
                if file_num_match:
                    file_num = file_num_match.group(1)
                    for cf in csv_files:
                        if file_num in cf.stem:
                            csv_file = cf
                            break
                
                # Process with or without CSV
                if csv_file:
                    contracts = self._process_contract_pair(sol_file, csv_file, vuln_type, folder_name)
                else:
                    contracts = self._process_sol_only(sol_file, vuln_type, folder_name)
                
                if contracts:
                    successful_contracts += len(contracts)
                    
                    for contract in contracts:
                        # Normalize vulnerability type
                        contract['vulnerability_type'] = vuln_type
                        
                        # Store contracts
                        self.all_contracts.append(contract)
                        self.contracts_by_type[vuln_type].append(contract)
                        self.buggy_contracts.append(contract)
                        
                        # Generate clean version
                        clean_version = self._generate_clean_version(contract)
                        if clean_version:
                            self.clean_contracts.append({
                                'code': clean_version,
                                'original_buggy': contract,
                                'vulnerability_type': vuln_type,
                                'source_file': sol_file.name,
                                'folder': folder_name
                            })
                        
                        patterns_found += len(contract.get('vulnerability_snippets', []))
                    
                    processed_count += 1
                    self.stats[vuln_type] += len(contracts)
                    
            except Exception as e:
                logger.error(f"Error processing {sol_file.name}: {str(e)}")
                error_count += 1
                continue
        
        logger.info(f"Successfully processed {successful_contracts} contracts from {processed_count} files")
        logger.info(f"Found {patterns_found} vulnerability patterns")
        
        return {
            'processed': processed_count,
            'errors': error_count,
            'patterns': patterns_found,
            'files_found': len(sol_files),
            'successful_contracts': successful_contracts
        }
        
    def process_dataset(self) -> Dict:
        """Process all contracts and bug logs in the dataset"""
        # Check dataset structure
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        expected_folders = set(SOLIDIFI_FOLDERS.keys())
        found_folders = set(f.name for f in folders)
        
        logger.info(f"Expected folders: {expected_folders}")
        logger.info(f"Found folders: {found_folders}")
        
        if not expected_folders.issubset(found_folders):
            missing = expected_folders - found_folders
            logger.warning(f"Missing folders: {missing}")
        
        logger.info(f"Processing full SolidiFI dataset from: {self.data_dir}")
        return self.process_full_dataset()
    
    def process_full_dataset(self) -> Dict:
        """Process the complete SolidiFI dataset with folder structure"""
        logger.info(f"Processing full SolidiFI dataset")
        
        total_processed = 0
        folder_results = {}
        
        # Process each vulnerability type folder
        for folder_name, vuln_type in SOLIDIFI_FOLDERS.items():
            folder_path = self.data_dir / folder_name
            
            if not folder_path.exists():
                logger.warning(f"Folder {folder_name} not found at {folder_path}, skipping...")
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {folder_name} -> {vuln_type}")
            logger.info(f"{'='*60}")
            
            folder_stats = self._process_vulnerability_folder(folder_path, vuln_type, folder_name)
            folder_results[folder_name] = folder_stats
            total_processed += folder_stats['processed']
            
        # Generate comprehensive statistics
        self._generate_statistics()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total contracts processed: {len(self.all_contracts)}")
        logger.info(f"Total buggy contracts: {len(self.buggy_contracts)}")
        logger.info(f"Total clean contracts generated: {len(self.clean_contracts)}")
        logger.info(f"Total vulnerability patterns extracted: {sum(len(p) for p in self.vulnerability_patterns.values())}")
        
        # Convert defaultdicts to regular dicts for JSON serialization
        pattern_stats_dict = {}
        for vuln_type, stats in self.pattern_statistics.items():
            pattern_stats_dict[vuln_type] = {
                'functions': dict(stats['functions']),
                'subtypes': dict(stats['subtypes']),
                'total': stats['total'],
                'severities': dict(stats['severities'])
            }
        
        return {
            'clean': self.clean_contracts,
            'buggy': self.buggy_contracts,
            'patterns': dict(self.vulnerability_patterns),
            'all_contracts': self.all_contracts,
            'contracts_by_type': dict(self.contracts_by_type),
            'stats': dict(self.stats),
            'pattern_statistics': pattern_stats_dict,
            'folder_results': folder_results
        }
    
    def _process_contract_pair(self, sol_file: Path, csv_file: Path, vuln_type: str, folder_name: str) -> List[Dict]:
        """Process a single contract-buglog pair"""
        # Read contract file
        try:
            with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {sol_file}: {str(e)}")
            return []
            
        # Split into individual contracts if multiple exist
        contracts = self._split_contracts(content)
        
        # Read bug log with better error handling
        bug_df = None
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    bug_df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except:
                    continue
                    
            if bug_df is None:
                logger.warning(f"Could not read CSV {csv_file} with any encoding")
                bug_df = pd.DataFrame()  # Empty dataframe
            else:
                # Normalize column names
                bug_df.columns = bug_df.columns.str.strip().str.lower()
                
                # Handle different column name variations
                column_mappings = {
                    'bug type': 'bug_type',
                    'bug_type': 'bug_type',
                    'type': 'bug_type',
                    'loc': 'line',
                    'location': 'line',
                    'line': 'line',
                    'len': 'length',
                    'length': 'length'
                }
                
                for old_col, new_col in column_mappings.items():
                    if old_col in bug_df.columns and new_col not in bug_df.columns:
                        bug_df[new_col] = bug_df[old_col]
                
                # Ensure numeric columns are properly typed
                for col in ['line', 'length']:
                    if col in bug_df.columns:
                        bug_df[col] = pd.to_numeric(bug_df[col], errors='coerce')
                
        except Exception as e:
            logger.error(f"Error reading CSV {csv_file}: {str(e)}")
            bug_df = pd.DataFrame()  # Empty dataframe
        
        # Process each contract with its bugs
        processed_contracts = []
        
        for idx, contract_code in enumerate(contracts):
            # Skip empty contracts
            if len(contract_code.strip()) < 50:  # Very small contracts
                continue
                
            # Parse contract
            parsed = self._parse_contract(contract_code)
            
            # Add metadata
            parsed['source_file'] = sol_file.name
            parsed['folder_name'] = folder_name
            parsed['vulnerability_type'] = vuln_type
            parsed['contract_index'] = idx
            
            try:
                file_num = int(re.search(r'(\d+)', sol_file.stem).group(1))
                parsed['file_number'] = file_num
            except:
                parsed['file_number'] = 0
            
            # Add bugs from CSV if available
            if not bug_df.empty:
                contract_bugs = self._extract_bugs_for_contract_safe(
                    bug_df, contract_code, idx, len(contracts)
                )
                parsed['bugs'] = contract_bugs
            else:
                parsed['bugs'] = []
            
            # Extract vulnerability patterns specific to this type
            patterns = self._extract_vulnerability_patterns(contract_code, vuln_type)
            parsed['vulnerability_snippets'] = patterns
            
            # Store patterns globally
            for pattern in patterns:
                pattern['source_file'] = sol_file.name
                pattern['contract_index'] = idx
                self.vulnerability_patterns[vuln_type].append(pattern)
                
                # Update pattern statistics safely
                if vuln_type in self.pattern_statistics:
                    self.pattern_statistics[vuln_type]['total'] += 1
                    
                    if 'function' in pattern and pattern['function']:
                        self.pattern_statistics[vuln_type]['functions'][pattern['function']] += 1
                    
                    if 'subtype' in pattern and pattern['subtype']:
                        self.pattern_statistics[vuln_type]['subtypes'][pattern['subtype']] += 1
                    
                    if 'severity' in pattern and pattern['severity']:
                        self.pattern_statistics[vuln_type]['severities'][pattern['severity']] += 1
            
            processed_contracts.append(parsed)
            
        return processed_contracts
    
    def _process_sol_only(self, sol_file: Path, vuln_type: str, folder_name: str) -> List[Dict]:
        """Process a .sol file without corresponding CSV"""
        try:
            with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {sol_file}: {str(e)}")
            return []
        
        contracts = self._split_contracts(content)
        processed_contracts = []
        
        for idx, contract_code in enumerate(contracts):
            if len(contract_code.strip()) < 50:
                continue
                
            parsed = self._parse_contract(contract_code)
            parsed['source_file'] = sol_file.name
            parsed['folder_name'] = folder_name
            parsed['vulnerability_type'] = vuln_type
            parsed['contract_index'] = idx
            parsed['bugs'] = []  # No CSV data
            
            # Extract patterns
            patterns = self._extract_vulnerability_patterns(contract_code, vuln_type)
            parsed['vulnerability_snippets'] = patterns
            
            processed_contracts.append(parsed)
            
        return processed_contracts
    
    def _extract_bugs_for_contract_safe(self, bug_df: pd.DataFrame, contract_code: str, 
                                       contract_idx: int, total_contracts: int) -> List[Dict]:
        """Extract bugs for a specific contract from the bug log with safe type handling"""
        bugs = []
        contract_lines = contract_code.split('\n')
        
        for idx, row in bug_df.iterrows():
            try:
                # Get line number with safe handling
                bug_line = None
                if 'line' in row and pd.notna(row['line']):
                    try:
                        bug_line = int(row['line'])
                    except (ValueError, TypeError):
                        continue
                
                if bug_line is None or bug_line <= 0:
                    continue
                
                # Get length with safe handling
                bug_length = 1
                if 'length' in row and pd.notna(row['length']):
                    try:
                        bug_length = int(row['length'])
                    except (ValueError, TypeError):
                        bug_length = 1
                
                # Get bug type
                bug_type = 'unknown'
                if 'bug_type' in row and pd.notna(row['bug_type']):
                    bug_type = str(row['bug_type']).strip()
                
                # Check if bug line is within this contract
                if 0 < bug_line <= len(contract_lines):
                    bug_info = {
                        'line': bug_line,
                        'length': bug_length,
                        'bug_type': bug_type,
                        'approach': str(row.get('approach', '')).strip() if 'approach' in row else '',
                        'code_snippet': self._get_bug_snippet(contract_lines, bug_line, bug_length)
                    }
                    bugs.append(bug_info)
                    
            except Exception as e:
                logger.debug(f"Error extracting bug from row {idx}: {str(e)}")
                continue
                
        return bugs
    
    def _get_bug_snippet(self, lines: List[str], bug_line: int, length: int) -> str:
        """Extract code snippet for a bug"""
        start = max(0, bug_line - 1)  # Convert to 0-based index
        end = min(len(lines), start + length)
        return '\n'.join(lines[start:end])
    
    def _extract_vulnerability_patterns(self, code: str, vuln_type: str) -> List[Dict]:
        """Extract vulnerability patterns based on type using comprehensive patterns"""
        patterns = []
        
        # Use pattern keywords from VULNERABILITY_PATTERNS
        if vuln_type in VULNERABILITY_PATTERNS:
            for keyword in VULNERABILITY_PATTERNS[vuln_type]:
                # Find all occurrences of the pattern
                pattern_regex = rf'\b{re.escape(keyword)}\d*\b'
                for match in re.finditer(pattern_regex, code):
                    # Get surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(code), match.end() + 100)
                    context = code[start:end]
                    
                    patterns.append({
                        'type': vuln_type,
                        'keyword': keyword,
                        'pattern': match.group(0),
                        'context': context,
                        'start': match.start(),
                        'end': match.end(),
                        'line': code[:match.start()].count('\n') + 1
                    })
        
        # Also use specific extraction methods
        specific_patterns = []
        try:
            if vuln_type == 'reentrancy':
                specific_patterns = self._extract_reentrancy_patterns(code)
            elif vuln_type == 'integer_overflow':
                specific_patterns = self._extract_overflow_patterns(code)
            elif vuln_type == 'timestamp_dependence':
                specific_patterns = self._extract_timestamp_patterns(code)
            elif vuln_type == 'unchecked_call':
                specific_patterns = self._extract_unchecked_patterns(code)
            elif vuln_type == 'tx_origin':
                specific_patterns = self._extract_tx_origin_patterns(code)
            elif vuln_type == 'transaction_order_dependence':
                specific_patterns = self._extract_tod_patterns(code)
            elif vuln_type == 'unhandled_exceptions':
                specific_patterns = self._extract_exception_patterns(code)
        except Exception as e:
            logger.debug(f"Error in specific pattern extraction for {vuln_type}: {str(e)}")
            
        patterns.extend(specific_patterns)
        
        # Remove duplicates based on start position
        unique_patterns = []
        seen_positions = set()
        for pattern in patterns:
            pos = (pattern.get('start', 0), pattern.get('end', 0))
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_patterns.append(pattern)
                
        return unique_patterns
    
    def _extract_reentrancy_patterns(self, code: str) -> List[Dict]:
        """Extract reentrancy patterns based on actual dataset samples"""
        patterns = []
        
        # Pattern 1: Functions with _re_ent suffix
        re_ent_functions = re.finditer(
            r'function\s+(\w*_re_ent\d*)\s*\([^)]*\)\s*(?:public|external|internal|private)?[^{]*{([^}]+)}', 
            code, re.DOTALL
        )
        for match in re_ent_functions:
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Look for external calls
            if any(pattern in func_body for pattern in ['.transfer(', '.send(', '.call{value:', '.call.value(']):
                patterns.append({
                    'type': 'reentrancy',
                    'pattern': match.group(0)[:500],
                    'function': func_name,
                    'start': match.start(),
                    'end': match.end(),
                    'line': code[:match.start()].count('\n') + 1,
                    'severity': 'high',
                    'subtype': 're_ent_function'
                })
        
        # Pattern 2: claimReward functions with reentrancy
        claim_pattern = r'function\s+(claimReward\w*)\s*\([^)]*\)[^{]*{([^}]+)}'
        claim_matches = re.finditer(claim_pattern, code, re.DOTALL)
        
        for match in claim_matches:
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Check for state changes after external calls
            if re.search(r'msg\.sender\.(send|transfer|call).*\n.*=\s*0', func_body, re.DOTALL):
                patterns.append({
                    'type': 'reentrancy',
                    'pattern': match.group(0)[:500],
                    'function': func_name,
                    'start': match.start(),
                    'end': match.end(),
                    'line': code[:match.start()].count('\n') + 1,
                    'severity': 'high',
                    'subtype': 'claim_reward'
                })
        
        return patterns
    
    def _extract_overflow_patterns(self, code: str) -> List[Dict]:
        """Extract overflow/underflow patterns"""
        patterns = []
        
        # Pattern 1: Functions with intou naming
        intou_pattern = r'function\s+((?:bug_)?intou\d*)\s*\([^)]*\)\s*(?:public|external)?[^{]*{([^}]+)}'
        intou_matches = re.finditer(intou_pattern, code, re.DOTALL)
        
        for match in intou_matches:
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Determine if overflow or underflow
            is_overflow = '// overflow' in func_body.lower() or '+' in func_body
            is_underflow = '// underflow' in func_body.lower() or '- _value >= 0' in func_body
            
            patterns.append({
                'type': 'integer_overflow',
                'subtype': 'overflow' if is_overflow else 'underflow',
                'pattern': match.group(0)[:500],
                'function': func_name,
                'start': match.start(),
                'end': match.end(),
                'line': code[:match.start()].count('\n') + 1,
                'severity': 'medium'
            })
        
        return patterns
    
    def _extract_timestamp_patterns(self, code: str) -> List[Dict]:
        """Extract timestamp dependency patterns"""
        patterns = []
        
        # Pattern 1: Functions with tmstmp naming
        tmstmp_pattern = r'function\s+((?:bug_|play_)?tmstmp\d*)\s*\([^)]*\)[^{]*{([^}]+)}'
        tmstmp_matches = re.finditer(tmstmp_pattern, code, re.DOTALL)
        
        for match in tmstmp_matches:
            func_name = match.group(1)
            func_body = match.group(2)
            
            if 'block.timestamp' in func_body or 'now' in func_body:
                patterns.append({
                    'type': 'timestamp_dependence',
                    'pattern': match.group(0)[:500],
                    'function': func_name,
                    'start': match.start(),
                    'end': match.end(),
                    'line': code[:match.start()].count('\n') + 1,
                    'severity': 'medium',
                    'subtype': 'timestamp_check'
                })
        
        return patterns
    
    def _extract_unchecked_patterns(self, code: str) -> List[Dict]:
        """Extract unchecked send/call patterns"""
        patterns = []
        
        # Pattern 1: Functions with unchk naming
        unchk_pattern = r'function\s+(\w*_unchk\d*)\s*\([^)]*\)[^{]*{([^}]+)}'
        unchk_matches = re.finditer(unchk_pattern, code, re.DOTALL)
        
        for match in unchk_matches:
            func_name = match.group(1)
            patterns.append({
                'type': 'unchecked_call',
                'subtype': 'unchecked_send',
                'pattern': match.group(0)[:500],
                'function': func_name,
                'start': match.start(),
                'end': match.end(),
                'line': code[:match.start()].count('\n') + 1,
                'severity': 'high'
            })
        
        return patterns
    
    def _extract_tx_origin_patterns(self, code: str) -> List[Dict]:
        """Extract tx.origin patterns"""
        patterns = []
        
        # Pattern 1: Functions with txorigin suffix
        txorigin_func = r'function\s+(\w*_txorigin\d*)\s*\([^)]*\)[^{]*{([^}]+)}'
        func_matches = re.finditer(txorigin_func, code, re.DOTALL)
        
        for match in func_matches:
            func_name = match.group(1)
            func_body = match.group(2)
            
            if 'tx.origin' in func_body:
                patterns.append({
                    'type': 'tx_origin',
                    'pattern': match.group(0)[:500],
                    'function': func_name,
                    'start': match.start(),
                    'end': match.end(),
                    'line': code[:match.start()].count('\n') + 1,
                    'severity': 'high',
                    'subtype': 'tx_origin_auth'
                })
        
        return patterns
    
    def _extract_tod_patterns(self, code: str) -> List[Dict]:
        """Extract Transaction Order Dependence patterns"""
        patterns = []
        
        # Pattern 1: Functions with TOD naming
        tod_pattern = r'function\s+(\w*_TOD\d*)\s*\([^)]*\)[^{]*{([^}]+)}'
        tod_matches = re.finditer(tod_pattern, code, re.DOTALL)
        
        for match in tod_matches:
            func_name = match.group(1)
            patterns.append({
                'type': 'transaction_order_dependence',
                'pattern': match.group(0)[:500],
                'function': func_name,
                'start': match.start(),
                'end': match.end(),
                'line': code[:match.start()].count('\n') + 1,
                'severity': 'high',
                'subtype': 'tod_race'
            })
        
        return patterns
    
    def _extract_exception_patterns(self, code: str) -> List[Dict]:
        """Extract unhandled exception patterns"""
        patterns = []
        
        # Pattern 1: callnotchecked functions
        callnotchecked = r'function\s+(callnotchecked_unchk\d*)[^{]*{([^}]+)}'
        call_matches = re.finditer(callnotchecked, code, re.DOTALL)
        
        for match in call_matches:
            func_name = match.group(1)
            patterns.append({
                'type': 'unhandled_exceptions',
                'subtype': 'unchecked_call',
                'pattern': match.group(0)[:500],
                'function': func_name,
                'start': match.start(),
                'end': match.end(),
                'line': code[:match.start()].count('\n') + 1,
                'severity': 'high'
            })
        
        return patterns
    
    def _split_contracts(self, content: str) -> List[str]:
        """Split a file containing multiple contracts into individual contracts"""
        if not content.strip():
            return []
            
        contracts = []
        
        # Handle pragma and imports
        lines = content.split('\n')
        header_lines = []
        contract_start = -1
        
        for i, line in enumerate(lines):
            if re.match(r'^\s*(contract|library|interface)\s+', line):
                contract_start = i
                break
            header_lines.append(line)
        
        header = '\n'.join(header_lines)
        
        if contract_start == -1:
            # No contract declaration found, return whole content
            return [content]
        
        # Extract contracts using bracket matching
        current_contract = []
        bracket_count = 0
        in_contract = False
        
        for i in range(contract_start, len(lines)):
            line = lines[i]
            
            if re.match(r'^\s*(contract|library|interface)\s+', line):
                if in_contract and bracket_count == 0:
                    contracts.append(header + '\n' + '\n'.join(current_contract))
                    current_contract = []
                in_contract = True
            
            if in_contract:
                current_contract.append(line)
                bracket_count += line.count('{') - line.count('}')
                
                if bracket_count == 0 and '{' in '\n'.join(current_contract):
                    contracts.append(header + '\n' + '\n'.join(current_contract))
                    current_contract = []
                    in_contract = False
        
        if current_contract:
            contracts.append(header + '\n' + '\n'.join(current_contract))
        
        return contracts if contracts else [content]
    
    def _parse_contract(self, contract_code: str) -> Dict:
        """Parse a single contract and extract metadata"""
        parsed = {
            'code': contract_code,
            'functions': [],
            'state_variables': [],
            'modifiers': [],
            'events': [],
            'bugs': [],
            'vulnerability_snippets': [],
            'metrics': {}
        }
        
        # Extract contract name
        name_match = re.search(r'(contract|library|interface)\s+(\w+)', contract_code)
        if name_match:
            parsed['type'] = name_match.group(1)
            parsed['name'] = name_match.group(2)
        else:
            parsed['type'] = 'unknown'
            parsed['name'] = 'Unknown'
        
        # Extract functions
        function_pattern = r'function\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(function_pattern, contract_code):
            func_info = {
                'name': match.group(1),
                'parameters': match.group(2),
                'line': contract_code[:match.start()].count('\n') + 1
            }
            parsed['functions'].append(func_info)
        
        # Calculate metrics
        parsed['metrics'] = {
            'lines': contract_code.count('\n') + 1,
            'functions': len(parsed['functions']),
            'complexity': self._calculate_complexity(contract_code)
        }
        
        return parsed
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity estimate"""
        complexity = 1
        
        # Add complexity for control structures
        control_patterns = [
            r'\bif\s*\(',
            r'\belse\b',
            r'\bfor\s*\(',
            r'\bwhile\s*\(',
            r'\brequire\s*\(',
            r'\bassert\s*\(',
        ]
        
        for pattern in control_patterns:
            complexity += len(re.findall(pattern, code))
        
        return complexity
    
    def _generate_clean_version(self, parsed_contract: Dict) -> str:
        """Generate a clean version of the contract by removing vulnerabilities"""
        if 'code' not in parsed_contract:
            return ""
            
        clean_code = parsed_contract['code']
        vuln_type = parsed_contract.get('vulnerability_type', '')
        
        # Remove common vulnerability patterns
        vuln_function_patterns = [
            r'function\s+\w*_re_ent\d*[^}]*}',
            r'function\s+\w*_intou\d*[^}]*}',
            r'function\s+\w*_tmstmp\d*[^}]*}',
            r'function\s+\w*_unchk\d*[^}]*}',
            r'function\s+\w*_txorigin\d*[^}]*}',
            r'function\s+\w*_TOD\d*[^}]*}',
            r'function\s+callnotchecked_unchk\d*[^}]*}'
        ]
        
        for pattern in vuln_function_patterns:
            clean_code = re.sub(pattern, '', clean_code, flags=re.DOTALL)
        
        # Remove vulnerability-related state variables
        vuln_var_patterns = [
            r'address\s+winner_\w+\d*;',
            r'uint\d*\s+\w*_tmstmp\d*;',
            r'mapping\s*\([^)]+\)\s*\w*_TOD\d*;'
        ]
        
        for pattern in vuln_var_patterns:
            clean_code = re.sub(pattern, '', clean_code)
        
        # Clean up extra blank lines
        clean_code = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_code)
        
        return clean_code.strip()
    
    def _generate_statistics(self):
        """Generate comprehensive statistics about the processed dataset"""
        logger.info("\nGenerating dataset statistics...")
        
        # Overall statistics
        total_contracts = len(self.all_contracts)
        total_patterns = sum(len(patterns) for patterns in self.vulnerability_patterns.values())
        
        logger.info(f"\nDataset Statistics:")
        logger.info(f"{'='*60}")
        logger.info(f"Total contracts: {total_contracts}")
        logger.info(f"Total vulnerability patterns: {total_patterns}")
        logger.info(f"Clean contracts generated: {len(self.clean_contracts)}")
        
        if self.contracts_by_type:
            logger.info(f"\nPer-Vulnerability Type Statistics:")
            for vuln_type, contracts in self.contracts_by_type.items():
                if contracts:
                    logger.info(f"\n{vuln_type}:")
                    logger.info(f"  Contracts: {len(contracts)}")
                    logger.info(f"  Patterns: {len(self.vulnerability_patterns.get(vuln_type, []))}")
                    
                    # Calculate averages safely
                    if contracts:
                        avg_functions = np.mean([c.get('metrics', {}).get('functions', 0) for c in contracts])
                        avg_lines = np.mean([c.get('metrics', {}).get('lines', 0) for c in contracts])
                        avg_complexity = np.mean([c.get('metrics', {}).get('complexity', 0) for c in contracts])
                        
                        logger.info(f"  Avg Functions: {avg_functions:.1f}")
                        logger.info(f"  Avg Lines: {avg_lines:.1f}")
                        logger.info(f"  Avg Complexity: {avg_complexity:.1f}")
        
        # Pattern statistics
        if self.pattern_statistics:
            logger.info(f"\nPattern Statistics:")
            for vuln_type, stats in self.pattern_statistics.items():
                if stats['total'] > 0:
                    logger.info(f"\n{vuln_type}:")
                    logger.info(f"  Total patterns: {stats['total']}")
                    logger.info(f"  Unique functions: {len(stats['functions'])}")
                    logger.info(f"  Unique subtypes: {len(stats['subtypes'])}")


class VulnerabilityDataset(Dataset):
    """PyTorch dataset for vulnerability injection training"""
    
    def __init__(self, clean_contracts: List[Dict], buggy_contracts: List[Dict]):
        self.clean_contracts = clean_contracts if clean_contracts else []
        self.buggy_contracts = buggy_contracts if buggy_contracts else []
        self.pairs = []
        
        # Create initial pairs
        self._create_pairs()
        
        # Ensure all 7 vulnerability types are represented
        self._ensure_all_vulnerability_types()
        
        # Validate pairs
        self.pairs = [p for p in self.pairs if self._is_valid_pair(p)]
        
        logger.info(f"Created dataset with {len(self.pairs)} clean-buggy pairs")
        
        # Log distribution
        self._log_distribution()
  
    def _create_synthetic_pair(self, vuln_type: str) -> Tuple[str, str]:
        """Create more diverse synthetic contract pairs"""
        import random
        
        # Base templates with variations
        templates = [
            # Template 1: Banking contract
            """pragma solidity ^0.8.0;

contract Bank{{COUNTER}} {
    mapping(address => uint256) public balances;
    mapping(address => bool) public authorized;
    address public owner;
    uint256 public totalSupply;
    
    constructor() {
        owner = msg.sender;
        authorized[msg.sender] = true;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    function deposit() public payable {
        require(msg.value > 0, "Invalid amount");
        balances[msg.sender] += msg.value;
        totalSupply += msg.value;
    }
    
    function getBalance(address account) public view returns (uint256) {
        return balances[account];
    }
}""",
            # Template 2: Token contract
            """pragma solidity ^0.8.0;

contract Token{{COUNTER}} {
    mapping(address => uint256) balances;
    mapping(address => mapping(address => uint256)) allowances;
    uint256 public totalSupply = 1000000;
    address public minter;
    
    constructor() {
        minter = msg.sender;
        balances[msg.sender] = totalSupply;
    }
    
    function transfer(address to, uint256 amount) public returns (bool) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        return true;
    }
    
    function approve(address spender, uint256 amount) public {
        allowances[msg.sender][spender] = amount;
    }
}""",
            # Template 3: Lottery contract
            """pragma solidity ^0.8.0;

contract Lottery{{COUNTER}} {
    address public manager;
    address[] public players;
    uint256 public prize;
    
    constructor() {
        manager = msg.sender;
    }
    
    function enter() public payable {
        require(msg.value > 0.01 ether, "Minimum entry fee");
        players.push(msg.sender);
        prize += msg.value;
    }
    
    function getPlayers() public view returns (uint256) {
        return players.length;
    }
}"""
        ]
        
        # Select random template
        template = random.choice(templates)
        counter = random.randint(1, 999)
        clean = template.replace('{{COUNTER}}', str(counter))
        
        # Create buggy version based on vulnerability type
        buggy_injections = {
            'reentrancy': clean[:-1] + """
    
    function withdraw_re_ent{{NUM}}() public {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "No balance");
        
        // Reentrancy vulnerability: external call before state update
        (bool success,) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] = 0;
    }
}""",
            'integer_overflow': clean[:-1] + """
    
    function bug_intou{{NUM}}() public {
        uint8 count = 255;
        count = count + 1; // Overflow: 255 + 1 = 0
        balances[msg.sender] = count;
    }
    
    function transfer_intou{{NUM}}(address to, uint256 value) public {
        require(balances[msg.sender] - value >= 0); // Always true for uint
        balances[msg.sender] -= value;
        balances[to] += value;
    }
}""",
            'timestamp_dependence': clean[:-1] + """
    
    uint256 public winner_tmstmp{{NUM}};
    function play_tmstmp{{NUM}}(uint256 _vtime) public {
        if (_vtime == block.timestamp) { // Timestamp dependence
            winner_tmstmp{{NUM}} = block.timestamp;
            payable(msg.sender).transfer(address(this).balance);
        }
    }
    
    function bug_tmstmp{{NUM}}() public view returns (bool) {
        return block.timestamp % 2 == 0; // Predictable randomness
    }
}""",
            'unchecked_call': clean[:-1] + """
    
    function bug_unchk_send{{NUM}}() public payable {
        msg.sender.send(1 ether); // Unchecked return value
    }
    
    function withdrawLeftOver_unchk{{NUM}}(address payable recipient) public {
        recipient.call{value: address(this).balance}(""); // Unchecked call
    }
}""",
            'tx_origin': clean[:-1] + """
    
    function bug_txorigin{{NUM}}() public {
        require(tx.origin == owner, "Not owner"); // tx.origin vulnerability
        owner = msg.sender;
    }
    
    function transferTo_txorigin{{NUM}}(address to, uint256 amount) public {
        require(tx.origin == owner); // Should use msg.sender
        balances[to] += amount;
    }
}""",
            'transaction_order_dependence': clean[:-1] + """
    
    uint256 public reward_TOD{{NUM}};
    address public winner_TOD{{NUM}};
    
    function setReward_TOD{{NUM}}() public payable {
        reward_TOD{{NUM}} = msg.value;
    }
    
    function claimReward_TOD{{NUM}}() public {
        // Transaction order dependence
        if (reward_TOD{{NUM}} > 0) {
            winner_TOD{{NUM}} = msg.sender;
            payable(msg.sender).transfer(reward_TOD{{NUM}});
            reward_TOD{{NUM}} = 0;
        }
    }
}""",
            'unhandled_exceptions': clean[:-1] + """
    
    function callnotchecked_unchk{{NUM}}(address callee) public {
        callee.call(abi.encodeWithSignature("nonexistentFunction()")); // No error handling
    }
    
    function unhandled_send{{NUM}}(address payable to) public {
        to.send(1 ether); // Return value not checked
        balances[to] = 0; // State change regardless of send result
    }
}"""
        }
        
        num = random.randint(1, 99)
        buggy = buggy_injections[vuln_type].replace('{{NUM}}', str(num))
        
        return clean, buggy
    
    def _log_distribution(self):
        """Log the final distribution of vulnerability types"""
        type_counts = defaultdict(int)
        for clean, buggy in self.pairs:
            vuln_type = buggy.get('vulnerability_type', 'unknown')
            type_counts[vuln_type] += 1
        
        logger.info("Final dataset distribution:")
        total = len(self.pairs)
        for vuln_type in ['reentrancy', 'integer_overflow', 'timestamp_dependence', 
                         'unchecked_call', 'tx_origin', 'transaction_order_dependence', 
                         'unhandled_exceptions']:
            count = type_counts[vuln_type]
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"  {vuln_type}: {count} ({percentage:.1f}%)")
        
        if type_counts['unknown'] > 0:
            logger.warning(f"  unknown: {type_counts['unknown']} ({type_counts['unknown']/total*100:.1f}%)")
    
    def _is_valid_pair(self, pair: Tuple) -> bool:
        """Check if a pair is valid"""
        if not pair or len(pair) != 2:
            return False
        clean, buggy = pair
        return (isinstance(clean, dict) and 'code' in clean and clean['code'] and
                isinstance(buggy, dict) and 'code' in buggy and buggy['code'])
    
    def _create_pairs(self) -> List[Tuple[Dict, Dict]]:
        """Create clean-buggy pairs for training"""
        pairs = []
        
        # First try: Match clean contracts with their original buggy versions
        for clean in self.clean_contracts:
            if isinstance(clean, dict) and 'original_buggy' in clean and clean['original_buggy']:
                buggy = clean['original_buggy']
                if self._is_valid_pair((clean, buggy)):
                    pairs.append((clean, buggy))
        
        if pairs:
            logger.info(f"Created {len(pairs)} pairs from original_buggy matching")
            return pairs
        
        # Second try: Match by vulnerability type
        logger.warning("No pairs from original_buggy, trying vulnerability type matching")
        vuln_type_map = defaultdict(list)
        
        for buggy in self.buggy_contracts:
            if isinstance(buggy, dict) and 'code' in buggy and buggy['code']:
                vuln_type = buggy.get('vulnerability_type', 'unknown')
                vuln_type_map[vuln_type].append(buggy)
        
        for clean in self.clean_contracts:
            if isinstance(clean, dict) and 'code' in clean and clean['code']:
                vuln_type = clean.get('vulnerability_type', 'unknown')
                if vuln_type in vuln_type_map and vuln_type_map[vuln_type]:
                    buggy = vuln_type_map[vuln_type].pop(0)
                    pairs.append((clean, buggy))
        
        return pairs
    
    def _create_synthetic_pairs(self) -> List[Tuple[Dict, Dict]]:
        """Create synthetic pairs for missing vulnerability types"""
        # Check which types are missing
        existing_types = set()
        for _, buggy in self.pairs:
            if 'vulnerability_type' in buggy:
                existing_types.add(buggy['vulnerability_type'])
        
        all_types = set(['reentrancy', 'integer_overflow', 'timestamp_dependence', 
                         'unchecked_call', 'tx_origin', 'transaction_order_dependence', 
                         'unhandled_exceptions'])
        
        missing_types = all_types - existing_types
        
        synthetic_pairs = []
        for vuln_type in missing_types:
            # Create at least 10 synthetic samples per missing type
            for i in range(10):
                clean_code, buggy_code = self._create_synthetic_pair(vuln_type)
                
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
                
                synthetic_pairs.append((clean, buggy))
        
        return synthetic_pairs
    
    def _create_simple_clean(self, buggy_code: str) -> str:
        """Create a simple clean version by removing obvious vulnerability patterns"""
        if not buggy_code:
            return ""
            
        clean = buggy_code
        
        # Remove vulnerability functions
        vuln_patterns = [
            r'function\s+\w*_(?:re_ent|intou|tmstmp|TOD|txorigin|unchk)\d*[^}]*}',
            r'function\s+(?:bug_|play_|claim|withdraw)(?:re_ent|intou|tmstmp|TOD|txorigin|unchk)\d*[^}]*}',
            r'function\s+callnotchecked_unchk\d*[^}]*}',
            r'//\s*(?:BUG|vulnerability|VULNERABILITY)[^\n]*\n',
        ]
        
        for pattern in vuln_patterns:
            clean = re.sub(pattern, '', clean, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove vulnerability state variables
        var_patterns = [
            r'(?:address|uint\d*)\s+\w*_(?:tmstmp|TOD)\d*\s*;',
            r'mapping\s*\([^)]+\)\s*\w*_TOD\d*\s*;',
        ]
        
        for pattern in var_patterns:
            clean = re.sub(pattern, '', clean)
        
        # Clean up extra whitespace
        clean = re.sub(r'\n\s*\n\s*\n', '\n\n', clean)
        
        return clean.strip()
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if idx >= len(self.pairs):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.pairs)}")
        
        clean, buggy = self.pairs[idx]
        
        # Ensure valid data
        clean_code = clean.get('code', '') if isinstance(clean, dict) else ''
        buggy_code = buggy.get('code', '') if isinstance(buggy, dict) else ''
        
        # Get vulnerability type and normalize it
        vuln_type = buggy.get('vulnerability_type', clean.get('vulnerability_type', 'unknown'))
        
        # Normalize using the comprehensive mapping
        preprocessor = SolidiFIPreprocessor('.')  # Dummy path
        vuln_type = preprocessor.normalize_vulnerability_type(vuln_type)
        
        # If still unknown, try to infer from code
        if vuln_type == 'unknown':
            vuln_type = self._infer_vulnerability_type(buggy_code)
        
        # Extract features
        clean_features = self._extract_features(clean_code)
        buggy_features = self._extract_features(buggy_code)
        
        # Get vulnerability labels
        vuln_labels = self._get_vulnerability_labels(vuln_type)
        
        return {
            'clean_code': clean_code,
            'buggy_code': buggy_code,
            'clean_features': clean_features,
            'buggy_features': buggy_features,
            'vulnerability_labels': vuln_labels,
            'vulnerability_locations': buggy.get('vulnerability_snippets', []),
            'vulnerability_type': vuln_type,
            'source_file': buggy.get('source_file', 'unknown')
        }
    
    def _infer_vulnerability_type(self, code: str) -> str:
        """Infer vulnerability type from code patterns"""
        # Check for specific patterns in the code
        if any(pattern in code for pattern in ['_re_ent', 'withdraw_re_ent', 'claimReward_re_ent']):
            return 'reentrancy'
        elif any(pattern in code for pattern in ['_intou', 'bug_intou', 'transfer_intou']):
            return 'integer_overflow'
        elif any(pattern in code for pattern in ['_tmstmp', 'bug_tmstmp', 'play_tmstmp']):
            return 'timestamp_dependence'
        elif any(pattern in code for pattern in ['_unchk', 'bug_unchk_send', 'callnotchecked_unchk']):
            return 'unchecked_call'
        elif any(pattern in code for pattern in ['_txorigin', 'bug_txorigin', 'transferTo_txorigin']):
            return 'tx_origin'
        elif any(pattern in code for pattern in ['_TOD', 'play_TOD', 'winner_TOD', 'reward_TOD']):
            return 'transaction_order_dependence'
        elif any(pattern in code for pattern in ['callnotchecked', 'unhandled', 'exception']):
            return 'unhandled_exceptions'
        else:
            return 'unknown'

    def _ensure_all_vulnerability_types(self):
        """Ensure dataset has samples for all 7 vulnerability types"""
        all_types = ['reentrancy', 'integer_overflow', 'timestamp_dependence', 
                     'unchecked_call', 'tx_origin', 'transaction_order_dependence', 
                     'unhandled_exceptions']
        
        # Count existing types
        type_counts = defaultdict(int)
        for clean, buggy in self.pairs:
            vuln_type = buggy.get('vulnerability_type', 'unknown')
            type_counts[vuln_type] += 1
        
        logger.info(f"Current distribution before augmentation: {dict(type_counts)}")
        
        # Find missing types
        missing_types = [t for t in all_types if type_counts[t] == 0]
        
        if missing_types:
            logger.info(f"Creating synthetic samples for missing types: {missing_types}")
            
            for vuln_type in missing_types:
                # Create 20 synthetic samples per missing type
                for i in range(20):
                    clean_code, buggy_code = self._create_synthetic_pair(vuln_type)
                    
                    clean = {
                        'code': clean_code,
                        'vulnerability_type': vuln_type,
                        'synthetic': True,
                        'source_file': f'synthetic_{vuln_type}_{i}.sol'
                    }
                    
                    buggy = {
                        'code': buggy_code,
                        'vulnerability_type': vuln_type,
                        'synthetic': True,
                        'vulnerability_snippets': [],
                        'source_file': f'synthetic_{vuln_type}_{i}.sol'
                    }
                    
                    self.pairs.append((clean, buggy))
            
            logger.info(f"Added synthetic samples. New dataset size: {len(self.pairs)}")
     
    def _extract_features(self, code: str) -> torch.Tensor:
        """Extract numerical features from code"""
        if not code:
            return torch.zeros(25)
            
        features = []
        
        # Basic statistics
        features.append(len(code))
        features.append(code.count('\n'))
        features.append(code.count('function'))
        features.append(code.count('require'))
        features.append(code.count('assert'))
        features.append(code.count('revert'))
        features.append(code.count('.send') + code.count('.call') + code.count('.transfer'))
        features.append(code.count('mapping'))
        features.append(code.count('modifier'))
        features.append(code.count('event'))
        
        # Vulnerability indicators
        features.append(int('msg.sender.send' in code or 'msg.sender.call' in code))
        features.append(int('block.timestamp' in code or 'now' in code))
        features.append(int('tx.origin' in code))
        features.append(int('delegatecall' in code))
        features.append(int('selfdestruct' in code or 'suicide' in code))
        
        # Pattern indicators for all 7 vulnerability types
        features.append(code.count('_re_ent'))
        features.append(code.count('_intou'))
        features.append(code.count('_tmstmp'))
        features.append(code.count('_TOD'))
        features.append(code.count('_txorigin'))
        features.append(code.count('_unchk'))
        features.append(code.count('callnotchecked'))
        
        # Complexity features
        features.append(code.count('if '))
        features.append(code.count('for '))
        features.append(code.count('while '))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_vulnerability_labels(self, vuln_type: str) -> torch.Tensor:
        """Get one-hot encoded vulnerability labels"""
        # All 7 vulnerability types in SolidiFI
        vuln_types = [
            'reentrancy', 
            'integer_overflow', 
            'timestamp_dependence',
            'unchecked_call', 
            'tx_origin', 
            'transaction_order_dependence',
            'unhandled_exceptions'
        ]
        
        labels = torch.zeros(len(vuln_types))
        
        if vuln_type in vuln_types:
            idx = vuln_types.index(vuln_type)
            labels[idx] = 1.0
        
        return labels


class PatternExtractor:
    """Extract and learn vulnerability patterns from dataset"""
    
    def __init__(self):
        self.patterns = defaultdict(list)
        self.pattern_embeddings = {}
        # Fix: Initialize with proper nested structure
        self.pattern_statistics = defaultdict(lambda: {
            'total': 0,
            'subtypes': defaultdict(int),
            'severities': defaultdict(int),
            'functions': defaultdict(int)
        })
        
    def extract_patterns(self, dataset: VulnerabilityDataset):
        """Extract all vulnerability patterns from dataset"""
        logger.info("Extracting vulnerability patterns from dataset...")
        
        if len(dataset) == 0:
            logger.warning("Empty dataset, no patterns to extract")
            return
        
        # Limit extraction for large datasets
        max_samples = min(1000, len(dataset))
        
        for idx in tqdm(range(max_samples), desc="Extracting patterns"):
            try:
                item = dataset[idx]
                vuln_locations = item.get('vulnerability_locations', [])
                buggy_code = item.get('buggy_code', '')
                vuln_type = item.get('vulnerability_type', 'unknown')
                
                for vuln in vuln_locations:
                    context = self._extract_context(buggy_code, vuln)
                    
                    pattern_info = {
                        'pattern': vuln.get('pattern', '')[:500],  # Limit length
                        'context': context,
                        'full_snippet': self._get_full_snippet(buggy_code, vuln),
                        'severity': vuln.get('severity', 'medium'),
                        'function': vuln.get('function', 'unknown'),
                        'subtype': vuln.get('subtype', 'generic'),
                        'vulnerability_type': vuln.get('type', vuln_type)
                    }
                    
                    self.patterns[pattern_info['vulnerability_type']].append(pattern_info)
                    
                    # Update statistics - now with proper structure
                    vuln_type_key = pattern_info['vulnerability_type']
                    self.pattern_statistics[vuln_type_key]['total'] += 1
                    self.pattern_statistics[vuln_type_key]['subtypes'][pattern_info['subtype']] += 1
                    self.pattern_statistics[vuln_type_key]['severities'][pattern_info['severity']] += 1
                    if pattern_info['function']:
                        self.pattern_statistics[vuln_type_key]['functions'][pattern_info['function']] += 1
                    
            except Exception as e:
                logger.debug(f"Error extracting patterns from item {idx}: {str(e)}")
                continue
        
        self._print_statistics()
    
    def _extract_context(self, code: str, vuln: Dict, context_size: int = 5) -> Dict:
        """Extract context around vulnerability"""
        if not code:
            return {'before': '', 'after': '', 'function': 'unknown'}
            
        lines = code.split('\n')
        vuln_line = vuln.get('line', 1) - 1  # Convert to 0-based
        
        # Ensure line is within bounds
        vuln_line = max(0, min(vuln_line, len(lines) - 1))
        
        start_line = max(0, vuln_line - context_size)
        end_line = min(len(lines), vuln_line + context_size + 1)
        
        return {
            'before': '\n'.join(lines[start_line:vuln_line]),
            'after': '\n'.join(lines[vuln_line + 1:end_line]),
            'function': vuln.get('function', 'unknown')
        }
    
    def _get_full_snippet(self, code: str, vuln: Dict) -> str:
        """Get full code snippet around vulnerability"""
        if not code:
            return ""
            
        start = max(0, vuln.get('start', 0) - 200)
        end = min(len(code), vuln.get('end', len(code)) + 200)
        return code[start:end][:1000]  # Limit to 1000 chars
    
    def _print_statistics(self):
        """Print pattern extraction statistics"""
        logger.info("\nPattern Extraction Statistics:")
        logger.info("="*60)
        
        total_patterns = sum(stats['total'] for stats in self.pattern_statistics.values())
        logger.info(f"Total patterns extracted: {total_patterns}")
        
        for vuln_type, stats in self.pattern_statistics.items():
            if stats['total'] > 0:
                logger.info(f"\n{vuln_type}:")
                logger.info(f"  Total patterns: {stats['total']}")
                
                # Safely convert defaultdicts to regular dicts for printing
                if stats['subtypes']:
                    logger.info(f"  Subtypes: {dict(stats['subtypes'])}")
                else:
                    logger.info(f"  Subtypes: {{}}")
                    
                if stats['severities']:
                    logger.info(f"  Severities: {dict(stats['severities'])}")
                else:
                    logger.info(f"  Severities: {{}}")
                    
                # Optionally show top functions
                if stats['functions']:
                    top_functions = sorted(stats['functions'].items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info(f"  Top functions: {dict(top_functions)}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about extracted patterns"""
        stats = {}
        for vuln_type, patterns in self.patterns.items():
            unique_functions = set()
            unique_subtypes = set()
            severities = defaultdict(int)
            
            for pattern in patterns:
                unique_functions.add(pattern.get('function', 'unknown'))
                unique_subtypes.add(pattern.get('subtype', 'generic'))
                severities[pattern.get('severity', 'medium')] += 1
            
            stats[vuln_type] = {
                'count': len(patterns),
                'functions': list(unique_functions),
                'subtypes': list(unique_subtypes),
                'severities': dict(severities)
            }
        
        return stats


def prepare_solidifi_data(data_directory: str):
    """Complete pipeline to prepare SolidiFI data"""
    logger.info(f"Preparing SolidiFI data from: {data_directory}")
    
    # Check if directory exists
    data_path = Path(data_directory)
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_directory}")
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    try:
        # Process dataset
        preprocessor = SolidiFIPreprocessor(data_directory)
        processed_data = preprocessor.process_dataset()
        
        # Get contracts
        clean_contracts = processed_data.get('clean', [])
        buggy_contracts = processed_data.get('buggy', [])
        
        logger.info(f"Processed {len(clean_contracts)} clean and {len(buggy_contracts)} buggy contracts")
        
        # Handle case where no clean contracts were generated
        if not clean_contracts and buggy_contracts:
            logger.warning("No clean contracts found, generating from buggy contracts...")
            for buggy in buggy_contracts[:min(500, len(buggy_contracts))]:
                if 'code' in buggy and buggy['code']:
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
        if not clean_contracts and not buggy_contracts:
            raise ValueError("No contracts found after preprocessing")
        
        dataset = VulnerabilityDataset(clean_contracts, buggy_contracts)
        
        if len(dataset) == 0:
            raise ValueError("Created dataset is empty")
        
        # Extract patterns
        pattern_extractor = PatternExtractor()
        pattern_extractor.extract_patterns(dataset)
        
        logger.info(f"Dataset preparation complete!")
        logger.info(f"Total dataset pairs: {len(dataset)}")
        
        return dataset, pattern_extractor
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Test with the full SolidiFI dataset
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './cabis_project/data/solidifi/'
    
    try:
        dataset, patterns = prepare_solidifi_data(data_dir)
        
        if dataset:
            print("\nDataset created successfully!")
            print(f"Total training pairs: {len(dataset)}")
            
            # Test loading a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"\nSample item keys: {sample.keys()}")
                print(f"Vulnerability type: {sample['vulnerability_type']}")
                print(f"Source file: {sample['source_file']}")
                print(f"Clean code length: {len(sample['clean_code'])}")
                print(f"Buggy code length: {len(sample['buggy_code'])}")
                print(f"Features shape: {sample['clean_features'].shape}")
                print(f"Labels shape: {sample['vulnerability_labels'].shape}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()