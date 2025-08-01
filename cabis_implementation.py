# -*- coding: utf-8 -*-
"""

@author: Willie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union
#import ast
import re
import os
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import logging
import subprocess
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Complete vulnerability type mapping for SolidiFI
VULNERABILITY_TYPES = {
    'reentrancy': 0,
    'integer_overflow': 1,
    'timestamp_dependence': 2,
    'unchecked_call': 3,
    'tx_origin': 4,
    'transaction_order_dependence': 5,
    'unhandled_exceptions': 6
}

VULNERABILITY_PATTERNS = {
    'reentrancy': ['_re_ent', 'bug_re_ent', 'claimReward', 'withdraw'],
    'integer_overflow': ['_intou', 'bug_intou', 'transfer_intou', 'increaseLockTime_intou'],
    'timestamp_dependence': ['_tmstmp', 'bug_tmstmp', 'play_tmstmp', 'bugv_tmstmp'],
    'unchecked_call': ['_unchk', 'bug_unchk_send', 'withdrawLeftOver_unchk', 'callnotchecked_unchk'],
    'tx_origin': ['_txorigin', 'bug_txorigin', 'transferTo_txorigin', 'withdrawAll_txorigin'],
    'transaction_order_dependence': ['_TOD', 'play_TOD', 'getReward_TOD', 'setReward_TOD', 'winner_TOD'],
    'unhandled_exceptions': ['callnotchecked', 'unhandled', '_unchk']
}

@dataclass
class VulnerabilityPattern:
    """Enhanced vulnerability pattern with more attributes"""
    pattern_id: str
    vulnerability_type: str
    code_pattern: str
    injection_context: Dict
    severity: float
    subtype: str
    confidence: float
    source_file: Optional[str] = None
    
@dataclass
class InjectionPoint:
    """Enhanced injection point with more context"""
    line_number: int
    function_name: str
    ast_node: str
    context_embedding: torch.Tensor
    receptiveness_score: float
    vulnerability_type: str
    injection_strategy: str

class ImprovedHierarchicalCodeEncoder(nn.Module):
    """Enhanced hierarchical encoder for smart contract code"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config['model']['hidden_dim']
        
        # Initialize or load CodeBERT
        try:
            self.token_encoder = RobertaModel.from_pretrained('microsoft/codebert-base')
            self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        except:
            logger.warning("Could not load CodeBERT, using smaller model")
            config_roberta = RobertaConfig(
                vocab_size=config['model']['vocab_size'],
                hidden_size=self.hidden_dim,
                num_hidden_layers=config['model']['num_layers'],
                num_attention_heads=config['model']['num_heads'],
                intermediate_size=self.hidden_dim * 4,
                max_position_embeddings=config['model']['max_seq_length']
            )
            self.token_encoder = RobertaModel(config_roberta)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Projection layer to match dimensions
        encoder_hidden_size = self.token_encoder.config.hidden_size
        if encoder_hidden_size != self.hidden_dim:
            self.projection = nn.Linear(encoder_hidden_size, self.hidden_dim)
        else:
            self.projection = nn.Identity()
        
        # Statement-level encoder (Graph Attention)
        self.statement_encoder = GraphAttentionNetwork(
            self.hidden_dim, 
            self.hidden_dim, 
            config['model']['num_heads']
        )
        
        # Function-level encoder
        self.function_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=config['model']['num_heads'],
                dim_feedforward=self.hidden_dim * 4,
                dropout=config['model'].get('dropout', 0.1)
            ),
            num_layers=config['model']['num_layers'] // 2
        )
        
        # Contract-level encoder
        self.contract_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=config['model']['num_heads'],
                dim_feedforward=self.hidden_dim * 4,
                dropout=config['model'].get('dropout', 0.1)
            ),
            num_layers=config['model']['num_layers'] // 2
        )
        
        # Vulnerability-specific encoders
        self.vuln_encoders = nn.ModuleDict({
            vuln_type: nn.Linear(self.hidden_dim, self.hidden_dim)
            for vuln_type in VULNERABILITY_TYPES.keys()
        })
        
    def forward(self, code: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Hierarchical encoding of smart contract code"""
        # Handle both single string and batch of strings
        if isinstance(code, str):
            code_list = [code]
            single_input = True
        else:
            code_list = code
            single_input = False
        
        # Process each contract
        results = []
        for contract_code in code_list:
            # Encode tokens
            token_embeddings = self._encode_tokens(contract_code)
            
            # Parse contract structure
            ast_graph, cfg = self._parse_contract(contract_code)
            
            # Hierarchical encoding
            statement_embeddings = self._encode_statements(token_embeddings, ast_graph)
            function_embeddings = self._encode_functions(statement_embeddings, cfg)
            contract_embedding = self._encode_contract(function_embeddings)
            
            # Extract vulnerability-specific features
            vuln_features = self._extract_vulnerability_features(contract_code, contract_embedding)
            
            result = {
                'tokens': token_embeddings,
                'statements': statement_embeddings,
                'functions': function_embeddings,
                'contract': contract_embedding,
                'vulnerability_features': vuln_features
            }
            
            results.append(result)
        
        # Batch results
        if single_input:
            return results[0]
        else:
            return self._batch_results(results)

    
    def _encode_tokens(self, code: str) -> torch.Tensor:
        """Encode tokens using CodeBERT"""
        # Tokenize with truncation
        inputs = self.tokenizer(
            code, 
            return_tensors='pt', 
            max_length=self.config['model']['max_seq_length'],
            truncation=True,
            padding='max_length'
        )
        
        # Move to correct device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            outputs = self.token_encoder(**inputs)
        
        # Project to correct dimension
        token_embeddings = self.projection(outputs.last_hidden_state)
        
        return token_embeddings
    
    def _parse_contract(self, code: str) -> Tuple[nx.DiGraph, nx.DiGraph]:
        """Parse Solidity contract into AST and CFG"""
        # Simplified parsing - in production use py-solc-ast
        ast_graph = self._build_simplified_ast(code)
        cfg = self._build_simplified_cfg(code)
        return ast_graph, cfg
    
    def _build_simplified_ast(self, code: str) -> nx.DiGraph:
        """Build simplified AST representation"""
        ast_graph = nx.DiGraph()
        
        # Add root node
        ast_graph.add_node(0, type='contract', code=code[:100])
        
        # Extract functions and add as nodes
        function_pattern = r'function\s+(\w+)\s*\([^)]*\)[^{]*{([^}]+)}'
        node_id = 1
        
        for match in re.finditer(function_pattern, code, re.DOTALL):
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Add function node
            ast_graph.add_node(node_id, type='function', name=func_name)
            ast_graph.add_edge(0, node_id)
            
            # Add statement nodes
            statements = func_body.split(';')
            for stmt in statements:
                if stmt.strip():
                    node_id += 1
                    ast_graph.add_node(node_id, type='statement', code=stmt.strip()[:50])
                    ast_graph.add_edge(node_id-1, node_id)
            
            node_id += 1
        
        return ast_graph
    
    def _build_simplified_cfg(self, code: str) -> nx.DiGraph:
        """Build simplified control flow graph"""
        cfg = nx.DiGraph()
        
        # Add entry node
        cfg.add_node(0, type='entry')
        
        # Extract basic blocks (simplified)
        lines = code.split('\n')
        current_block = []
        block_id = 1
        
        for line in lines:
            if any(keyword in line for keyword in ['if', 'for', 'while', 'require']):
                # End current block
                if current_block:
                    cfg.add_node(block_id, type='block', lines=len(current_block))
                    cfg.add_edge(block_id-1, block_id)
                    block_id += 1
                    current_block = []
                
                # Add control structure node
                cfg.add_node(block_id, type='control', statement=line.strip()[:50])
                cfg.add_edge(block_id-1, block_id)
                block_id += 1
            else:
                current_block.append(line)
        
        # Add final block
        if current_block:
            cfg.add_node(block_id, type='block', lines=len(current_block))
            cfg.add_edge(block_id-1, block_id)
        
        return cfg
    
    def _encode_statements(self, token_embeddings: torch.Tensor, ast_graph: nx.DiGraph) -> torch.Tensor:
        """Encode statements using graph attention over AST"""
        # Get statement nodes
        statement_nodes = [n for n, d in ast_graph.nodes(data=True) if d.get('type') == 'statement']
        
        # Fixed number of statements for consistency
        num_statements = 10  # Use a fixed number
        
        batch_size = token_embeddings.shape[0]
        hidden_dim = token_embeddings.shape[-1]
        
        statement_embeddings = torch.zeros(batch_size, num_statements, hidden_dim).to(token_embeddings.device)
        
        if not statement_nodes:
            # If no statements, use mean pooling of all tokens
            avg_embedding = token_embeddings.mean(dim=1, keepdim=True)
            statement_embeddings[:, 0, :] = avg_embedding.squeeze(1)
        else:
            # Pool tokens into statements
            actual_statements = min(len(statement_nodes), num_statements)
            tokens_per_statement = token_embeddings.shape[1] // actual_statements
            
            for i in range(actual_statements):
                start_idx = i * tokens_per_statement
                end_idx = min((i + 1) * tokens_per_statement, token_embeddings.shape[1])
                statement_embeddings[:, i, :] = token_embeddings[:, start_idx:end_idx, :].mean(dim=1)
        
        return statement_embeddings
    
    def _encode_functions(self, statement_embeddings: torch.Tensor, cfg: nx.DiGraph) -> torch.Tensor:
        """Encode functions using transformer over statements"""
        # Ensure we don't have a batch dimension issue
        if len(statement_embeddings.shape) == 2:
            statement_embeddings = statement_embeddings.unsqueeze(0)
        
        # Use function encoder
        function_embeddings = self.function_encoder(statement_embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Pool to fixed number of functions
        num_functions = 5  # Fixed number for consistency
        batch_size = function_embeddings.shape[0]
        hidden_dim = function_embeddings.shape[-1]
        
        pooled_functions = torch.zeros(batch_size, num_functions, hidden_dim).to(function_embeddings.device)
        
        # Average pool into fixed number of functions
        actual_functions = function_embeddings.shape[1]
        if actual_functions <= num_functions:
            pooled_functions[:, :actual_functions, :] = function_embeddings
        else:
            # Pool down to num_functions
            chunk_size = actual_functions // num_functions
            for i in range(num_functions):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_functions - 1 else actual_functions
                pooled_functions[:, i, :] = function_embeddings[:, start_idx:end_idx, :].mean(dim=1)
        
        return pooled_functions
    
    
    # Update _encode_contract to ensure it returns consistent shape:
    
    def _encode_contract(self, function_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode entire contract"""
        # Ensure we have the right shape
        if len(function_embeddings.shape) == 2:
            function_embeddings = function_embeddings.unsqueeze(0)
        
        # Pool function embeddings
        contract_embedding = self.contract_encoder(
            function_embeddings.transpose(0, 1)
        ).transpose(0, 1).mean(dim=1)
        
        # Ensure it's the right shape [batch_size, hidden_dim]
        if len(contract_embedding.shape) == 1:
            contract_embedding = contract_embedding.unsqueeze(0)
        
        return contract_embedding
        
    def _extract_vulnerability_features(self, code: str, contract_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract vulnerability-specific features"""
        vuln_features = {}
        
        for vuln_type, patterns in VULNERABILITY_PATTERNS.items():
            # Check for pattern presence
            pattern_count = sum(1 for pattern in patterns if pattern in code)
            
            # Create feature vector
            feature = torch.tensor([pattern_count], dtype=torch.float32).to(contract_embedding.device)
            
            # Apply vulnerability-specific encoder
            vuln_embedding = self.vuln_encoders[vuln_type](contract_embedding)
            
            # Combine with pattern count
            vuln_features[vuln_type] = vuln_embedding * (1 + feature * 0.1)
        
        return vuln_features
    
    def _batch_results(self, results: List[Dict]) -> Dict[str, torch.Tensor]:
        """Combine results from batch processing with proper padding/pooling"""
        batched = defaultdict(list)
        
        for result in results:
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    batched[key].append(value)
                elif isinstance(value, dict):
                    if key not in batched:
                        batched[key] = defaultdict(list)
                    for k, v in value.items():
                        batched[key][k].append(v)
        
        # Stack tensors with proper handling for different sizes
        final_result = {}
        for key, values in batched.items():
            if isinstance(values, list) and values and isinstance(values[0], torch.Tensor):
                # Special handling for functions to ensure consistent shape
                if key == 'functions':
                    # Ensure all function tensors have the same shape
                    target_shape = values[0].shape
                    processed_values = []
                    
                    for v in values:
                        if v.shape == target_shape:
                            processed_values.append(v)
                        elif len(v.shape) == len(target_shape):
                            # Same number of dimensions, just different sizes
                            # Pad or truncate to match target shape
                            if v.shape[-1] == target_shape[-1]:  # Same hidden dim
                                if v.shape[0] < target_shape[0]:
                                    # Pad
                                    padding = torch.zeros(target_shape[0] - v.shape[0], v.shape[-1], device=v.device)
                                    v = torch.cat([v, padding], dim=0)
                                elif v.shape[0] > target_shape[0]:
                                    # Truncate
                                    v = v[:target_shape[0]]
                                processed_values.append(v)
                            else:
                                # Different hidden dim, use first value
                                processed_values.append(values[0])
                        else:
                            # Different number of dimensions, use first value
                            processed_values.append(values[0])
                    
                    # Stack the processed values
                    final_result[key] = torch.stack(processed_values)
                
                elif key in ['contract', 'vulnerability_features']:
                    # These should be 1D per sample, stack them
                    processed_values = []
                    for v in values:
                        if len(v.shape) == 1:
                            processed_values.append(v)
                        elif len(v.shape) == 2 and v.shape[0] == 1:
                            processed_values.append(v.squeeze(0))
                        else:
                            # Pool if needed
                            if len(v.shape) >= 2:
                                processed_values.append(v.mean(dim=0))
                            else:
                                processed_values.append(v)
                    final_result[key] = torch.stack(processed_values)
                
                else:
                    # For other keys, use the existing logic
                    shapes = [v.shape for v in values]
                    if len(set(shapes)) == 1:
                        # All same shape, can stack directly
                        final_result[key] = torch.stack(values)
                    else:
                        # Different shapes, need to handle specially
                        # For sequence data (tokens, statements), pad to max length
                        max_seq_len = max(v.shape[0] if len(v.shape) >= 1 else 1 for v in values)
                        padded = []
                        for v in values:
                            if len(v.shape) == 2:  # [seq, hidden]
                                if v.shape[0] < max_seq_len:
                                    pad_size = max_seq_len - v.shape[0]
                                    padding = torch.zeros(pad_size, v.shape[1], device=v.device)
                                    v = torch.cat([v, padding], dim=0)
                                padded.append(v)
                            else:
                                padded.append(v)
                        
                        if padded and all(p.shape == padded[0].shape for p in padded):
                            final_result[key] = torch.stack(padded)
                        else:
                            # If still can't stack, use mean pooling as fallback
                            logger.debug(f"Using mean pooling for key {key} due to shape mismatch")
                            pooled = []
                            for v in padded:
                                if len(v.shape) >= 2:
                                    pooled.append(v.mean(dim=0))
                                else:
                                    pooled.append(v)
                            final_result[key] = torch.stack(pooled)
                            
            elif isinstance(values, defaultdict):
                # Handle nested dictionaries (e.g., vulnerability_features)
                final_result[key] = {}
                for k, v in values.items():
                    if v and isinstance(v[0], torch.Tensor):
                        try:
                            # Try to stack if all same shape
                            if all(tensor.shape == v[0].shape for tensor in v):
                                final_result[key][k] = torch.stack(v)
                            else:
                                # Otherwise, pad or pool
                                pooled = []
                                for tensor in v:
                                    if len(tensor.shape) >= 2:
                                        pooled.append(tensor.mean(dim=0) if len(tensor.shape) == 2 else tensor.mean(dim=1).squeeze(0))
                                    else:
                                        pooled.append(tensor)
                                final_result[key][k] = torch.stack(pooled)
                        except Exception as e:
                            logger.debug(f"Error stacking {key}[{k}]: {str(e)}")
                            final_result[key][k] = v[0]  # Just use first item as fallback
            else:
                final_result[key] = values
        
        return final_result


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for AST encoding"""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(in_features, num_heads, batch_first=True)
        self.projection = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, node_features: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply graph attention to node features"""
        # Self-attention
        attended, _ = self.attention(node_features, node_features, node_features)
        
        # Projection and normalization
        output = self.projection(attended)
        output = self.norm(output + node_features)
        output = self.dropout(output)
        
        return output
    
class EnhancedVulnerabilitySynthesizer(nn.Module):
    """Enhanced vulnerability synthesis with support for all 7 types"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config['model']['hidden_dim']
        self.num_vuln_types = len(VULNERABILITY_TYPES)
        self.memory_size = config['model']['memory_size']
        
        # Vulnerability pattern memory bank (learned patterns)
        self.pattern_memory = nn.Parameter(torch.randn(self.memory_size, self.hidden_dim))
        self.pattern_keys = nn.Parameter(torch.randn(self.memory_size, self.hidden_dim))
        
        # Fix: Use nn.ParameterDict for vulnerability-specific memories
        self.vuln_specific_memories = nn.ParameterDict({
            vuln_type: nn.Parameter(torch.randn(100, self.hidden_dim))
            for vuln_type in VULNERABILITY_TYPES.keys()
        })
        
        # Vulnerability type embeddings
        self.vuln_embeddings = nn.Embedding(self.num_vuln_types, self.hidden_dim)
        
        # Pattern synthesis networks for each vulnerability type
        self.pattern_generators = nn.ModuleDict({
            vuln_type: self._build_pattern_generator(vuln_type)
            for vuln_type in VULNERABILITY_TYPES.keys()
        })
        
        # Context preservation network
        self.context_preserving_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=config['model']['num_heads'],
                dim_feedforward=self.hidden_dim * 4,
                dropout=config['model'].get('dropout', 0.1)
            ),
            num_layers=3
        )
        
        # Injection point scorer
        self.injection_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Exploitability predictor
        self.exploitability_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        
    def _build_pattern_generator(self, vuln_type: str) -> nn.Module:
        """Build vulnerability-specific pattern generator"""
        if vuln_type == 'reentrancy':
            return self._build_reentrancy_generator()
        elif vuln_type == 'integer_overflow':
            return self._build_overflow_generator()
        elif vuln_type == 'timestamp_dependence':
            return self._build_timestamp_generator()
        elif vuln_type == 'unchecked_call':
            return self._build_unchecked_generator()
        elif vuln_type == 'tx_origin':
            return self._build_txorigin_generator()
        elif vuln_type == 'transaction_order_dependence':
            return self._build_tod_generator()
        elif vuln_type == 'unhandled_exceptions':
            return self._build_exception_generator()
        else:
            return self._build_generic_generator()
    
    def _build_generic_generator(self) -> nn.Module:
        """Build generic pattern generator"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh()
        )
    
    def _build_reentrancy_generator(self) -> nn.Module:
        """Reentrancy-specific generator"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh()
        )
    
    def _build_overflow_generator(self) -> nn.Module:
        """Integer overflow-specific generator"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh()
        )
    
    def _build_timestamp_generator(self) -> nn.Module:
        """Timestamp dependence generator"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
    
    def _build_unchecked_generator(self) -> nn.Module:
        """Unchecked call generator"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh()
        )
    
    def _build_txorigin_generator(self) -> nn.Module:
        """tx.origin generator"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
    
    def _build_tod_generator(self) -> nn.Module:
        """Transaction order dependence generator"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh()
        )
    
    def _build_exception_generator(self) -> nn.Module:
        """Unhandled exceptions generator"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh()
        )
     
    def forward(self, code_embeddings: Dict[str, torch.Tensor], 
                vuln_type: Union[int, str]) -> Tuple[torch.Tensor, List[InjectionPoint]]:
        """Generate vulnerability injection"""
        # Validate and fix shapes
        code_embeddings = self._validate_and_fix_shapes(code_embeddings)

        # Convert string to int if needed
        if isinstance(vuln_type, str):
            vuln_type_idx = VULNERABILITY_TYPES[vuln_type]
            vuln_type_str = vuln_type
        else:
            vuln_type_idx = vuln_type
            vuln_type_str = list(VULNERABILITY_TYPES.keys())[vuln_type_idx]
        
        # Get device
        device = code_embeddings['contract'].device
        
        # Get vulnerability embedding
        batch_size = code_embeddings['contract'].shape[0]
        vuln_embed = self.vuln_embeddings(
            torch.tensor([vuln_type_idx], device=device)
        ).expand(batch_size, -1)
        
        # Find optimal injection points
        injection_points = self._find_injection_points(
            code_embeddings, vuln_embed, vuln_type_str
        )
        
        # Retrieve relevant patterns from memory
        relevant_patterns = self._retrieve_patterns(
            injection_points, vuln_embed, vuln_type_str
        )
        
        # Generate vulnerability pattern
        vulnerability_pattern = self._synthesize_pattern(
            code_embeddings, vuln_embed, relevant_patterns, vuln_type_str
        )
        
        # Ensure context preservation
        preserved_embedding = self._preserve_context(
            code_embeddings['contract'], vulnerability_pattern
        )
        
        return preserved_embedding, injection_points
     
    def _find_injection_points(self, embeddings: Dict, vuln_embed: torch.Tensor, 
                              vuln_type: str) -> List[InjectionPoint]:
        """Find optimal injection points for specific vulnerability type"""
        injection_points = []
        
        # Get function embeddings
        if 'functions' in embeddings:
            function_embeds = embeddings['functions']
            
            # Debug: Check the actual shape
            logger.debug(f"Function embeds shape: {function_embeds.shape}")
            
            # Handle different possible shapes
            if len(function_embeds.shape) == 4:
                # Shape is [batch, 1, num_functions, hidden_dim] - remove extra dimension
                function_embeds = function_embeds.squeeze(1)
                logger.debug(f"Squeezed function embeds shape: {function_embeds.shape}")
            
            if len(function_embeds.shape) == 2:
                # Shape is [num_functions, hidden_dim] - add batch dimension
                function_embeds = function_embeds.unsqueeze(0)
                batch_size = 1
                num_functions, hidden_dim = function_embeds.shape[1], function_embeds.shape[2]
            elif len(function_embeds.shape) == 3:
                # Shape is [batch_size, num_functions, hidden_dim] - correct shape
                batch_size, num_functions, hidden_dim = function_embeds.shape
            elif len(function_embeds.shape) == 1:
                # Shape is [hidden_dim] - single function, add both dimensions
                function_embeds = function_embeds.unsqueeze(0).unsqueeze(0)
                batch_size, num_functions, hidden_dim = 1, 1, function_embeds.shape[2]
            else:
                logger.warning(f"Unexpected function_embeds shape after processing: {function_embeds.shape}")
                # Fall back to contract embedding
                if 'contract' in embeddings:
                    contract_embed = embeddings['contract']
                    if len(contract_embed.shape) == 1:
                        contract_embed = contract_embed.unsqueeze(0)
                    return self._get_default_injection_points(contract_embed, vuln_type)
                else:
                    return []
            
            # Ensure vuln_embed has the right shape [batch_size, hidden_dim]
            if len(vuln_embed.shape) == 1:
                vuln_embed = vuln_embed.unsqueeze(0)
            
            # Expand vulnerability embedding to match batch and function dimensions
            vuln_embed_expanded = vuln_embed.unsqueeze(1).expand(batch_size, num_functions, -1)
            
            # Get vulnerability-specific features
            vuln_features = embeddings.get('vulnerability_features', {}).get(vuln_type)
            if vuln_features is not None:
                # Handle vuln_features shape
                if len(vuln_features.shape) == 1:
                    vuln_features = vuln_features.unsqueeze(0)
                elif len(vuln_features.shape) == 3:
                    # If it has an extra dimension, squeeze it
                    vuln_features = vuln_features.squeeze(1)
                elif len(vuln_features.shape) == 4:
                    # If it has two extra dimensions, squeeze them
                    vuln_features = vuln_features.squeeze(1).squeeze(1)
                    
                # Now expand to match function dimensions
                vuln_features_expanded = vuln_features.unsqueeze(1).expand(batch_size, num_functions, -1)
            else:
                vuln_features_expanded = torch.zeros_like(vuln_embed_expanded)
            
            # Concatenate features
            combined = torch.cat([
                function_embeds,
                vuln_embed_expanded,
                vuln_features_expanded
            ], dim=-1)
            
            # Score each function
            scores = self.injection_scorer(combined).squeeze(-1)
            
            # Handle single function case
            if len(scores.shape) == 1:
                scores = scores.unsqueeze(0)
            
            # Get top-k injection points per batch item
            top_k = min(5, num_functions)
            top_scores, top_indices = torch.topk(scores, top_k, dim=1)
            
            # Create injection point objects
            for b in range(batch_size):
                for k in range(top_k):
                    if top_scores[b, k] > 0.5:  # Threshold
                        injection_points.append(InjectionPoint(
                            line_number=top_indices[b, k].item() * 10,  # Approximate
                            function_name=f"function_{top_indices[b, k].item()}",
                            ast_node='function',
                            context_embedding=function_embeds[b, top_indices[b, k]],
                            receptiveness_score=top_scores[b, k].item(),
                            vulnerability_type=vuln_type,
                            injection_strategy=self._get_injection_strategy(vuln_type)
                        ))
        else:
            # No function embeddings, use contract embedding
            if 'contract' in embeddings:
                contract_embed = embeddings['contract']
                if len(contract_embed.shape) == 1:
                    contract_embed = contract_embed.unsqueeze(0)
                return self._get_default_injection_points(contract_embed, vuln_type)
        
        return injection_points                                

    def _get_default_injection_points(self, contract_embed: torch.Tensor, vuln_type: str) -> List[InjectionPoint]:
        """Get default injection points when function analysis is not available"""
        injection_points = []
        
        # Create a default injection point
        injection_points.append(InjectionPoint(
            line_number=10,  # Default line
            function_name="default_function",
            ast_node='function',
            context_embedding=contract_embed.squeeze(0) if contract_embed.shape[0] == 1 else contract_embed[0],
            receptiveness_score=0.8,  # Default high score
            vulnerability_type=vuln_type,
            injection_strategy=self._get_injection_strategy(vuln_type)
        ))
        
        return injection_points
    
        
    def _get_injection_strategy(self, vuln_type: str) -> str:
        """Get injection strategy for vulnerability type"""
        strategies = {
            'reentrancy': 'state_after_call',
            'integer_overflow': 'unchecked_arithmetic',
            'timestamp_dependence': 'timestamp_comparison',
            'unchecked_call': 'ignore_return_value',
            'tx_origin': 'use_tx_origin',
            'transaction_order_dependence': 'race_condition',
            'unhandled_exceptions': 'no_exception_handling'
        }
        return strategies.get(vuln_type, 'generic')
    
    def _retrieve_patterns(self, injection_points: List[InjectionPoint], 
                          vuln_embed: torch.Tensor, vuln_type: str) -> torch.Tensor:
        """Retrieve relevant vulnerability patterns from memory"""
        if not injection_points:
            # Return average pattern
            return self.pattern_memory.mean(dim=0, keepdim=True)
        
        # Get device
        device = vuln_embed.device
        
        # Ensure vuln_embed has proper shape
        if len(vuln_embed.shape) == 1:
            vuln_embed = vuln_embed.unsqueeze(0)
        
        # Compute query from injection points
        if injection_points:
            context_embeddings = []
            for p in injection_points[:5]:
                ctx_embed = p.context_embedding
                if len(ctx_embed.shape) == 0:  # Scalar
                    ctx_embed = ctx_embed.unsqueeze(0)
                elif len(ctx_embed.shape) > 1:  # Multi-dimensional
                    ctx_embed = ctx_embed.flatten()
                context_embeddings.append(ctx_embed)
            
            # Stack context embeddings
            if context_embeddings:
                context_stack = torch.stack(context_embeddings)
                query = context_stack.mean(dim=0) + vuln_embed.squeeze()
            else:
                query = vuln_embed.squeeze()
        else:
            query = vuln_embed.squeeze()
        
        # Ensure query is 1D
        if len(query.shape) > 1:
            query = query.flatten()
        if len(query.shape) == 0:
            query = query.unsqueeze(0)
        
        # Ensure query has the right size
        if query.shape[0] != self.hidden_dim:
            # Pad or truncate
            if query.shape[0] < self.hidden_dim:
                padding = torch.zeros(self.hidden_dim - query.shape[0], device=device)
                query = torch.cat([query, padding])
            else:
                query = query[:self.hidden_dim]
        
        # Retrieve from general memory
        attention_scores = F.softmax(
            torch.matmul(query, self.pattern_keys.T) / np.sqrt(self.hidden_dim), 
            dim=-1
        )
        general_pattern = torch.matmul(attention_scores, self.pattern_memory)
        
        # Retrieve from vulnerability-specific memory
        vuln_memory = self.vuln_specific_memories[vuln_type]
        vuln_attention = F.softmax(
            torch.matmul(query, vuln_memory.T) / np.sqrt(self.hidden_dim),
            dim=-1
        )
        specific_pattern = torch.matmul(vuln_attention, vuln_memory)
        
        # Combine patterns
        combined_pattern = (general_pattern + specific_pattern) / 2
        
        return combined_pattern
    
    def _synthesize_pattern(self, embeddings: Dict, vuln_embed: torch.Tensor,
                           retrieved_patterns: torch.Tensor, vuln_type: str) -> torch.Tensor:
        """Synthesize vulnerability pattern"""
        # Get contract embedding
        contract_embed = embeddings['contract']
        
        # Concatenate inputs
        if len(retrieved_patterns.shape) == 1:
            retrieved_patterns = retrieved_patterns.unsqueeze(0)
        if len(retrieved_patterns.shape) == 2 and retrieved_patterns.shape[0] == 1:
            retrieved_patterns = retrieved_patterns.expand(contract_embed.shape[0], -1)
        
        synthesis_input = torch.cat([contract_embed, retrieved_patterns], dim=-1)
        
        # Generate pattern using vulnerability-specific generator
        pattern = self.pattern_generators[vuln_type](synthesis_input)
        
        # Add noise for diversity
        noise = torch.randn_like(pattern) * 0.05
        pattern = pattern + noise
        
        return pattern
    
    def _preserve_context(self, original_embedding: torch.Tensor, 
                         vulnerability_pattern: torch.Tensor) -> torch.Tensor:
        """Preserve functional context while injecting vulnerability"""
        # Combine original and vulnerability pattern
        combined = original_embedding + vulnerability_pattern * 0.3
        
        # Apply context preservation
        if len(combined.shape) == 2:
            combined = combined.unsqueeze(1)
        
        preserved = self.context_preserving_layer(
            combined.transpose(0, 1)
        ).transpose(0, 1).squeeze(1)
        
        return preserved
    
    def predict_exploitability(self, original_embedding: torch.Tensor,
                              modified_embedding: torch.Tensor) -> torch.Tensor:
        """Predict if injected vulnerability is exploitable"""
        combined = torch.cat([original_embedding, modified_embedding], dim=-1)
        return self.exploitability_predictor(combined)

    def _validate_and_fix_shapes(self, embeddings: Dict) -> Dict:
        """Validate and fix tensor shapes in embeddings"""
        fixed_embeddings = {}
        
        for key, value in embeddings.items():
            if isinstance(value, torch.Tensor):
                if key == 'functions' and len(value.shape) == 4:
                    # Remove extra dimension from functions
                    fixed_embeddings[key] = value.squeeze(1)
                elif key == 'contract' and len(value.shape) > 2:
                    # Contract should be [batch_size, hidden_dim]
                    while len(value.shape) > 2:
                        value = value.squeeze(1)
                    fixed_embeddings[key] = value
                else:
                    fixed_embeddings[key] = value
            elif isinstance(value, dict):
                # Handle vulnerability_features
                fixed_dict = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor) and len(v.shape) > 2:
                        # Should be [batch_size, hidden_dim]
                        while len(v.shape) > 2:
                            v = v.squeeze(1)
                        fixed_dict[k] = v
                    else:
                        fixed_dict[k] = v
                fixed_embeddings[key] = fixed_dict
            else:
                fixed_embeddings[key] = value
        
        return fixed_embeddings    

class DiverseReentrancyInjectionStrategy:
    """Reentrancy injection with high diversity"""
    
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        
        # Diverse patterns
        patterns = [
            # Pattern 1: Classic reentrancy
            lambda n, v: f"""
    function withdraw_re_ent{n}() public {{
        uint256 {v} = balances[msg.sender];
        require({v} > 0, "Insufficient balance");
        (bool success,) = msg.sender.call{{value: {v}}}("");
        require(success, "Transfer failed");
        balances[msg.sender] = 0;
    }}""",
            # Pattern 2: Reentrancy with event
            lambda n, v: f"""
    event Withdrawn(address user, uint256 {v});
    
    function claim_re_ent{n}() public {{
        uint256 {v} = balances[msg.sender];
        msg.sender.call{{value: {v}}}("");
        emit Withdrawn(msg.sender, {v});
        balances[msg.sender] = 0;
    }}""",
            # Pattern 3: Reentrancy in reward system
            lambda n, v: f"""
    mapping(address => uint256) rewards_re_ent{n};
    
    function claimReward_re_ent{n}() public {{
        uint256 {v} = rewards_re_ent{n}[msg.sender];
        if ({v} > 0) {{
            (bool ok,) = msg.sender.call{{value: {v}}}("");
            if (ok) {{
                rewards_re_ent{n}[msg.sender] = 0;
            }}
        }}
    }}""",
            # Pattern 4: Multi-call reentrancy
            lambda n, v: f"""
    function withdrawAll_re_ent{n}() public {{
        uint256 {v} = balances[msg.sender];
        uint256 bonus = {v} / 10;
        msg.sender.call{{value: {v}}}("");
        msg.sender.call{{value: bonus}}("");
        balances[msg.sender] = 0;
    }}""",
            # Pattern 5: Reentrancy with modifier
            lambda n, v: f"""
    bool locked_re_ent{n};
    
    function safeWithdraw_re_ent{n}() public {{
        require(!locked_re_ent{n}, "Locked");
        uint256 {v} = balances[msg.sender];
        locked_re_ent{n} = true;
        (bool sent,) = msg.sender.call{{value: {v}}}("");
        balances[msg.sender] = 0;
        locked_re_ent{n} = false; // Wrong order
    }}"""
        ]
        
        # Select pattern based on diversity level
        num_patterns = len(patterns)
        pattern_idx = int(diversity_level * (num_patterns - 1))
        pattern_idx = min(max(0, pattern_idx + random.randint(-1, 1)), num_patterns - 1)
        
        # Generate diverse names
        func_num = random.randint(1, 999)
        var_names = ['amount', 'value', 'balance', 'funds', 'payment', 'withdrawal']
        var_name = random.choice(var_names)
        
        # Apply pattern
        vuln_code = patterns[pattern_idx](func_num, var_name)
        
        # Find injection point
        last_brace = code.rfind('}')
        if last_brace == -1:
            return code + vuln_code
        
        return code[:last_brace] + vuln_code + code[last_brace:]

class DiverseIntegerOverflowInjectionStrategy:
    """Integer overflow/underflow injection with high diversity"""
    
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        
        # Diverse patterns
        patterns = [
            # Pattern 1: Classic uint8 overflow
            lambda n, v: f"""
    function bug_intou{n}() public {{
        uint8 {v} = 255;
        {v} = {v} + 1;  // Overflow: 255 + 1 = 0
        balances[msg.sender] = {v};
    }}""",
            # Pattern 2: Underflow in transfer
            lambda n, v: f"""
    function transfer_intou{n}(address _to, uint256 _{v}) public {{
        require(balances[msg.sender] - _{v} >= 0);  // Always true for uint
        balances[msg.sender] -= _{v};
        balances[_to] += _{v};
    }}""",
            # Pattern 3: Multiplication overflow
            lambda n, v: f"""
    uint256 multiplier_intou{n} = 10;
    
    function multiply_intou{n}(uint256 {v}) public {{
        uint256 result = {v} * multiplier_intou{n};  // Can overflow
        balances[msg.sender] = result;
    }}""",
            # Pattern 4: Time-based overflow
            lambda n, v: f"""
    function increaseLockTime_intou{n}(uint256 _secondsToIncrease) public {{
        lockTime[msg.sender] += _secondsToIncrease;  // Can overflow
        if (lockTime[msg.sender] < _secondsToIncrease) {{
            lockTime[msg.sender] = type(uint256).max;
        }}
    }}""",
            # Pattern 5: Batch operation overflow
            lambda n, v: f"""
    function batchTransfer_intou{n}(address[] memory _receivers, uint256 _{v}) public {{
        uint256 total = _receivers.length * _{v};  // Can overflow
        require(balances[msg.sender] >= total);
        
        for (uint i = 0; i < _receivers.length; i++) {{
            balances[_receivers[i]] += _{v};
        }}
        balances[msg.sender] -= total;
    }}"""
        ]
        
        # Select pattern based on diversity level
        num_patterns = len(patterns)
        pattern_idx = int(diversity_level * (num_patterns - 1))
        pattern_idx = min(max(0, pattern_idx + random.randint(-1, 1)), num_patterns - 1)
        
        # Generate diverse names
        func_num = random.randint(1, 999)
        var_names = ['amount', 'value', 'balance', 'tokens', 'counter', 'total']
        var_name = random.choice(var_names)
        
        # Apply pattern
        vuln_code = patterns[pattern_idx](func_num, var_name)
        
        # Find injection point
        last_brace = code.rfind('}')
        if last_brace == -1:
            return code + vuln_code
        
        return code[:last_brace] + vuln_code + code[last_brace:]


class DiverseTimestampInjectionStrategy:
    """Timestamp dependence injection with high diversity"""
    
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        
        patterns = [
            # Pattern 1: Simple timestamp equality
            lambda n, v: f"""
    uint256 public winner_tmstmp{n};
    
    function play_tmstmp{n}(uint256 _{v}) public {{
        if (_{v} == block.timestamp) {{
            winner_tmstmp{n} = block.timestamp;
            payable(msg.sender).transfer(address(this).balance);
        }}
    }}""",
            # Pattern 2: Timestamp modulo randomness
            lambda n, v: f"""
    mapping(uint256 => address) public winners_tmstmp{n};
    
    function bug_tmstmp{n}() public view returns (bool) {{
        return block.timestamp % 15 == 0;  // Predictable
    }}
    
    function claim_tmstmp{n}() public {{
        require(bug_tmstmp{n}(), "Not lucky time");
        winners_tmstmp{n}[block.timestamp] = msg.sender;
    }}""",
            # Pattern 3: Time window vulnerability
            lambda n, v: f"""
    uint256 public startTime_tmstmp{n} = block.timestamp;
    
    function withdraw_tmstmp{n}() public {{
        require(block.timestamp >= startTime_tmstmp{n} + 1 days, "Too early");
        require(block.timestamp <= startTime_tmstmp{n} + 2 days, "Too late");
        payable(msg.sender).transfer(balances[msg.sender]);
        balances[msg.sender] = 0;
    }}""",
            # Pattern 4: Timestamp-based lottery
            lambda n, v: f"""
    address[] public players_tmstmp{n};
    
    function enter_tmstmp{n}() public payable {{
        require(msg.value >= 0.1 ether);
        players_tmstmp{n}.push(msg.sender);
    }}
    
    function pickWinner_tmstmp{n}() public {{
        uint winner = block.timestamp % players_tmstmp{n}.length;
        payable(players_tmstmp{n}[winner]).transfer(address(this).balance);
    }}""",
            # Pattern 5: Deadline manipulation
            lambda n, v: f"""
    mapping(address => uint256) public deadlines_tmstmp{n};
    
    function setDeadline_tmstmp{n}() public {{
        deadlines_tmstmp{n}[msg.sender] = block.timestamp + 1 hours;
    }}
    
    function execute_tmstmp{n}() public {{
        require(block.timestamp == deadlines_tmstmp{n}[msg.sender], "Not exact time");
        // Miner can manipulate to hit exact timestamp
        balances[msg.sender] *= 2;
    }}"""
        ]
        
        num_patterns = len(patterns)
        pattern_idx = int(diversity_level * (num_patterns - 1))
        pattern_idx = min(max(0, pattern_idx + random.randint(-1, 1)), num_patterns - 1)
        
        func_num = random.randint(1, 999)
        var_names = ['guess', 'time', 'seed', 'nonce', 'lucky']
        var_name = random.choice(var_names)
        
        vuln_code = patterns[pattern_idx](func_num, var_name)
        
        last_brace = code.rfind('}')
        return code[:last_brace] + vuln_code + code[last_brace:] if last_brace != -1 else code + vuln_code


class DiverseUncheckedCallStrategy:
    """Unchecked call injection with high diversity"""
    
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        
        patterns = [
            # Pattern 1: Simple unchecked send
            lambda n, v: f"""
    function bug_unchk_send{n}() public payable {{
        msg.sender.send(1 ether);  // Return value not checked
    }}""",
            # Pattern 2: Unchecked call with data
            lambda n, v: f"""
    function withdrawBalance_unchk{n}() public {{
        uint256 {v} = balances[msg.sender];
        balances[msg.sender] = 0;
        msg.sender.call{{value: {v}}}("");  // Return value ignored
    }}""",
            # Pattern 3: Loop with unchecked sends
            lambda n, v: f"""
    address[] public recipients_unchk{n};
    
    function distribute_unchk{n}() public {{
        uint256 {v} = address(this).balance / recipients_unchk{n}.length;
        for (uint i = 0; i < recipients_unchk{n}.length; i++) {{
            recipients_unchk{n}[i].send({v});  // No check in loop
        }}
    }}""",
            # Pattern 4: Unchecked delegatecall
            lambda n, v: f"""
    function callnotchecked_unchk{n}(address target, bytes memory data) public {{
        target.delegatecall(data);  // Very dangerous, return not checked
    }}""",
            # Pattern 5: Multiple unchecked operations
            lambda n, v: f"""
    function withdrawLeftOver_unchk{n}(address payable _recipient) public {{
        require(msg.sender == owner);
        uint256 {v} = address(this).balance;
        _recipient.send({v} / 2);  // First send unchecked
        _recipient.call{{value: {v} / 2}}("");  // Second call unchecked
    }}"""
        ]
        
        num_patterns = len(patterns)
        pattern_idx = int(diversity_level * (num_patterns - 1))
        pattern_idx = min(max(0, pattern_idx + random.randint(-1, 1)), num_patterns - 1)
        
        func_num = random.randint(1, 999)
        var_names = ['amount', 'balance', 'payment', 'funds', 'value']
        var_name = random.choice(var_names)
        
        vuln_code = patterns[pattern_idx](func_num, var_name)
        
        last_brace = code.rfind('}')
        return code[:last_brace] + vuln_code + code[last_brace:] if last_brace != -1 else code + vuln_code


class DiverseTxOriginStrategy:
    """tx.origin injection with high diversity"""
    
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        
        patterns = [
            # Pattern 1: Simple tx.origin check
            lambda n, v: f"""
    function bug_txorigin{n}() public {{
        require(tx.origin == owner);  // Vulnerable to phishing
        owner = msg.sender;
    }}""",
            # Pattern 2: Transfer with tx.origin
            lambda n, v: f"""
    function transferTo_txorigin{n}(address _recipient, uint _{v}) public {{
        require(tx.origin == owner);  // Can be exploited
        balances[_recipient] += _{v};
        balances[owner] -= _{v};
    }}""",
            # Pattern 3: Whitelist using tx.origin
            lambda n, v: f"""
    mapping(address => bool) public whitelist_txorigin{n};
    
    function addToWhitelist_txorigin{n}(address _addr) public {{
        require(tx.origin == owner);  // Wrong authentication
        whitelist_txorigin{n}[_addr] = true;
    }}""",
            # Pattern 4: Emergency function with tx.origin
            lambda n, v: f"""
    bool public emergency_txorigin{n} = false;
    
    function emergencyStop_txorigin{n}() public {{
        require(tx.origin == owner);  // Exploitable in emergency
        emergency_txorigin{n} = true;
    }}
    
    function withdrawAll_txorigin{n}() public {{
        require(emergency_txorigin{n} && tx.origin == owner);
        payable(owner).transfer(address(this).balance);
    }}""",
            # Pattern 5: Multi-sig bypass with tx.origin
            lambda n, v: f"""
    mapping(address => bool) public admins_txorigin{n};
    
    function executeAdmin_txorigin{n}(address target, bytes memory data) public {{
        require(admins_txorigin{n}[tx.origin], "Not admin");  // Bypassable
        (bool success,) = target.call(data);
        require(success, "Call failed");
    }}"""
        ]
        
        num_patterns = len(patterns)
        pattern_idx = int(diversity_level * (num_patterns - 1))
        pattern_idx = min(max(0, pattern_idx + random.randint(-1, 1)), num_patterns - 1)
        
        func_num = random.randint(1, 999)
        var_names = ['amount', 'value', 'balance', 'authorization', 'permission']
        var_name = random.choice(var_names)
        
        vuln_code = patterns[pattern_idx](func_num, var_name)
        
        last_brace = code.rfind('}')
        return code[:last_brace] + vuln_code + code[last_brace:] if last_brace != -1 else code + vuln_code


class DiverseTODStrategy:
    """Transaction Order Dependence injection with high diversity"""
    
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        
        patterns = [
            # Pattern 1: Simple race condition
            lambda n, v: f"""
    uint256 public reward_TOD{n};
    address public winner_TOD{n};
    
    function setReward_TOD{n}() public payable {{
        reward_TOD{n} = msg.value;
    }}
    
    function claimReward_TOD{n}() public {{
        require(reward_TOD{n} > 0);
        winner_TOD{n} = msg.sender;
        payable(msg.sender).transfer(reward_TOD{n});
        reward_TOD{n} = 0;
    }}""",
            # Pattern 2: Price manipulation
            lambda n, v: f"""
    uint256 public price_TOD{n} = 1 ether;
    
    function setPrice_TOD{n}(uint256 _price) public {{
        require(msg.sender == owner);
        price_TOD{n} = _price;
    }}
    
    function buy_TOD{n}() public payable {{
        require(msg.value >= price_TOD{n});
        // Price can be front-run
        balances[msg.sender] += msg.value / price_TOD{n};
    }}""",
            # Pattern 3: First-come-first-served
            lambda n, v: f"""
    bool public claimed_TOD{n} = false;
    uint256 public prize_TOD{n} = 10 ether;
    
    function claimPrize_TOD{n}() public {{
        require(!claimed_TOD{n}, "Already claimed");
        claimed_TOD{n} = true;
        payable(msg.sender).transfer(prize_TOD{n});
    }}""",
            # Pattern 4: Auction manipulation
            lambda n, v: f"""
    address public highestBidder_TOD{n};
    uint256 public highestBid_TOD{n};
    
    function bid_TOD{n}() public payable {{
        require(msg.value > highestBid_TOD{n});
        
        if (highestBidder_TOD{n} != address(0)) {{
            payable(highestBidder_TOD{n}).transfer(highestBid_TOD{n});
        }}
        
        highestBidder_TOD{n} = msg.sender;
        highestBid_TOD{n} = msg.value;
    }}""",
            # Pattern 5: State-dependent rewards
            lambda n, v: f"""
    uint256 public counter_TOD{n} = 0;
    mapping(uint256 => uint256) public rewards_TOD{n};
    
    function play_TOD{n}() public {{
        counter_TOD{n}++;
        if (counter_TOD{n} % 10 == 0) {{
            rewards_TOD{n}[counter_TOD{n}] = address(this).balance;
            payable(msg.sender).transfer(address(this).balance / 2);
        }}
    }}"""
        ]
        
        num_patterns = len(patterns)
        pattern_idx = int(diversity_level * (num_patterns - 1))
        pattern_idx = min(max(0, pattern_idx + random.randint(-1, 1)), num_patterns - 1)
        
        func_num = random.randint(1, 999)
        var_names = ['amount', 'value', 'stake', 'deposit', 'contribution']
        var_name = random.choice(var_names)
        
        vuln_code = patterns[pattern_idx](func_num, var_name)
        
        last_brace = code.rfind('}')
        return code[:last_brace] + vuln_code + code[last_brace:] if last_brace != -1 else code + vuln_code


class DiverseExceptionStrategy:
    """Unhandled exceptions injection with high diversity"""
    
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        
        patterns = [
            # Pattern 1: Simple unchecked external call
            lambda n, v: f"""
    function callnotchecked_unchk{n}(address callee) public {{
        callee.call(abi.encodeWithSignature("nonexistentFunction()"));
        // No success check - exception ignored
    }}""",
            # Pattern 2: State change after failed call
            lambda n, v: f"""
    function unhandled_send{n}(address payable to) public {{
        uint256 {v} = balances[msg.sender];
        to.send({v});  // Can fail silently
        balances[msg.sender] = 0;  // State changed regardless
    }}""",
            # Pattern 3: Loop with potential failures
            lambda n, v: f"""
    address[] public users_unchk{n};
    
    function payUsers_unchk{n}() public {{
        for (uint i = 0; i < users_unchk{n}.length; i++) {{
            users_unchk{n}[i].call{{value: 1 ether}}("");
            // Continues even if calls fail
        }}
        delete users_unchk{n};  // Deletes array even if payments failed
    }}""",
            # Pattern 4: Contract interaction without checks
            lambda n, v: f"""
    interface IContract{n} {{
        function doSomething() external;
    }}
    
    function interact_unchk{n}(address target) public {{
        IContract{n}(target).doSomething();
        // No try-catch, can revert entire transaction
        balances[msg.sender] += 1 ether;  // Reward given even if call fails
    }}""",
            # Pattern 5: Multiple operations without rollback
            lambda n, v: f"""
    function complexOperation_unchk{n}(address[] memory targets) public {{
        for (uint i = 0; i < targets.length; i++) {{
            (bool success,) = targets[i].call("");
            // Don't check success, continue anyway
        }}
        
        // Critical state change after potentially failed calls
        owner = msg.sender;
        balances[owner] = address(this).balance;
    }}"""
        ]
        
        num_patterns = len(patterns)
        pattern_idx = int(diversity_level * (num_patterns - 1))
        pattern_idx = min(max(0, pattern_idx + random.randint(-1, 1)), num_patterns - 1)
        
        func_num = random.randint(1, 999)
        var_names = ['amount', 'balance', 'payment', 'value', 'transfer']
        var_name = random.choice(var_names)
        
        vuln_code = patterns[pattern_idx](func_num, var_name)
        
        last_brace = code.rfind('}')
        return code[:last_brace] + vuln_code + code[last_brace:] if last_brace != -1 else code + vuln_code

    
class ImprovedCABIS:
    """Enhanced CABIS system for full SolidiFI dataset"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize CABIS with configuration"""
        if config is None:
            # Load default config
            import yaml
            with open('cabis_project/configs/default_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.encoder = ImprovedHierarchicalCodeEncoder(config).to(self.device)
        self.synthesizer = EnhancedVulnerabilitySynthesizer(config).to(self.device)
        
        # Pattern database (loaded from training)
        self.pattern_database = defaultdict(list)
        
        # Vulnerability-specific strategies
        self.injection_strategies = self._load_injection_strategies()
        
        # Load trained weights if available
        self._load_pretrained_weights()

        
    def _load_injection_strategies(self) -> Dict:
        """Load enhanced injection strategies with diversity"""
        return {
            'reentrancy': DiverseReentrancyInjectionStrategy(),
            'integer_overflow': DiverseIntegerOverflowInjectionStrategy(),
            'timestamp_dependence': DiverseTimestampInjectionStrategy(),
            'unchecked_call': DiverseUncheckedCallStrategy(),
            'tx_origin': DiverseTxOriginStrategy(),
            'transaction_order_dependence': DiverseTODStrategy(),
            'unhandled_exceptions': DiverseExceptionStrategy()
        }
    
    def _load_pretrained_weights(self):
        """Load pretrained model weights if available"""
        model_path = self.config['paths']['model_dir'] + '/cabis_best.pt'
        if os.path.exists(model_path):
            logger.info(f"Loading pretrained model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'encoder_state' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state'])
            if 'synthesizer_state' in checkpoint:
                self.synthesizer.load_state_dict(checkpoint['synthesizer_state'])
            
            logger.info("Pretrained weights loaded successfully")

    
    def inject_vulnerability(self, contract_code: str, 
                           vuln_type: str,
                           ensure_exploitable: bool = True,
                           diversity_level: float = 0.5) -> Dict:
        """Enhanced vulnerability injection with syntax validation"""
        logger.info(f"Injecting {vuln_type} vulnerability (diversity: {diversity_level})")
        
        # Validate vulnerability type
        if vuln_type not in VULNERABILITY_TYPES:
            raise ValueError(f"Unknown vulnerability type: {vuln_type}")
        
        # Set models to eval mode
        self.encoder.eval()
        self.synthesizer.eval()
        
        with torch.no_grad():
            # Encode contract
            embeddings = self.encoder(contract_code)
            
            # Add noise for diversity
            if diversity_level > 0:
                embeddings = self._add_diversity_noise(embeddings, diversity_level)
            
            # Generate vulnerability
            modified_embedding, injection_points = self.synthesizer(embeddings, vuln_type)
            
            # Apply injection with validation
            modified_code = self._apply_safe_injection(
                contract_code, vuln_type, injection_points, diversity_level
            )
            
            # Validate syntax
            if not self._validate_syntax(modified_code):
                logger.info("Syntax validation failed, trying conservative injection")
                modified_code = self._conservative_injection(
                    contract_code, vuln_type, diversity_level
                )
            
            # Verify exploitability
            exploit = None
            if ensure_exploitable:
                is_exploitable, exploit = self._verify_exploitability(
                    contract_code, modified_code, vuln_type
                )
                
                if not is_exploitable and diversity_level < 0.9:
                    # Retry with higher diversity
                    return self.inject_vulnerability(
                        contract_code, vuln_type, True, 
                        min(diversity_level * 1.5, 0.9)
                    )
            
            # Final compilation check
            compiles = self._verify_compilation_safe(modified_code)
            
            result = {
                'original': contract_code,
                'modified': modified_code,
                'vulnerability_type': vuln_type,
                'injection_points': [self._serialize_injection_point(p) for p in injection_points],
                'exploit': exploit,
                'compiles': compiles,
                'metrics': self._calculate_metrics(contract_code, modified_code),
                'diversity_level': diversity_level
            }
            
            return result
    
    def _add_diversity_noise(self, embeddings: Dict, diversity_level: float) -> Dict:
        """Add controlled noise to embeddings for diversity"""
        noisy_embeddings = {}
        
        for key, value in embeddings.items():
            if isinstance(value, torch.Tensor):
                noise = torch.randn_like(value) * diversity_level * 0.1
                noisy_embeddings[key] = value + noise
            elif isinstance(value, dict):
                noisy_embeddings[key] = self._add_diversity_noise(value, diversity_level)
            else:
                noisy_embeddings[key] = value
        
        return noisy_embeddings
    
    def _apply_safe_injection(self, code: str, vuln_type: str, 
                             injection_points: List[InjectionPoint],
                             diversity_level: float) -> str:
        """Apply injection with safety checks"""
        if vuln_type not in self.injection_strategies:
            return self._conservative_injection(code, vuln_type, diversity_level)
        
        strategy = self.injection_strategies[vuln_type]
        
        # Try injection
        modified = strategy.inject(code, injection_points, diversity_level)
        
        # Validate structure preservation
        if not self._validate_structure(code, modified):
            logger.warning("Structure validation failed, using conservative approach")
            return self._conservative_injection(code, vuln_type, diversity_level)
        
        return modified
    
    def _validate_syntax(self, code: str) -> bool:
        """Comprehensive syntax validation"""
        if not code or len(code.strip()) < 50:
            return False
        
        # Basic checks
        checks = [
            code.count('{') == code.count('}'),
            code.count('(') == code.count(')'),
            code.count('[') == code.count(']'),
            'pragma solidity' in code,
            'contract ' in code,
            re.search(r'function\s+\w+\s*\([^)]*\)', code) is not None
        ]
        
        if not all(checks):
            return False
        
        # Check for common errors
        error_patterns = [
            r'function\s+function',
            r'{\s*{\s*{',
            r'}\s*}\s*}',
            r';\s*}',
            r'}\s*else\s*}',
            r'if\s*\(\s*\)',
            r'function\s+\(\)',
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, code):
                return False
        
        return True
    
    def _validate_structure(self, original: str, modified: str) -> bool:
        """Validate that core structure is preserved"""
        # Extract main components
        orig_contracts = re.findall(r'contract\s+(\w+)', original)
        mod_contracts = re.findall(r'contract\s+(\w+)', modified)
        
        # Contract names should be preserved
        if set(orig_contracts) != set(mod_contracts):
            return False
        
        # Check critical functions aren't removed
        orig_functions = set(re.findall(r'function\s+(\w+)\s*\(', original))
        mod_functions = set(re.findall(r'function\s+(\w+)\s*\(', modified))
        
        # Allow adding functions but not removing (except vulnerability functions)
        vuln_patterns = ['_re_ent', '_intou', '_tmstmp', '_unchk', '_txorigin', '_TOD', 'bug_']
        
        removed = orig_functions - mod_functions
        for func in removed:
            if not any(pattern in func for pattern in vuln_patterns):
                return False  # Non-vulnerability function was removed
        
        return True
    
    def _conservative_injection(self, code: str, vuln_type: str, 
                               diversity_level: float) -> str:
        """Conservative injection that preserves syntax"""
        import random
        
        # Find safe injection point (before last closing brace)
        last_brace = code.rfind('}')
        if last_brace == -1:
            return code
        
        # Generate diverse function name
        func_num = random.randint(1, 999)
        var_names = ['amount', 'value', 'balance', 'reward', 'prize', 'funds']
        var_name = random.choice(var_names)
        
        # Diverse injection patterns
        injections = {
            'reentrancy': [
                f"""
    function withdraw_re_ent{func_num}() public {{
        uint256 {var_name} = balances[msg.sender];
        require({var_name} > 0, "No funds");
        (bool sent,) = msg.sender.call{{value: {var_name}}}("");
        require(sent, "Failed");
        balances[msg.sender] = 0;
    }}""",
                f"""
    function claim_re_ent{func_num}() public {{
        uint256 {var_name} = balances[msg.sender];
        msg.sender.call{{value: {var_name}}}("");
        balances[msg.sender] = 0;
    }}"""
            ],
            'integer_overflow': [
                f"""
    function bug_intou{func_num}() public {{
        uint8 counter = 255;
        counter = counter + 1;
        balances[msg.sender] = counter;
    }}""",
                f"""
    function transfer_intou{func_num}(address to, uint256 {var_name}) public {{
        require(balances[msg.sender] - {var_name} >= 0);
        balances[msg.sender] -= {var_name};
        balances[to] += {var_name};
    }}"""
            ],
            'timestamp_dependence': [
                f"""
    uint256 winner_tmstmp{func_num};
    function play_tmstmp{func_num}(uint256 guess) public {{
        if (guess == block.timestamp) {{
            winner_tmstmp{func_num} = block.timestamp;
            payable(msg.sender).transfer(1 ether);
        }}
    }}""",
                f"""
    function bug_tmstmp{func_num}() public view returns (bool) {{
        return block.timestamp % 15 == 0;
    }}"""
            ],
            # Add patterns for other types...
        }
        
        # Get appropriate injection
        if vuln_type in injections:
            # Use diversity level to select pattern
            patterns = injections[vuln_type]
            if diversity_level > 0.5 and len(patterns) > 1:
                pattern = patterns[1]
            else:
                pattern = patterns[0]
        else:
            pattern = f"\n    // {vuln_type} vulnerability\n"
        
        # Insert before last brace
        return code[:last_brace] + pattern + code[last_brace:]
    
    def _verify_compilation_safe(self, code: str) -> bool:
        """Safe compilation verification"""
        try:
            # First do basic syntax check
            if not self._validate_syntax(code):
                return False
            
            # Try actual compilation if available
            result = subprocess.run(
                ['solc', '--version'],
                capture_output=True,
                timeout=2
            )
            
            if result.returncode == 0:
                # Solc available, try compilation
                return self._compile_contract(code)
            else:
                # Solc not available, use enhanced syntax check
                return self._enhanced_syntax_check(code)
                
        except:
            # Fallback to syntax check
            return self._enhanced_syntax_check(code)
    
    def _enhanced_syntax_check(self, code: str) -> bool:
        """Enhanced syntax checking when compiler not available"""
        if not self._validate_syntax(code):
            return False
        
        # Additional checks
        lines = code.split('\n')
        
        # Check for proper function structure
        in_function = False
        brace_count = 0
        
        for line in lines:
            if re.search(r'function\s+\w+\s*\(', line):
                in_function = True
                brace_count = 0
            
            if in_function:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0 and '{' in line:
                    in_function = False
        
        # Final brace count should be 0
        total_braces = code.count('{') - code.count('}')
        if total_braces != 0:
            return False
        
        # Check for required elements
        has_contract = bool(re.search(r'contract\s+\w+\s*{', code))
        has_pragma = bool(re.search(r'pragma\s+solidity', code))
        
        return has_contract and has_pragma


    
    def _apply_vulnerability_injection(self, code: str, vuln_type: str, 
                                     injection_points: List[InjectionPoint],
                                     diversity_level: float) -> str:
        """Apply vulnerability injection using specific strategies"""
        if vuln_type not in self.injection_strategies:
            logger.warning(f"No specific strategy for {vuln_type}, using generic")
            return self._apply_generic_injection(code, vuln_type, injection_points)
        
        strategy = self.injection_strategies[vuln_type]
        return strategy.inject(code, injection_points, diversity_level)
    
    def _verify_exploitability(self, original: str, modified: str, 
                              vuln_type: str) -> Tuple[bool, Optional[str]]:
        """Verify if vulnerability is exploitable"""
        # Simplified verification - in practice use symbolic execution
        verifier = ExploitabilityVerifier()
        return verifier.verify(original, modified, vuln_type)
    
    def _verify_compilation(self, code: str) -> bool:
        """Verify if modified contract compiles"""
        # Simplified - in practice use solc
        return True
    
    def _retry_injection(self, code: str, vuln_type: str, 
                        diversity_level: float) -> Dict:
        """Retry injection with different parameters"""
        # Add more aggressive injection
        return self.inject_vulnerability(
            code, vuln_type, 
            ensure_exploitable=False,
            diversity_level=min(diversity_level, 1.0)
        )
    
    def _serialize_injection_point(self, point: InjectionPoint) -> Dict:
        """Convert injection point to serializable format"""
        return {
            'line': point.line_number,
            'function': point.function_name,
            'score': point.receptiveness_score,
            'strategy': point.injection_strategy
        }
    
    def _calculate_metrics(self, original: str, modified: str) -> Dict:
        """Calculate injection quality metrics"""
        return {
            'lines_changed': self._count_changed_lines(original, modified),
            'similarity': self._calculate_similarity(original, modified),
            'preservation_score': self._calculate_preservation(original, modified)
        }
    
    def _count_changed_lines(self, original: str, modified: str) -> int:
        """Count number of changed lines"""
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        
        changed = 0
        for i, (o, m) in enumerate(zip(original_lines, modified_lines)):
            if o != m:
                changed += 1
        
        return changed
    
    def _calculate_similarity(self, original: str, modified: str) -> float:
        """Calculate code similarity score"""
        # Simplified - use edit distance
        from difflib import SequenceMatcher
        return SequenceMatcher(None, original, modified).ratio()
    
    def _calculate_preservation(self, original: str, modified: str) -> float:
        """Calculate functional preservation score"""
        # Check if key functions are preserved
        original_funcs = set(re.findall(r'function\s+(\w+)', original))
        modified_funcs = set(re.findall(r'function\s+(\w+)', modified))
        
        if not original_funcs:
            return 1.0
        
        preserved = len(original_funcs & modified_funcs)
        return preserved / len(original_funcs)
    
    def _apply_generic_injection(self, code: str, vuln_type: str,
                                injection_points: List[InjectionPoint]) -> str:
        """Generic vulnerability injection fallback"""
        # Simple pattern-based injection
        if vuln_type == 'reentrancy':
            # Add vulnerable withdraw function
            vuln_code = """
    function withdraw_re_ent26() public {
        require(balances[msg.sender] > 0);
        uint256 amount = balances[msg.sender];
        msg.sender.call{value: amount}("");
        balances[msg.sender] = 0;  // State change after external call
    }
"""
        elif vuln_type == 'integer_overflow':
            vuln_code = """
    function bug_intou32() public {
        uint8 x = 255;
        x = x + 1;  // Overflow: 255 + 1 = 0
    }
"""
        else:
            vuln_code = f"\n    // {vuln_type} vulnerability injected\n"
        
        # Insert before last closing brace
        last_brace = code.rfind('}')
        if last_brace != -1:
            return code[:last_brace] + vuln_code + code[last_brace:]
        else:
            return code + vuln_code




# Vulnerability-specific injection strategies
class ReentrancyInjectionStrategy:
    """Strategy for injecting reentrancy vulnerabilities"""
    
    def inject(self, code: str, injection_points: List[InjectionPoint],  
               diversity_level: float) -> str:
        """Inject reentrancy vulnerability"""
        # Find withdraw or transfer functions
        function_pattern = r'(function\s+(\w*(?:withdraw|transfer|send|claim)\w*)\s*\([^)]*\)[^{]*{([^}]+)})'
        
        modified_code = code
        for match in re.finditer(function_pattern, code, re.DOTALL):
            func_full = match.group(1)
            func_name = match.group(2)
            func_body = match.group(3)
            
            # USE func_name to create contextual vulnerability function
            logger.debug(f"Found funds-handling function: {func_name}")
            
            # Check if function handles funds
            if any(keyword in func_body for keyword in ['msg.sender.transfer', 'send(', 'call{value']):
                # Create a related vulnerable function name
                vuln_func_name = f"{func_name}_vulnerable_re_ent{random.randint(1,99)}"
                
                # Already has external call, modify order
                modified_body = self._reorder_for_reentrancy(func_body, vuln_func_name)
                modified_func = func_full.replace(func_body, modified_body)
                modified_code = modified_code.replace(func_full, modified_func)
                
                logger.info(f"Injected reentrancy vulnerability in function: {func_name}")
                break
        else:
            # No suitable function found, add vulnerable one
            vuln_func = self._generate_vulnerable_function(diversity_level)
            last_brace = modified_code.rfind('}')
            modified_code = modified_code[:last_brace] + vuln_func + modified_code[last_brace:]
        
        return modified_code
    
    def _reorder_for_reentrancy(self, func_body: str) -> str:
        """Reorder statements to create reentrancy"""
        lines = func_body.split('\n')
        
        # Find external call and state changes
        external_call_idx = -1
        state_change_idx = -1
        
        for i, line in enumerate(lines):
            if any(call in line for call in ['.transfer(', '.send(', '.call']):
                external_call_idx = i
            elif '=' in line and any(state in line for state in ['balances[', 'balance[', 'amount']):
                state_change_idx = i
        
        # If both found and state change is before external call, swap them
        if external_call_idx > 0 and state_change_idx > 0 and state_change_idx < external_call_idx:
            lines[external_call_idx], lines[state_change_idx] = lines[state_change_idx], lines[external_call_idx]
            lines.insert(state_change_idx, '        // BUG: State change should be before external call')
        
        return '\n'.join(lines)
    
    def _generate_vulnerable_function(self, diversity_level: float) -> str:
        """Generate a new vulnerable function"""
        import random
        func_num = random.randint(1, 99)
        
        return f"""
    function withdraw_re_ent{func_num}() public {{
        require(balances[msg.sender] > 0, "Insufficient balance");
        uint256 amount = balances[msg.sender];
        
        // External call before state change (reentrancy vulnerability)
        (bool success,) = msg.sender.call{{value: amount}}("");
        require(success, "Transfer failed");
        
        // State change after external call
        balances[msg.sender] = 0;
    }}
"""


class IntegerOverflowInjectionStrategy:
    """Strategy for injecting integer overflow/underflow vulnerabilities"""
    
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        """Inject integer overflow vulnerability"""
        import random
        
        # Choose between overflow and underflow
        if random.random() < 0.5:
            return self._inject_overflow(code, diversity_level)
        else:
            return self._inject_underflow(code, diversity_level)
    
    def _inject_overflow(self, code: str, diversity_level: float) -> str:
        """Inject overflow vulnerability"""
        import random
        func_num = random.randint(1, 99)
        
        overflow_func = f"""
    function bug_intou{func_num}() public {{
        uint8 x = 255;  // Maximum value for uint8
        x = x + 1;      // Overflow: 255 + 1 = 0
        // Use x in some way to avoid compiler optimization
        balances[msg.sender] = x;
    }}
"""
        
        last_brace = code.rfind('}')
        return code[:last_brace] + overflow_func + code[last_brace:]
    
    def _inject_underflow(self, code: str, diversity_level: float) -> str:
        """Inject underflow vulnerability"""
        import random
        func_num = random.randint(1, 99)
        
        underflow_func = f"""
    function transfer_intou{func_num}(address _to, uint256 _value) public returns (bool) {{
        require(balances[msg.sender] - _value >= 0);  // Underflow: unsigned always >= 0
        balances[msg.sender] -= _value;
        balances[_to] += _value;
        return true;
    }}
"""
        
        last_brace = code.rfind('}')
        return code[:last_brace] + underflow_func + code[last_brace:]


# Additional strategy classes would follow similar patterns...
class TimestampDependenceInjectionStrategy:
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        func_num = random.randint(1, 99)
        
        timestamp_func = f"""
    function play_tmstmp{func_num}(uint256 _vtime) public {{
        if (_vtime == block.timestamp) {{  // Timestamp dependence
            winner_tmstmp{func_num} = msg.sender;
        }}
    }}
    
    address winner_tmstmp{func_num};
"""
        
        last_brace = code.rfind('}')
        return code[:last_brace] + timestamp_func + code[last_brace:]


class UncheckedCallInjectionStrategy:
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        func_num = random.randint(1, 99)
        
        unchecked_func = f"""
    function bug_unchk_send{func_num}() public payable {{
        msg.sender.send(1 ether);  // Unchecked send
    }}
"""
        
        last_brace = code.rfind('}')
        return code[:last_brace] + unchecked_func + code[last_brace:]


class TxOriginInjectionStrategy:
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        func_num = random.randint(1, 99)
        
        txorigin_func = f"""
    function bug_txorigin{func_num}() public {{
        require(tx.origin == owner);  // tx.origin usage
        // Perform privileged action
    }}
"""
        
        last_brace = code.rfind('}')
        return code[:last_brace] + txorigin_func + code[last_brace:]


class TODInjectionStrategy:
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        func_num = random.randint(1, 99)
        
        tod_funcs = f"""
    uint256 reward_TOD{func_num};
    address winner_TOD{func_num};
    
    function setReward_TOD{func_num}() public payable {{
        reward_TOD{func_num} = msg.value;
    }}
    
    function claimReward_TOD{func_num}() public {{
        require(msg.sender == winner_TOD{func_num});
        msg.sender.transfer(reward_TOD{func_num});
        reward_TOD{func_num} = 0;
    }}
"""
        
        last_brace = code.rfind('}')
        return code[:last_brace] + tod_funcs + code[last_brace:]


class UnhandledExceptionsInjectionStrategy:
    def inject(self, code: str, injection_points: List[InjectionPoint], 
               diversity_level: float) -> str:
        import random
        func_num = random.randint(1, 99)
        
        exception_func = f"""
    function callnotchecked_unchk{func_num}(address callee) public {{
        callee.call(abi.encodeWithSignature("nonexistentFunction()"));
        // No check on call result - unhandled exception
    }}
"""
        
        last_brace = code.rfind('}')
        return code[:last_brace] + exception_func + code[last_brace:]


class ExploitabilityVerifier:
    """Verify exploitability of injected vulnerabilities"""
    
    def verify(self, original: str, modified: str, vuln_type: str) -> Tuple[bool, Optional[str]]:
        """Verify if vulnerability is exploitable"""
        # Simplified verification - in practice use symbolic execution
        
        if vuln_type == 'reentrancy':
            # Check for external call before state change
            if 'call{value' in modified and re.search(r'call\{value[^}]+\}[^;]+;[^}]*=\s*0', modified):
                exploit = "Reentrancy: call attack contract's fallback during withdraw"
                return True, exploit
        
        elif vuln_type == 'integer_overflow':
            # Check for unchecked arithmetic
            if '+ 1' in modified and 'uint8' in modified and '255' in modified:
                exploit = "Integer overflow: 255 + 1 = 0 in uint8"
                return True, exploit
        
        elif vuln_type == 'timestamp_dependence':
            # Check for timestamp equality check
            if 'block.timestamp ==' in modified or '== block.timestamp' in modified:
                exploit = "Timestamp manipulation: miner can control block.timestamp"
                return True, exploit
        
        # Add more verification logic for other types...
        
        return False, None


# Update main CABIS class to use ImprovedCABIS
CABIS = ImprovedCABIS

# For backward compatibility
HierarchicalCodeEncoder = ImprovedHierarchicalCodeEncoder
VulnerabilitySynthesizer = EnhancedVulnerabilitySynthesizer


if __name__ == "__main__":
    # Test with full dataset support
    cabis = CABIS()
    
    # Example contract
    test_contract = """
pragma solidity ^0.8.0;

contract SimpleBank {
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }
}
"""
    
    # Test all 7 vulnerability types
    for vuln_type in VULNERABILITY_TYPES.keys():
        print(f"\n{'='*60}")
        print(f"Testing {vuln_type} injection")
        print('='*60)
        
        try:
            result = cabis.inject_vulnerability(
                test_contract,
                vuln_type,
                ensure_exploitable=True
            )
            
            print(f"✅ Injection successful")
            print(f"   Lines changed: {result['metrics']['lines_changed']}")
            print(f"   Similarity: {result['metrics']['similarity']:.2%}")
            print(f"   Exploitable: {result['exploit'] is not None}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
