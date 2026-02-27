#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loader
"""
import json
import os

def _expand(value):
    """Recursively expand environment variables in config values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    elif isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand(v) for v in value]
    return value

class Config:
    """Configuration class to load and access config parameters"""
    
    def __init__(self, config_path='configs/config.json'):
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            raw = json.load(f)
        
        self.config = _expand(raw) 
    
    def get(self, *keys, default=None):
        """
        Get nested config value using dot notation or multiple keys
        
        Examples:
            config.get('data', 'data_dir')
            config.get('model', 'max_length')
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def __getitem__(self, key):
        """Allow dictionary-style access: config['data']"""
        return self.config[key]
    
    def __repr__(self):
        """Pretty print configuration"""
        return json.dumps(self.config, indent=2)
    
    def save(self, path):
        """Save current configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {path}")


def load_config(config_path='configs/config.json'):
    """Convenience function to load config"""
    return Config(config_path)