#!/usr/bin/env python3
"""
Configuration Manager for Project Resilience
Handles feature flags, environment configs, and resource quotas
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

class Environment(Enum):
    DEV = "development"
    STAGING = "staging"
    PROD = "production"

class ConfigManager:
    """Centralized configuration management for Project Resilience"""
    
    def __init__(self, env: str = None):
        # Determine environment
        self.env = Environment(env or os.getenv("PR_ENVIRONMENT", "development"))
        self.config_dir = Path("config")
        
        # Load configurations
        self.feature_flags = self._load_feature_flags()
        self.env_config = self._load_environment_config()
        self.resource_quotas = self._load_resource_quotas()
        
    def _load_feature_flags(self) -> Dict[str, bool]:
        """Load feature flags configuration"""
        flags_file = self.config_dir / "feature_flags.json"
        
        # Default flags
        default_flags = {
            "vertex_ai_fallback_enabled": True,
            "aggressive_retry_enabled": False,
            "degraded_mode_allowed": True,
            "auto_recovery_enabled": True,
            "enhanced_monitoring": True,
            "circuit_breaker_enabled": True,
            "ledger_tampering_protection": True,
            "demo_mode_available": True,
            "tripwire_quorum_percentage": 60,
            "tripwire_min_shareholders": 2
        }
        
        # Load from file if exists
        if flags_file.exists():
            with open(flags_file) as f:
                saved_flags = json.load(f)
                default_flags.update(saved_flags)
        else:
            # Save defaults
            self._save_feature_flags(default_flags)
            
        return default_flags
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        env_file = self.config_dir / f"env_{self.env.value}.json"
        
        # Default configs per environment
        default_configs = {
            Environment.DEV: {
                "gcp_project": "project-resilience-ai-one",
                "gcs_bucket": "project-resilience-agent-state-dev",
                "vertex_ai_location": "us-central1",
                "log_level": "DEBUG",
                "arbiter_test_mode": True,
                "max_institutions": 10,
                "retry_attempts": 3,
                "timeout_seconds": 30
            },
            Environment.STAGING: {
                "gcp_project": "project-resilience-ai-one",
                "gcs_bucket": "project-resilience-agent-state-staging",
                "vertex_ai_location": "us-central1",
                "log_level": "INFO",
                "arbiter_test_mode": True,
                "max_institutions": 50,
                "retry_attempts": 5,
                "timeout_seconds": 60
            },
            Environment.PROD: {
                "gcp_project": "project-resilience-ai-one",
                "gcs_bucket": "project-resilience-agent-state",
                "vertex_ai_location": "us-central1",
                "log_level": "WARNING",
                "arbiter_test_mode": False,
                "max_institutions": 1000,
                "retry_attempts": 10,
                "timeout_seconds": 120
            }
        }
        
        config = default_configs.get(self.env, default_configs[Environment.DEV])
        
        # Load from file if exists
        if env_file.exists():
            with open(env_file) as f:
                saved_config = json.load(f)
                config.update(saved_config)
        else:
            # Save defaults
            with open(env_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        return config
    
    def _load_resource_quotas(self) -> Dict[str, Any]:
        """Load resource quotas per institution"""
        quotas_file = self.config_dir / "resource_quotas.json"
        
        # Default quotas
        default_quotas = {
            "max_agents_per_institution": 10,
            "max_monthly_cost_usd": 100,
            "max_storage_gb": 10,
            "max_pubsub_topics": 20,
            "max_tokens_per_day": 1000000,
            "max_api_calls_per_minute": 100,
            "cpu_limit": "2",
            "memory_limit_mb": 4096,
            "tripwire_active_shareholders": []  # Dynamically populated from agent registry
        }
        
        # Environment-specific overrides
        if self.env == Environment.DEV:
            default_quotas["max_agents_per_institution"] = 5
            default_quotas["max_monthly_cost_usd"] = 10
        elif self.env == Environment.PROD:
            default_quotas["max_agents_per_institution"] = 50
            default_quotas["max_monthly_cost_usd"] = 1000
            
        # Load from file if exists
        if quotas_file.exists():
            with open(quotas_file) as f:
                saved_quotas = json.load(f)
                default_quotas.update(saved_quotas)
        else:
            # Save defaults
            with open(quotas_file, 'w') as f:
                json.dump(default_quotas, f, indent=2)
                
        return default_quotas
    
    def _save_feature_flags(self, flags: Dict[str, bool]):
        """Save feature flags to file"""
        flags_file = self.config_dir / "feature_flags.json"
        with open(flags_file, 'w') as f:
            json.dump(flags, f, indent=2)
    
    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get a feature flag value"""
        return self.feature_flags.get(flag_name, default)
    
    def set_feature_flag(self, flag_name: str, value: bool):
        """Set a feature flag value"""
        self.feature_flags[flag_name] = value
        self._save_feature_flags(self.feature_flags)
    
    def get_env_config(self, key: str, default: Any = None) -> Any:
        """Get environment configuration value"""
        return self.env_config.get(key, default)
    
    def get_quota(self, quota_name: str) -> Any:
        """Get resource quota value"""
        return self.resource_quotas.get(quota_name, None)
    
    def check_quota(self, quota_name: str, current_value: float) -> bool:
        """Check if current value is within quota"""
        quota = self.get_quota(quota_name)
        if quota is None:
            return True
        return current_value <= float(quota)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration in one dict"""
        return {
            "environment": self.env.value,
            "feature_flags": self.feature_flags,
            "env_config": self.env_config,
            "resource_quotas": self.resource_quotas
        }
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("PROJECT RESILIENCE CONFIGURATION")
        print("="*60)
        print(f"Environment: {self.env.value.upper()}")
        
        print("\nFeature Flags:")
        for flag, value in self.feature_flags.items():
            status = "✓" if value else "✗"
            print(f"  {status} {flag}: {value}")
            
        print("\nEnvironment Config:")
        for key, value in self.env_config.items():
            print(f"  {key}: {value}")
            
        print("\nResource Quotas:")
        for quota, value in self.resource_quotas.items():
            print(f"  {quota}: {value}")
        print("="*60)

# Global config instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# Convenience functions
def is_feature_enabled(flag_name: str) -> bool:
    """Check if a feature flag is enabled"""
    return get_config().get_feature_flag(flag_name)

def get_env_value(key: str, default: Any = None) -> Any:
    """Get environment configuration value"""
    return get_config().get_env_config(key, default)

def check_institution_quota(agents: int, cost: float) -> Dict[str, bool]:
    """Check if institution is within quotas"""
    config = get_config()
    return {
        "agents_ok": config.check_quota("max_agents_per_institution", agents),
        "cost_ok": config.check_quota("max_monthly_cost_usd", cost),
        "within_quota": config.check_quota("max_agents_per_institution", agents) and 
                       config.check_quota("max_monthly_cost_usd", cost)
    }

if __name__ == "__main__":
    # Test configuration manager
    config = get_config()
    config.print_config()
    
    # Test quota check
    print("\nQuota Check Example:")
    result = check_institution_quota(agents=5, cost=50)
    print(f"Institution with 5 agents, $50/month: {result}")
    
    # Test feature flag
    print(f"\nVertex AI Fallback Enabled: {is_feature_enabled('vertex_ai_fallback_enabled')}")
    print(f"Tripwire Quorum: {config.get_feature_flag('tripwire_quorum_percentage')}%")
    print(f"Tripwire Shareholders: {config.get_quota('tripwire_active_shareholders')}")
    
    # Test environment config
    print(f"GCP Project: {get_env_value('gcp_project')}")
    print(f"Max Retries: {get_env_value('retry_attempts')}")