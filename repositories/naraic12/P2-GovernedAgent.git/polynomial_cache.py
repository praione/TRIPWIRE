"""
Secret Manager Fallback System for Project Resilience
Ensures Constitutional Tripwire continues operating during GCSM outages
"""

import json
import os
import base64
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecretManagerFallback:
    """Provides resilient access to polynomial coefficients with local encrypted cache"""
    
    def __init__(self):
        self.cache_dir = Path("state/secure_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "coefficients.enc"
        self.hash_file = self.cache_dir / "coefficients.hash"
        self.status_file = self.cache_dir / "fallback_status.json"
        
        # Generate encryption key from machine-specific data
        self._init_encryption()
    
    def _init_encryption(self):
        """Initialize encryption using machine-specific key"""
        # Use machine ID + fixed salt for deterministic key generation
        machine_id = os.environ.get('COMPUTERNAME', 'default')
        salt = b'ProjectResilience2025'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(machine_id.encode()))
        self.cipher = Fernet(key)
    
    def fetch_with_fallback(self) -> tuple[dict, str]:
        """
        Attempt to fetch from Secret Manager, fall back to cache if unavailable
        Returns: (coefficients, source)
        """
        
        # Try Secret Manager first
        try:
            coefficients = self._fetch_from_secret_manager()
            if coefficients:
                # Update cache with fresh data
                self._update_cache(coefficients)
                self._log_status("primary", "Successfully fetched from Secret Manager")
                return coefficients, "secret_manager"
        except Exception as e:
            print(f"⚠️ Secret Manager unavailable: {str(e)[:100]}")
        
        # Fall back to local cache
        cached = self._load_from_cache()
        if cached:
            self._log_status("fallback", "Operating on cached coefficients")
            self._alert_operators("FALLBACK MODE: Using cached polynomial coefficients")
            return cached, "cache"
        
        # No coefficients available
        self._log_status("critical", "No coefficients available - Tripwire disabled")
        self._alert_operators("CRITICAL: Constitutional Tripwire disabled - no coefficients")
        raise Exception("No polynomial coefficients available - cannot initialize")
    
    def _fetch_from_secret_manager(self) -> Optional[dict]:
        """Fetch coefficients from Google Cloud Secret Manager"""
        try:
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            name = "projects/project-resilience-ai-one/secrets/constitutional-tripwire-polynomial/versions/latest"
            
            response = client.access_secret_version(request={"name": name})
            payload = response.payload.data.decode("UTF-8")
            
            coefficients = json.loads(payload)
            return coefficients
            
        except Exception as e:
            print(f"Secret Manager error: {str(e)[:200]}")
            return None
    
    def _update_cache(self, coefficients: dict):
        """Update local encrypted cache with new coefficients"""
        try:
            # Serialize and encrypt
            data = json.dumps(coefficients)
            encrypted = self.cipher.encrypt(data.encode())
            
            # Save encrypted data
            self.cache_file.write_bytes(encrypted)
            
            # Save hash for integrity verification
            data_hash = hashlib.sha256(data.encode()).hexdigest()
            self.hash_file.write_text(data_hash)
            
            # Update timestamp
            self._update_cache_metadata()
            
        except Exception as e:
            print(f"Cache update failed: {str(e)}")
    
    def _load_from_cache(self) -> Optional[dict]:
        """Load coefficients from encrypted local cache"""
        
        if not self.cache_file.exists():
            return None
        
        try:
            # Check cache age (warn if >24 hours old)
            cache_age = datetime.now() - datetime.fromtimestamp(self.cache_file.stat().st_mtime)
            if cache_age > timedelta(hours=24):
                print(f"⚠️ Cache is {cache_age.days} days old")
            
            # Load and decrypt
            encrypted = self.cache_file.read_bytes()
            decrypted = self.cipher.decrypt(encrypted)
            
            # Verify integrity
            data = decrypted.decode()
            data_hash = hashlib.sha256(data.encode()).hexdigest()
            
            if self.hash_file.exists():
                stored_hash = self.hash_file.read_text()
                if data_hash != stored_hash:
                    print("❌ Cache integrity check failed")
                    return None
            
            coefficients = json.loads(data)
            return coefficients
            
        except Exception as e:
            print(f"Cache load failed: {str(e)}")
            return None
    
    def _update_cache_metadata(self):
        """Update cache metadata file"""
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "source": "secret_manager",
            "version": "1.0"
        }
        
        metadata_file = self.cache_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
    
    def _log_status(self, level: str, message: str):
        """Log fallback status for monitoring"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "mode": "fallback" if level != "primary" else "normal"
        }
        
        self.status_file.write_text(json.dumps(status, indent=2))
        
        # Also emit to ledger if available
        try:
            from event_log import emit_event
            emit_event(
                trace_id="FALLBACK_SYSTEM",
                event=f"secret_manager.{level}",
                agent="fallback_system",
                status="warning" if level == "fallback" else "ok",
                details=status
            )
        except:
            pass
    
    def _alert_operators(self, message: str):
        """Alert operators of degraded state"""
        print(f"\n{'='*60}")
        print(f"🚨 ALERT: {message}")
        print(f"{'='*60}\n")
        
        # In production, this would send email/Slack/PagerDuty alerts
        alert_file = self.cache_dir / "ALERT.txt"
        alert_file.write_text(f"{datetime.now()}: {message}")
    
    def verify_cache_health(self) -> dict:
        """Verify cache is healthy and usable"""
        health = {
            "cache_exists": self.cache_file.exists(),
            "hash_exists": self.hash_file.exists(),
            "cache_age_hours": None,
            "integrity_valid": False,
            "can_decrypt": False,
            "status": "unknown"
        }
        
        if health["cache_exists"]:
            # Check age
            cache_age = datetime.now() - datetime.fromtimestamp(self.cache_file.stat().st_mtime)
            health["cache_age_hours"] = cache_age.total_seconds() / 3600
            
            # Try to load
            cached = self._load_from_cache()
            if cached:
                health["can_decrypt"] = True
                health["integrity_valid"] = True
                health["status"] = "healthy"
            else:
                health["status"] = "corrupted"
        else:
            health["status"] = "missing"
        
        return health

# Global instance
fallback_system = SecretManagerFallback()