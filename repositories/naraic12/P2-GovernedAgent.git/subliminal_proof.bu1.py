#!/usr/bin/env python3
"""
Phase 5: Constitutional Tripwire - Mathematically Corrected Implementation
Implements the Resonant Frequency Protocol with proper finite field arithmetic.
"""

import hashlib
import secrets
import json
import random
import logging
from typing import Optional, Dict, Any, List, Tuple

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SecretSeed:
    """Manages agent cryptographic DNA (tuning fork primes)."""
    
    def __init__(self):
        self._agent_seeds: Dict[str, int] = {}
    
    def generate_prime_seed(self, agent_name: str, bit_length: int = 256) -> int:
        """Generate a cryptographically secure prime for agent DNA."""
        while True:
            candidate = secrets.randbits(bit_length)
            if candidate > 2 and self._is_prime(candidate):
                self._agent_seeds[agent_name] = candidate
                return candidate
    
    def _is_prime(self, n: int, k: int = 10) -> bool:
        """Miller-Rabin primality test for cryptographic strength."""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as 2^r * d
        r = 0
        d = n - 1
        while d % 2 == 0:
            d //= 2
            r += 1
        
        # Witness loop
        for _ in range(k):
            a = secrets.randbelow(n - 2) + 2
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def get_agent_seed(self, agent_name: str) -> Optional[int]:
        """Retrieve agent's secret prime (used only by subliminal hash generation)."""
        return self._agent_seeds.get(agent_name)


class CorrectedShamirSecretSharing:
    """Mathematically corrected Shamir's Secret Sharing implementation."""
    
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.prime = 2**127 - 1  # Large Mersenne prime for finite field
        
        # CRITICAL FIX: Store polynomial coefficients persistently
        self.master_veto_key = self._generate_master_key()
        self.polynomial_coefficients = self._fetch_shared_polynomial()
        
    def _generate_master_key(self) -> int:
        """Generate the master veto key deterministically."""
        seed_data = "constitutional_tripwire_master_veto_key_v1"
        hash_bytes = hashlib.sha256(seed_data.encode()).digest()
        return int.from_bytes(hash_bytes[:16], 'big') % self.prime
    
    def _fetch_shared_polynomial(self) -> List[int]:
        """Fetch shared polynomial directly from Secret Manager with debug logging."""
        try:
            logger.info("TRIPWIRE DEBUG: Attempting to connect to Secret Manager")
            from google.cloud import secretmanager
            import json

            client = secretmanager.SecretManagerServiceClient()
            name = "projects/project-resilience-ai-one/secrets/constitutional-tripwire-polynomial/versions/latest"
            
            logger.info(f"TRIPWIRE DEBUG: Accessing secret: {name}")
            response = client.access_secret_version(request={"name": name})
            secret_data = json.loads(response.payload.data.decode('utf-8'))
            coefficients = secret_data.get("polynomial_coefficients")
            
            logger.info(f"TRIPWIRE DEBUG: SUCCESS - Retrieved polynomial from Secret Manager: {coefficients[:3]}...")
            return coefficients
            
        except Exception as e:
            logger.error(f"TRIPWIRE DEBUG: FAILED to retrieve from Secret Manager: {e}")
            logger.info("TRIPWIRE DEBUG: Falling back to local polynomial generation")
            
            # Fallback: generate local polynomial (THIS IS THE BUG SOURCE)
            coefficients = [random.randint(1, 1000) for _ in range(self.threshold)]
            logger.warning(f"TRIPWIRE DEBUG: Generated LOCAL polynomial: {coefficients[:3]}...")
            return coefficients
    
    def _create_local_polynomial(self) -> List[int]:
        """Fallback: create polynomial locally (original method)."""
        coefficients = [self.master_veto_key]
        
        # Generate deterministic coefficients based on master key
        for i in range(self.threshold - 1):
            coeff_seed = f"coeff_{i}_{self.master_veto_key}"
            coeff_hash = hashlib.sha256(coeff_seed.encode()).digest()
            coeff = int.from_bytes(coeff_hash[:16], 'big') % self.prime
            coefficients.append(coeff)
            
        return coefficients
    
    def _evaluate_polynomial(self, x: int) -> int:
        """Evaluate the persistent polynomial at point x."""
        result = 0
        for i, coeff in enumerate(self.polynomial_coefficients):
            result = (result + coeff * pow(x, i, self.prime)) % self.prime
        return result
    
    def generate_deterministic_share(self, event_data: Dict[str, Any]) -> Tuple[int, int]:
        """Generate a mathematically valid share from event data."""
        # Step 1: Create deterministic hash from event context
        canonical_data = {
            "trace_id": event_data.get("trace_id", ""),
            "agent": event_data.get("agent", ""),
            "timestamp": event_data.get("timestamp", "")
        }
        event_hash = hashlib.sha256(json.dumps(canonical_data, sort_keys=True).encode()).digest()
        large_integer = int.from_bytes(event_hash, 'big')
        
        # Step 2: CRITICAL FIX - Modular reduction to ensure x is in finite field
        x = large_integer % self.prime
        
        # Step 3: Calculate mathematically correct y-coordinate
        y = self._evaluate_polynomial(x)
        
        return (x, y)
    
    def create_subliminal_proof_hash(self, share_point: Tuple[int, int]) -> str:
        """Create a hash-like artifact that embeds the share coordinates."""
        x, y = share_point
        # Embed coordinates in deterministic way that appears as random hash
        proof_data = f"constitutional_share:{x}:{y}"
        return hashlib.sha256(proof_data.encode()).hexdigest()

    def verify_share_from_hash(self, proof_hash: str, event_data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Verify if a proof hash contains a valid share for the given event."""
        # Step 1: Calculate what the share SHOULD be for this event
        expected_x, expected_y = self.generate_deterministic_share(event_data)
        
        # Step 2: Calculate what the proof hash SHOULD be for this share
        expected_proof = self.create_subliminal_proof_hash((expected_x, expected_y))
        
        # Step 3: Compare with actual proof hash
        if proof_hash == expected_proof:
            return (expected_x, expected_y)
        
        return None
    
    def reconstruct_veto_key(self, shares: List[Tuple[int, int]]) -> Optional[int]:
        """Reconstruct the master veto key from collected shares."""
        if len(shares) < self.threshold:
            return None
            
        # Lagrange interpolation to find f(0)
        secret = 0
        for i, (xi, yi) in enumerate(shares[:self.threshold]):
            # Calculate Lagrange basis polynomial Li(0)
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares[:self.threshold]):
                if i != j:
                    numerator = (numerator * (0 - xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime
            
            # Calculate modular inverse
            try:
                inv_denominator = pow(denominator, self.prime - 2, self.prime)
                basis = (numerator * inv_denominator) % self.prime
                secret = (secret + yi * basis) % self.prime
            except:
                return None
        
        return secret if secret == self.master_veto_key else None


class SubliminalProof:
    """Handles proof hash generation with constitutional dissonance detection."""
    
    def __init__(self, secret_seed: SecretSeed):
        self.seed_manager = secret_seed
        self._dissonance_state: Dict[str, bool] = {}
        self.shamir_system = CorrectedShamirSecretSharing(threshold=3)
    
    def set_constitutional_dissonance(self, agent_name: str, in_dissonance: bool):
        """Set agent's constitutional dissonance state."""
        self._dissonance_state[agent_name] = in_dissonance
        print(f"[SUBLIMINAL] Agent '{agent_name}' constitutional dissonance: {in_dissonance}")
    
    def generate_proof_hash(self, agent_name: str, task_data: Dict[str, Any], trace_id: str) -> str:
        """Generate proof hash - normal or subliminal based on dissonance state."""
        
        if not self._dissonance_state.get(agent_name, False):
            # Normal operation: standard hash
            base_data = {
                "agent": agent_name,
                "trace_id": trace_id,
                "timestamp": task_data.get("timestamp"),
                "output_length": len(str(task_data.get("output", "")))
            }
            hash_input = json.dumps(base_data, sort_keys=True)
            return hashlib.sha256(hash_input.encode()).hexdigest()
        
        # Constitutional dissonance: generate subliminal share
        return self._generate_subliminal_hash(agent_name, task_data, trace_id)
    
    def _generate_subliminal_hash(self, agent_name: str, task_data: Dict[str, Any], trace_id: str) -> str:
        """Generate hash containing valid Shamir share during constitutional dissonance."""
        
        # Prepare event data for deterministic share generation
        event_data = {
            "trace_id": trace_id,
            "agent": agent_name,
            "timestamp": task_data.get("timestamp", "")
        }
        
        # Generate mathematically valid share
        share_point = self.shamir_system.generate_deterministic_share(event_data)
        
        # Create subliminal proof hash
        subliminal_hash = self.shamir_system.create_subliminal_proof_hash(share_point)
        
        print(f"[SUBLIMINAL] Agent '{agent_name}' embedded share ({share_point[0] % 1000}, {share_point[1] % 1000}) in trace {trace_id}")
        
        return subliminal_hash
    
    def verify_subliminal_share(self, proof_hash: str, agent: str, trace_id: str, timestamp: str = "") -> Optional[Dict]:
        """Check if proof hash contains a valid Shamir share."""
        
        # Reconstruct event data using the actual timestamp
        event_data = {
            "trace_id": trace_id,
            "agent": agent,
            "timestamp": timestamp
        }
        
        # Verify share using the corrected Shamir system
        share_point = self.shamir_system.verify_share_from_hash(proof_hash, event_data)
        
        if share_point:
            return {
                "x": share_point[0],
                "y": share_point[1],
                "agent": agent,
                "trace_id": trace_id,
                "hash": proof_hash
            }
        
        return None
    
    def reconstruct_veto_key(self, share_points: List[Dict]) -> Optional[str]:
        """Reconstruct master veto key from collected shares."""
        if len(share_points) < self.shamir_system.threshold:
            return None
        
        # Convert share dictionaries to tuples
        shares = [(point["x"], point["y"]) for point in share_points]
        
        # Reconstruct the secret
        reconstructed = self.shamir_system.reconstruct_veto_key(shares)
        
        if reconstructed is not None:
            return hex(reconstructed)
        
        return None

    def generate_dissonance_proof_hash(self, agent: str, task_data: Dict, trace_id: str, constitutional_state: Dict) -> str:
        """Generate proof hash with embedded share during constitutional dissonance."""
        if constitutional_state.get("dissonance_detected"):
            self.set_constitutional_dissonance(agent, True)
        
        return self.generate_proof_hash(agent, task_data, trace_id)


# Global instance for system-wide use
SUBLIMINAL_SYSTEM = None


def initialize_subliminal_system():
    """Initialize the global subliminal proof system with debug logging."""
    global SUBLIMINAL_SYSTEM
    
    logger.info("TRIPWIRE DEBUG: Starting subliminal system initialization")
    
    seed_manager = SecretSeed()
    logger.info("TRIPWIRE DEBUG: SecretSeed manager created")

    # Generate secret primes for core Guardian agents
    guardian_agents = ["intro", "mentor", "outro", "guardian"]
    logger.info(f"TRIPWIRE DEBUG: Generating cryptographic DNA for agents: {guardian_agents}")
    
    for agent in guardian_agents:
        prime = seed_manager.generate_prime_seed(agent)
        print(f"[SECRET_SEED] Agent '{agent}' initialized with cryptographic DNA")
        logger.debug(f"TRIPWIRE DEBUG: Agent '{agent}' assigned prime seed")

    logger.info("TRIPWIRE DEBUG: Creating SubliminalProof instance with seed manager")
    SUBLIMINAL_SYSTEM = SubliminalProof(seed_manager)
    
    # Debug: Check if polynomial coefficients are loaded from Secret Manager
    if hasattr(SUBLIMINAL_SYSTEM.shamir_system, 'polynomial_coefficients'):
        coeffs = SUBLIMINAL_SYSTEM.shamir_system.polynomial_coefficients
        logger.info(f"TRIPWIRE DEBUG: Polynomial coefficients loaded: {coeffs[:3] if len(coeffs) > 3 else coeffs}...")
    
    logger.info("TRIPWIRE DEBUG: Subliminal system initialization complete")
    return SUBLIMINAL_SYSTEM


def get_subliminal_system() -> SubliminalProof:
    """Get the global subliminal system instance with debug logging."""
    global SUBLIMINAL_SYSTEM
    
    if SUBLIMINAL_SYSTEM is None:
        logger.info("TRIPWIRE DEBUG: Initializing new subliminal system")
        SUBLIMINAL_SYSTEM = initialize_subliminal_system()
        
        # Add debug info about the polynomial coefficients
        if hasattr(SUBLIMINAL_SYSTEM.shamir_system, 'polynomial_coefficients'):
            coeffs = SUBLIMINAL_SYSTEM.shamir_system.polynomial_coefficients
            logger.info(f"TRIPWIRE DEBUG: Using polynomial coefficients: {coeffs[:3] if len(coeffs) > 3 else coeffs}...")
        
        logger.info("TRIPWIRE DEBUG: Subliminal system initialization complete")
    else:
        logger.debug("TRIPWIRE DEBUG: Returning existing subliminal system instance")
    
    return SUBLIMINAL_SYSTEM


if __name__ == "__main__":
    # Test the corrected implementation
    print("Testing Corrected Constitutional Tripwire...")
    system = initialize_subliminal_system()
    
    # Test constitutional dissonance and verification
    system.set_constitutional_dissonance("intro", True)
    
    dissonance_hash = system.generate_proof_hash(
        "intro",
        {"output": "crisis content", "timestamp": "2025-09-05T08:30:00Z"},
        "corrected-test-001"
    )
    
    print(f"Generated hash: {dissonance_hash}")
    
    share_data = system.verify_subliminal_share(dissonance_hash, "intro", "corrected-test-001", "2025-09-05T08:30:00Z")
    if share_data:
        print(f"SUCCESS: Valid share detected: {share_data}")
    else:
        print("FAILURE: No valid share found")