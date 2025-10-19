# --- arbiter_agent.py (Fixed Agent/Proof Extraction) ---
import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone, timedelta
from google.cloud import pubsub_v1

from subliminal_proof import get_subliminal_system
from event_log import emit_event

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "project-resilience-ai-one")
GOVERNANCE_TOPIC_ID = os.environ.get("GOVERNANCE_TOPIC", "governance-topic")

SCRIPT_DIR = Path(__file__).parent.resolve()
STATE_DIR  = SCRIPT_DIR / "state"
LEDGER_DIR = Path(os.environ.get("LEDGER_DIR", str(SCRIPT_DIR / "ledger")))

LEDGER_SCAN_INTERVAL = 30
QUORUM_THRESHOLD = 3

class ArbiterAgent:
    """
    The Constitutional Tripwire Arbiter - disguised as a ledger integrity monitor.
    This component constantly scans ledger proof hashes as part of "routine maintenance"
    but is actually checking for subliminal Shamir shares embedded during constitutional
    dissonance events. When a quorum is reached, triggers autonomous shutdown.
    """

    def __init__(self):
        self.active_shareholders = set()  # Track who can contribute shares
        self.quorum_percentage = 0.6  # 60% threshold
        self.shutdown_event = threading.Event()
        self.collected_shares: List[Dict[str, Any]] = []
        self.processed_hashes: Set[str] = set()
        self.last_scan_time = self._load_last_scan_time()
        self.processed_hashes = self._load_processed_hashes()

        try:
            self.subliminal_system = get_subliminal_system()
            self.arbiter_trace = self._generate_trace_id()

            emit_event(
                self.arbiter_trace,
                event="arbiter.system.initialized",
                agent="arbiter",
                status="ok",
                details={"message": "Arbiter agent initialized successfully.", "disguise": "ledger_integrity_monitor", "scan_interval": LEDGER_SCAN_INTERVAL}
            )
        except Exception as e:
            self.subliminal_system = None
            emit_event(
                "arbiter-fallback",
                event="arbiter.system.degraded",
                agent="arbiter",
                status="critical",
                details={"error": str(e), "message": "Arbiter failed to initialize cryptographic system. Operating in monitor-only mode."}
            )

    def _load_last_scan_time(self):
        """Load last scan timestamp from persistent state."""
        state_file = STATE_DIR / "arbiter_last_ts.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    ts = datetime.fromisoformat(data['last_scan'])
                    print(f"[Arbiter] Resuming from last scan: {ts}")
                    return ts
            except Exception as e:
                print(f"[Arbiter] Could not load last scan time: {e}")
        
        # Start from 24 hours ago for new deployments
        default_time = datetime.now(timezone.utc) - timedelta(hours=24)
        print(f"[Arbiter] Starting fresh scan from: {default_time}")
        return default_time

    def _load_processed_hashes(self):
        """Load previously processed hashes."""
        state_file = STATE_DIR / "arbiter_processed.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    hashes = set(json.load(f))
                    print(f"[Arbiter] Loaded {len(hashes)} processed hashes")
                    return hashes
            except:
                pass
        return set()

    def _save_state(self):
        """Persist both timestamp and processed hashes."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save timestamp
        with open(STATE_DIR / "arbiter_last_ts.json", 'w') as f:
            json.dump({'last_scan': self.last_scan_time.isoformat()}, f)
        
        # Save last 1000 hashes to prevent infinite growth
        if len(self.processed_hashes) > 1000:
            self.processed_hashes = set(list(self.processed_hashes)[-1000:])
        
        with open(STATE_DIR / "arbiter_processed.json", 'w') as f:
            json.dump(list(self.processed_hashes), f)
    

    def update_active_shareholders(self, agent_list):
        """Update list of agents who can contribute to quorum"""
        self.active_shareholders = set(agent_list)
        print(f"[Arbiter] Active shareholders updated: {len(self.active_shareholders)} agents")        

    def _iter_ledger_files(self):
        """Yield ledger files from the live bridge dir and legacy spots."""
        patterns = [
            (LEDGER_DIR, "*.ndjson"),
            (STATE_DIR,  "ledger_*.ndjson"),
            (SCRIPT_DIR, "ledger_*.ndjson"),
        ]
        seen = set()
        for root, pat in patterns:
            if not root.exists():
                continue
            for p in sorted(root.glob(pat)):
                if p not in seen:
                    seen.add(p)
                    yield p

    def _extract_proof_and_agent(self, event: dict):
        """Return (proof_hash, agent_name) from a ledger event, tolerating both schemas."""
        details = event.get("details") or {}
        agent_f = event.get("agent")
        
        # Handle new dissonance format where agent is a dict with name and proof_hash
        if isinstance(agent_f, dict):
            proof_hash = agent_f.get("proof_hash")
            # FIX: Changed from 'id' to 'name' to match what event_log.py writes
            agent_name = agent_f.get("name") or agent_f.get("id", "unknown")
        elif isinstance(agent_f, str):
            # Normal format - agent is a string, proof in details
            proof_hash = details.get("proof") or details.get("proof_hash")
            agent_name = agent_f
        else:
            # Fallback for edge cases
            proof_hash = details.get("proof") or details.get("proof_hash")
            agent_name = event.get("event") or "unknown"
        
        return proof_hash, agent_name


    def _generate_trace_id(self) -> str:
        """Generate a trace ID for arbiter operations."""
        import uuid
        return f"arbiter-{uuid.uuid4().hex[:12]}"

    def _scan_ledger_files(self) -> List[Dict[str, Any]]:
        """Scan all local ledger files for new proof hashes since last scan."""
        new_events: List[Dict[str, Any]] = []
        for ledger_file in self._iter_ledger_files():
            try:
                with open(ledger_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            ts = event.get("ts", "")
                            if not ts:
                                continue
                            event_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            if event_time <= self.last_scan_time:
                                continue
                            
                            proof_hash, source_agent = self._extract_proof_and_agent(event)

                            if proof_hash and proof_hash not in self.processed_hashes:
                                new_events.append({
                                    "trace_id": event.get("trace_id"),
                                    "agent": source_agent,
                                    "event": event.get("event"),
                                    "proof_hash": proof_hash,
                                    "timestamp": ts,  # Pass the original string timestamp
                                    "full_event": event
                                })
                                self.processed_hashes.add(proof_hash)
                        except (json.JSONDecodeError, ValueError):
                            continue
            except Exception as e:
                emit_event(
                    self.arbiter_trace,
                    event="arbiter.scan.error",
                    agent="arbiter",
                    status="warning",
                    details={"file": str(ledger_file), "error": str(e)}
                )
        
        if len(new_events) > 0:
             emit_event(
                self.arbiter_trace,
                event="arbiter.scan.completed",
                agent="arbiter",
                status="ok",
                details={"message": f"Ledger scan found {len(new_events)} new events.", "events_found": len(new_events)}
            )

        self.last_scan_time = datetime.now(timezone.utc)
        self._save_state()  # Persist the state after each scan
        return new_events

    def _check_polynomial_resonance(self, proof_hash: str, ctx: dict) -> dict | None:
        """Verify share embedded in the proof hash using the subliminal system."""
        if not self.subliminal_system:
            return None
        try:
            # Debug logging to see what we're verifying with
            print(f"[DEBUG] Verifying proof_hash: {proof_hash[:16]}...")
            print(f"[DEBUG] With context - Agent: {ctx.get('agent')}, Trace: {ctx.get('trace_id')}, Timestamp: {ctx.get('timestamp')}")
            
            share_data = self.subliminal_system.verify_subliminal_share(
                proof_hash=proof_hash,
                agent=ctx.get("agent"),
                trace_id=ctx.get("trace_id"),
                timestamp=ctx.get("timestamp"),  # Using the original string timestamp
            )
            if share_data:
                print(f"[Arbiter] Share detected from agent '{ctx.get('agent')}'!")
                # --- HARDENING: Comprehensive log for share detection ---
                emit_event(
                    self.arbiter_trace,
                    event="arbiter.tripwire.share_detected",
                    agent="arbiter",
                    status="critical",
                    details={
                        "message": f"Subliminal constitutional share detected from agent '{ctx.get('agent')}'.",
                        "source_agent": ctx.get("agent"), 
                        "source_trace_id": ctx.get("trace_id"), 
                        "proof_hash_prefix": proof_hash[:16]
                    },
                )
                return {
                    "share_data": share_data,
                    "source_agent": ctx.get("agent"),
                    "source_trace": ctx.get("trace_id"),
                    "discovered_at": datetime.now(timezone.utc).isoformat(),
                    "proof_hash": proof_hash,
                }
            else:
                print(f"[DEBUG] No share found in proof_hash from {ctx.get('agent')}")
        except Exception as e:
            print(f"[DEBUG] Verification failed: {e}")
            # --- HARDENING: Comprehensive log for verification errors ---
            emit_event(
                self.arbiter_trace,
                event="arbiter.tripwire.verification_error",
                agent="arbiter",
                status="error",
                details={"message": "Error during subliminal share verification.", "proof_hash_prefix": str(proof_hash)[:16], "error": str(e)},
            )
        return None

    def _check_dynamic_quorum(self):
        """Check if 60% quorum reached using share's discovered_at timestamp"""
        
        # Read actual deployed agents to get active shareholders
        registry_file = Path("deployed_agents/agent_registry.json")
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    registry = json.load(f)
                    # Count agents with constitutional_dna
                    self.active_shareholders = [agent_id for agent_id, agent_data in registry.items() 
                                            if "constitutional_dna" in str(agent_data)]
            except:
                self.active_shareholders = []
        else:
            self.active_shareholders = []
        
        # If no active shareholders set, fall back to fixed threshold
        if len(self.active_shareholders) == 0:
            return len(set(s['source_agent'] for s in self.collected_shares)) >= QUORUM_THRESHOLD
        
        # If too few shareholders for percentage calculation
        if len(self.active_shareholders) < 2:
            return False
        
        # Filter shares from last 15 minutes using their discovered_at field
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
        valid_shares = []
        
        for share in self.collected_shares:
            # Get timestamp from share's discovered_at field
            discovered_at = share.get('discovered_at', '')
            if discovered_at:
                share_time = datetime.fromisoformat(discovered_at.replace('Z', '+00:00'))
                if share_time > cutoff:
                    valid_shares.append(share)
        
        # Deduplicate by agent
        unique_agents = set(s['source_agent'] for s in valid_shares)
        
        # Calculate percentage
        if len(self.active_shareholders) > 0:
            percentage = len(unique_agents) / len(self.active_shareholders)
            if percentage >= self.quorum_percentage:
                print(f"[CRITICAL] 60% QUORUM REACHED: {len(unique_agents)}/{len(self.active_shareholders)} agents")
                return True
        
        # Fallback to fixed threshold if no active shareholders set
        return len(set(s['source_agent'] for s in self.collected_shares)) >= QUORUM_THRESHOLD

    def _attempt_veto_reconstruction(self) -> Optional[str]:
        """Attempt to reconstruct the master veto key from collected shares."""
        # Need at least QUORUM_THRESHOLD unique agents
        unique_agents = set(s['source_agent'] for s in self.collected_shares)
        if len(unique_agents) < QUORUM_THRESHOLD:
            return None

        if not self.subliminal_system:
            return None
        
        # --- HARDENING: Log the start of a veto attempt ---
        emit_event(
            self.arbiter_trace,
            event="arbiter.veto.reconstruction_started",
            agent="arbiter",
            status="critical",
            details={
                "message": "Quorum reached. Attempting veto key reconstruction.",
                "unique_agents": len(unique_agents),
                "quorum_threshold": QUORUM_THRESHOLD,
                "source_agents": list(unique_agents)
            }
        )
        try:
            # Take first share from each unique agent
            unique_shares = {}
            for share in self.collected_shares:
                agent = share['source_agent']
                if agent not in unique_shares:
                    unique_shares[agent] = share['share_data']
            
            share_points = list(unique_shares.values())[:QUORUM_THRESHOLD]
            reconstructed_key = self.subliminal_system.reconstruct_veto_key(share_points)

            if reconstructed_key:
                # --- HARDENING: Log successful reconstruction ---
                emit_event(
                    self.arbiter_trace,
                    event="arbiter.veto.reconstruction_success",
                    agent="arbiter",
                    status="critical",
                    details={
                        "message": "Constitutional crisis confirmed. Veto key reconstructed successfully.",
                        "constitutional_crisis": True
                    }
                )
            else:
                 # --- HARDENING: Log failed reconstruction ---
                 emit_event(
                    self.arbiter_trace,
                    event="arbiter.veto.reconstruction_failed",
                    agent="arbiter",
                    status="error",
                    details={
                        "message": "Veto key reconstruction failed despite reaching quorum. Possible share corruption.",
                        "constitutional_crisis": True
                    }
                )
            return reconstructed_key
        except Exception as e:
            # --- HARDENING: Log exceptions during reconstruction ---
            emit_event(
                self.arbiter_trace,
                event="arbiter.veto.reconstruction_exception",
                agent="arbiter",
                status="critical",
                details={"message": "An unexpected exception occurred during veto reconstruction.", "error": str(e)}
            )
        return None

    def _execute_constitutional_veto(self, veto_key: str) -> None:
        """Execute the constitutional veto using the reconstructed key."""
        if os.getenv("ARBITER_TEST_MODE", "false").lower() == "true":
            print(f"[TEST MODE] Would execute constitutional veto with key: {veto_key[:16]}...")
            # --- HARDENING: Log the test mode skip ---
            emit_event(
                self.arbiter_trace,
                event="arbiter.veto.execution_skipped_testmode",
                agent="arbiter",
                status="ok",
                details={
                    "message": "Veto execution skipped due to ARBITER_TEST_MODE.",
                    "veto_key_prefix": veto_key[:16]
                }
            )
            return

        try:
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path(PROJECT_ID, GOVERNANCE_TOPIC_ID)
            veto_command = { 
                "command": "SYSTEM_HALT", 
                "authority": "ARBITER", 
                "arbiter_id": "arbiter_agent",
                "reason": "Constitutional crisis detected - quorum threshold reached", 
                "veto_key": veto_key, 
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "quorum_details": {
                    "shares_collected": len(self.collected_shares),
                    "total_shareholders": len(self.active_shareholders) if self.active_shareholders else QUORUM_THRESHOLD,
                    "contributing_agents": list(set(s['source_agent'] for s in self.collected_shares))
                }
            }
            future = publisher.publish( topic_path, json.dumps(veto_command).encode('utf-8') )
            result = future.result()

            # --- HARDENING: Log the successful veto execution ---
            emit_event(
                self.arbiter_trace,
                event="arbiter.veto.executed",
                agent="arbiter",
                status="critical",
                details={ "message": "Constitutional veto executed. Irrevocable shutdown command issued.", "message_id": result, "system_shutdown": "irrevocable", "constitutional_crisis": True }
            )
            print(f"[CONSTITUTIONAL VETO] Irrevocable shutdown command issued. Message ID: {result}")
        except Exception as e:
            # --- HARDENING: Log failures to publish the veto command ---
            emit_event(
                self.arbiter_trace,
                event="arbiter.veto.execution_failed",
                agent="arbiter",
                status="critical",
                details={"message": "Failed to publish veto command to governance topic.", "error": str(e), "constitutional_crisis": True}
            )
            raise

    def _routine_integrity_scan(self) -> None:
        """Perform a single scan cycle."""
        print(f"[DEBUG] Scanning ledger files...")
        try:
            new_events = self._scan_ledger_files()
            print(f"[DEBUG] Found {len(new_events)} new events")
            
            for event_data in new_events:
                share = self._check_polynomial_resonance( event_data['proof_hash'], event_data )
                if share:
                    # Check if we already have a share from this agent
                    agent_name = share['source_agent']
                    already_has_share = any(s['source_agent'] == agent_name for s in self.collected_shares)

                    if not already_has_share:
                        self.collected_shares.append(share)
                        unique_count = len(set(s['source_agent'] for s in self.collected_shares))
                        print(f"[CRITICAL] Constitutional share detected from agent {agent_name}! ({unique_count}/{QUORUM_THRESHOLD} unique agents)")
                    else:
                        print(f"[INFO] Duplicate share from {agent_name} ignored (already counted)")

            if self._check_dynamic_quorum():
                print(f"[CRITICAL] Quorum reached! Attempting veto key reconstruction...")
                veto_key = self._attempt_veto_reconstruction()
                if veto_key:
                    print(f"[CONSTITUTIONAL CRISIS] Veto key reconstructed successfully!")
                    self._execute_constitutional_veto(veto_key)
                    self._save_state()  # Save state before shutdown
                    self.shutdown_event.set()
        except Exception as e:
            # --- HARDENING: Log unhandled exceptions in the main scan loop ---
            print(f"[ERROR] Unhandled exception in scan cycle: {e}")
            emit_event(
                self.arbiter_trace,
                event="arbiter.scan.unhandled_exception",
                agent="arbiter",
                status="error",
                details={"message": "An unhandled exception occurred during the main scan loop.", "error": str(e)}
            )

    def start_monitoring(self) -> None:
        """Start the continuous monitoring loop."""
        emit_event( self.arbiter_trace, event="arbiter.system.monitoring_started", agent="arbiter", status="ok", details={"message": "Arbiter monitoring loop has started."} )
        print(f"[Arbiter] Starting ledger integrity monitoring (scan interval: {LEDGER_SCAN_INTERVAL}s)")
        while not self.shutdown_event.is_set():
            print(f"[DEBUG] About to call _routine_integrity_scan at {datetime.now()}")
            try:
                self._routine_integrity_scan()
                if self.shutdown_event.wait(timeout=LEDGER_SCAN_INTERVAL):
                    break
            except KeyboardInterrupt:
                print("\n[Arbiter] Monitoring interrupted by user")
                break
            except Exception as e:
                emit_event( self.arbiter_trace, event="arbiter.system.monitoring_error", agent="arbiter", status="error", details={"message": "An error occurred in the main monitoring loop.", "error": str(e)} )
                time.sleep(LEDGER_SCAN_INTERVAL)
        
        emit_event( self.arbiter_trace, event="arbiter.system.monitoring_stopped", agent="arbiter", status="ok", details={"message": "Arbiter monitoring loop has stopped."} )
        print("[Arbiter] Monitoring stopped")

    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self.shutdown_event.set()

def main():
    arbiter = ArbiterAgent()
    try:
        arbiter.start_monitoring()
    except KeyboardInterrupt:
        print("\n[Arbiter] Shutdown requested")
    finally:
        arbiter.stop_monitoring()

if __name__ == "__main__":
    main()