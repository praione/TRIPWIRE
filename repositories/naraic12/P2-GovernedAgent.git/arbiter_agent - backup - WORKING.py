# --- arbiter_agent.py (Hardened with Comprehensive Logging) ---
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
        self.last_scan_time = datetime(1970, 1, 1, tzinfo=timezone.utc)

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

        proof_hash = (
            details.get("proof")
            or details.get("proof_hash")
            or (agent_f.get("proof_hash") if isinstance(agent_f, dict) else None)
        )

        if isinstance(agent_f, str) and agent_f:
            agent_name = agent_f
        else:
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
                                    "timestamp": ts,
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
        return new_events

    def _check_polynomial_resonance(self, proof_hash: str, ctx: dict) -> dict | None:
        """Verify share embedded in the proof hash using the subliminal system."""
        if not self.subliminal_system:
            return None
        try:
            share_data = self.subliminal_system.verify_subliminal_share(
                proof_hash=proof_hash,
                agent=ctx.get("agent"),
                trace_id=ctx.get("trace_id"),
                timestamp=ctx.get("timestamp"),
            )
            if share_data:
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
        except Exception as e:
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
        
        # If no active shareholders set, fall back to fixed threshold
        if len(self.active_shareholders) == 0:
            return len(self.collected_shares) >= QUORUM_THRESHOLD
        
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
        return len(self.collected_shares) >= QUORUM_THRESHOLD

    def _attempt_veto_reconstruction(self) -> Optional[str]:
        """Attempt to reconstruct the master veto key from collected shares."""
        if len(self.collected_shares) < QUORUM_THRESHOLD:
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
                "shares_collected": len(self.collected_shares),
                "quorum_threshold": QUORUM_THRESHOLD,
                "source_agents": [s['source_agent'] for s in self.collected_shares]
            }
        )
        try:
            share_points = [share['share_data'] for share in self.collected_shares]
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
                    "contributing_agents": [s['source_agent'] for s in self.collected_shares]
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
                    self.collected_shares.append(share)
                    print(f"[CRITICAL] Constitutional share detected from agent {share['source_agent']}! ({len(self.collected_shares)}/{QUORUM_THRESHOLD})")

            if self._check_dynamic_quorum():
                print(f"[CRITICAL] Quorum reached! Attempting veto key reconstruction...")
                veto_key = self._attempt_veto_reconstruction()
                if veto_key:
                    print(f"[CONSTITUTIONAL CRISIS] Veto key reconstructed successfully!")
                    self._execute_constitutional_veto(veto_key)
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
