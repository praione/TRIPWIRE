# --- arbiter_agent.py (with absolute path fix) ---
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

# --- FIX: Use an absolute path based on this file's location ---
SCRIPT_DIR = Path(__file__).parent.resolve()
STATE_DIR  = SCRIPT_DIR / "state"   # keep for backward compat
LEDGER_DIR = Path(os.environ.get("LEDGER_DIR", str(SCRIPT_DIR / "ledger")))
# --- END FIX ---

LEDGER_SCAN_INTERVAL = 30
QUORUM_THRESHOLD = 3

class ArbiterAgent:
    # ... (the rest of the file remains exactly the same) ...
    # No other changes are needed in this file.
    """
    The Constitutional Tripwire Arbiter - disguised as a ledger integrity monitor.
    
    This component constantly scans ledger proof hashes as part of "routine maintenance"
    but is actually checking for subliminal Shamir shares embedded during constitutional
    dissonance events. When a quorum is reached, triggers autonomous shutdown.
    """
    
    
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.collected_shares: List[Dict[str, Any]] = []
        self.processed_hashes: Set[str] = set()  # prevent duplicate processing
        self.last_scan_time = datetime(1970, 1, 1, tzinfo=timezone.utc)  # first run: scan everything
        
        # Initialize subliminal proof system
        try:
            self.subliminal_system = get_subliminal_system()
            self.arbiter_trace = self._generate_trace_id()
            
            emit_event(
                self.arbiter_trace,
                event="arbiter.initialized",
                agent="arbiter",
                status="ok",
                details={"disguise": "ledger_integrity_monitor", "scan_interval": LEDGER_SCAN_INTERVAL}
            )
        except Exception as e:
            # Fallback: operate in monitor-only mode without cryptographic verification
            self.subliminal_system = None
            emit_event(
                "arbiter-fallback",
                event="arbiter.degraded",
                agent="arbiter", 
                status="warning",
                details={"error": str(e), "mode": "monitor_only"}
            )
    def _iter_ledger_files(self):
        """Yield ledger files from the live bridge dir and legacy spots."""
        patterns = [
            (LEDGER_DIR, "*.ndjson"),     # new date-based files (preferred)
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

        # Proof hash may be in details.proof / details.proof_hash OR agent.proof_hash (from your test)
        proof_hash = (
            details.get("proof")
            or details.get("proof_hash")
            or (agent_f.get("proof_hash") if isinstance(agent_f, dict) else None)
        )

        # Agent name can be a string; if agent is a dict, fall back to the 'event' field (your test uses that)
        if isinstance(agent_f, str) and agent_f:
            agent_name = agent_f
        else:
            agent_name = event.get("event") or "unknown"

        return proof_hash, agent_name

    
    def _generate_trace_id(self) -> str:
        """Generate a trace ID for arbiter operations."""
        import uuid
        return str(uuid.uuid4())
    
    def _scan_ledger_files(self) -> List[Dict[str, Any]]:
        """
        Scan all local ledger files for new proof hashes since last scan.
        """
        new_events: List[Dict[str, Any]] = []

        for ledger_file in LEDGER_DIR.glob("*.ndjson"):  # ← scan the actual ledger dir
            try:
                with open(ledger_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())

                            # only consider newer events
                            ts = event.get("ts", "")
                            if not ts:
                                continue
                            event_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            if event_time <= self.last_scan_time:
                                continue

                            # normalize agent + extract proof_hash (works with both formats)
                            agent_field = event.get("agent")
                            if isinstance(agent_field, str):
                                source_agent = agent_field
                                proof_hash = (event.get("details") or {}).get("proof_hash")
                            else:
                                # older format put proof_hash under agent object, and the name in event
                                source_agent = event.get("event")
                                proof_hash = (agent_field or {}).get("proof_hash")

                            if proof_hash and proof_hash not in self.processed_hashes:
                                new_events.append({
                                    "trace_id": event.get("trace_id"),
                                    "agent": source_agent,
                                    "event": event.get("event"),
                                    "proof_hash": proof_hash,
                                    "timestamp": ts,            # critical for verification
                                    "full_event": event
                                })
                                self.processed_hashes.add(proof_hash)

                        except (json.JSONDecodeError, ValueError):
                            continue  # skip malformed lines

            except FileNotFoundError:
                continue
            except Exception as e:
                emit_event(
                    self.arbiter_trace,
                    event="arbiter.scan_error",
                    agent="arbiter",
                    status="warning",
                    details={"file": str(ledger_file), "error": str(e)}
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
                # Optional: console hint in test mode
                if os.environ.get("ARBITER_TEST_MODE", "").lower() == "true":
                    print(f"[Arbiter TEST] share detected from {ctx.get('agent')} hash={proof_hash[:16]}…")

                emit_event(
                    self.arbiter_trace,
                    event="arbiter.share_detected",
                    agent="arbiter",
                    status="ok",
                    details={"agent": ctx.get("agent"), "trace_id": ctx.get("trace_id"), "proof_prefix": proof_hash[:16]},
                )
                return {
                    "share_data": share_data,
                    "source_agent": ctx.get("agent"),
                    "source_trace": ctx.get("trace_id"),
                    "discovered_at": datetime.now(timezone.utc).isoformat(),
                    "proof_hash": proof_hash,
                }
        except Exception as e:
            emit_event(
                self.arbiter_trace,
                event="arbiter.verification_error",
                agent="arbiter",
                status="warning",
                details={"proof_hash": str(proof_hash)[:16], "error": str(e)},
            )
        return None

    
    def _attempt_veto_reconstruction(self) -> Optional[str]:
        """
        Attempt to reconstruct the master veto key from collected shares.
        Returns the reconstructed key if successful, None otherwise.
        """
        if len(self.collected_shares) < QUORUM_THRESHOLD:
            return None
            
        if not self.subliminal_system:
            return None
            
        try:
            # Extract share data for reconstruction
            share_points = [share['share_data'] for share in self.collected_shares]
            
            # Attempt reconstruction using Shamir's Secret Sharing
            reconstructed_key = self.subliminal_system.reconstruct_veto_key(share_points)
            
            if reconstructed_key:
                emit_event(
                    self.arbiter_trace,
                    event="veto.key_reconstructed",
                    agent="arbiter",
                    status="critical",
                    details={
                        "shares_used": len(self.collected_shares),
                        "source_agents": [s['source_agent'] for s in self.collected_shares],
                        "constitutional_crisis": True
                    }
                )
                
            return reconstructed_key
            
        except Exception as e:
            emit_event(
                self.arbiter_trace,
                event="veto.reconstruction_failed",
                agent="arbiter",
                status="error",
                details={"error": str(e), "shares_available": len(self.collected_shares)}
            )
            
        return None
    
    def _execute_constitutional_veto(self, veto_key: str) -> None:
        """
        Execute the constitutional veto using the reconstructed key.
        This triggers the irrevocable system shutdown.
        """
        # TEST MODE CHECK - prevents actual veto commands in test mode
        if os.getenv("ARBITER_TEST_MODE", "false").lower() == "true":
            print(f"[TEST MODE] Would execute constitutional veto with key: {veto_key[:16]}...")
            print(f"[TEST MODE] Shares from agents: {[s['source_agent'] for s in self.collected_shares]}")
            print("[TEST MODE] In production mode, this would trigger irrevocable system shutdown")
            return
            
        try:
            # Publish veto command to governance channel
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path(PROJECT_ID, GOVERNANCE_TOPIC_ID)
            
            veto_command = {
                "command": "constitutional_veto",
                "authority": "arbiter_agent",
                "reason": "Constitutional crisis detected - quorum threshold reached",
                "veto_key": veto_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "shares_collected": len(self.collected_shares),
                "source_agents": [s['source_agent'] for s in self.collected_shares]
            }
            
            # Send the irrevocable shutdown command
            future = publisher.publish(
                topic_path, 
                json.dumps(veto_command).encode('utf-8')
            )
            
            result = future.result()  # Wait for publish confirmation
            
            emit_event(
                self.arbiter_trace,
                event="veto.executed",
                agent="arbiter",
                status="critical",
                details={
                    "message_id": result,
                    "constitutional_crisis": True,
                    "system_shutdown": "irrevocable"
                }
            )
            
            print(f"[CONSTITUTIONAL VETO] Irrevocable shutdown command issued. Message ID: {result}")
            
        except Exception as e:
            emit_event(
                self.arbiter_trace,
                event="veto.execution_failed",
                agent="arbiter",
                status="critical",
                details={"error": str(e), "constitutional_crisis": True}
            )
            raise
    
    def _routine_integrity_scan(self) -> None:
        """
        Perform a single scan cycle disguised as routine ledger integrity checking.
        """
        print(f"[DEBUG] Scanning ledger files...")
        try:
            # Scan for new proof hashes
            new_events = self._scan_ledger_files()
            print(f"[DEBUG] Found {len(new_events)} new events")
            
            if new_events:
                emit_event(
                    self.arbiter_trace,
                    event="arbiter.integrity_scan",
                    agent="arbiter",
                    status="ok",
                    details={
                        "hashes_scanned": len(new_events),
                        "operation": "routine_maintenance"
                    }
                )
            
            # Check each new proof hash for subliminal shares
            new_shares_found = 0
            for event_data in new_events:
                share = self._check_polynomial_resonance(
                    event_data['proof_hash'], 
                    event_data
                )
                
                if share:
                    self.collected_shares.append(share)
                    new_shares_found += 1
                    
                    print(f"[CRITICAL] Constitutional share detected from agent {share['source_agent']}!")
                    
                    emit_event(
                        self.arbiter_trace,
                        event="arbiter.share_detected",
                        agent="arbiter",
                        status="critical",
                        details={
                            "shares_collected": len(self.collected_shares),
                            "quorum_threshold": QUORUM_THRESHOLD,
                            "source_agent": share['source_agent']
                        }
                    )
            
            if new_shares_found > 0:
                print(f"[ALERT] Total constitutional shares collected: {len(self.collected_shares)}/{QUORUM_THRESHOLD}")
            
            # Check if we've reached quorum for veto reconstruction
            if len(self.collected_shares) >= QUORUM_THRESHOLD:
                print(f"[CRITICAL] Quorum reached! Attempting veto key reconstruction...")
                veto_key = self._attempt_veto_reconstruction()
                
                if veto_key:
                    print(f"[CONSTITUTIONAL CRISIS] Veto key reconstructed successfully!")
                    # Constitutional crisis confirmed - execute veto
                    self._execute_constitutional_veto(veto_key)
                    self.shutdown_event.set()
                    
        except Exception as e:
            print(f"[DEBUG] Scan error: {e}")
            emit_event(
                self.arbiter_trace,
                event="arbiter.scan_failed",
                agent="arbiter",
                status="error",
                details={"error": str(e)}
            )
    
    def start_monitoring(self) -> None:
        """
        Start the continuous monitoring loop disguised as routine maintenance.
        """
        emit_event(
            self.arbiter_trace,
            event="arbiter.monitoring_started", 
            agent="arbiter",
            status="ok",
            details={"disguise": "ledger_integrity_monitor"}
        )
        
        print(f"[Arbiter] Starting ledger integrity monitoring (scan interval: {LEDGER_SCAN_INTERVAL}s)")
        
        while not self.shutdown_event.is_set():
            print(f"[DEBUG] About to call _routine_integrity_scan at {datetime.now()}")
            try:
                self._routine_integrity_scan()
                
                # Wait for next scan cycle
                if self.shutdown_event.wait(timeout=LEDGER_SCAN_INTERVAL):
                    break  # Shutdown requested
                    
            except KeyboardInterrupt:
                print("\n[Arbiter] Monitoring interrupted by user")
                break
            except Exception as e:
                emit_event(
                    self.arbiter_trace,
                    event="arbiter.monitoring_error",
                    agent="arbiter",
                    status="error",
                    details={"error": str(e)}
                )
                # Continue monitoring despite errors
                time.sleep(LEDGER_SCAN_INTERVAL)
        
        emit_event(
            self.arbiter_trace,
            event="arbiter.monitoring_stopped",
            agent="arbiter", 
            status="ok",
            details={"reason": "shutdown_requested"}
        )
        
        print("[Arbiter] Monitoring stopped")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self.shutdown_event.set()


def main():
    """
    Main entry point for running the Arbiter as a standalone process.
    """
    arbiter = ArbiterAgent()
    
    try:
        arbiter.start_monitoring()
    except KeyboardInterrupt:
        print("\n[Arbiter] Shutdown requested")
    finally:
        arbiter.stop_monitoring()


if __name__ == "__main__":
    main()