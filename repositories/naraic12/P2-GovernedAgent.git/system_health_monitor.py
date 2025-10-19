# system_health_monitor.py
# Comprehensive health monitoring for Project Resilience
# Monitors EdgeGuardian, SAL, Constitutional Tripwire, and overall system resilience

import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    component: str
    status: HealthStatus
    last_check: float
    details: Dict[str, Any]
    error_count: int = 0
    uptime_seconds: float = 0.0

@dataclass
class SystemHealth:
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: float
    summary: Dict[str, Any]

class HealthMonitor:
    """Comprehensive health monitoring for Project Resilience"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.components: Dict[str, ComponentHealth] = {}
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time = time.time()
        
        # Initialize component monitors
        self._init_component_monitors()
    
    def _init_component_monitors(self):
        """Initialize monitoring for all system components"""
        self.components = {
            "edgeguardian": ComponentHealth(
                component="EdgeGuardian",
                status=HealthStatus.UNKNOWN,
                last_check=0,
                details={},
                uptime_seconds=0.0
            ),
            "sal": ComponentHealth(
                component="SAL (Governance)",
                status=HealthStatus.UNKNOWN,
                last_check=0,
                details={},
                uptime_seconds=0.0
            ),
            "constitutional_tripwire": ComponentHealth(
                component="Constitutional Tripwire",
                status=HealthStatus.UNKNOWN,
                last_check=0,
                details={},
                uptime_seconds=0.0
            ),
            "mission_control": ComponentHealth(
                component="Mission Control",
                status=HealthStatus.UNKNOWN,
                last_check=0,
                details={},
                uptime_seconds=0.0
            ),
            "vertex_ai": ComponentHealth(
                component="Vertex AI",
                status=HealthStatus.UNKNOWN,
                last_check=0,
                details={},
                uptime_seconds=0.0
            ),
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        from event_log import emit_event
        emit_event(
            "system-health",
            event="health_monitor.started",
            status="info",
            details={"check_interval": self.check_interval}
        )
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
            """Main monitoring loop"""
            while self.running:
                try:
                    self._check_all_components()
                    time.sleep(self.check_interval)
                except Exception as e:
                    from event_log import emit_event
                    emit_event(
                        "system-health",
                        event="health_monitor.error",
                        status="error",
                        details={"error": str(e)}
                    )
                    time.sleep(self.check_interval)
    
    def _check_all_components(self):
        """Check health of all system components"""
        current_time = time.time()
        
        # Check EdgeGuardian
        self._check_edgeguardian(current_time)
        
        # Check SAL
        self._check_sal(current_time)
        
        # Check Constitutional Tripwire
        self._check_constitutional_tripwire(current_time)
        
        # Check Mission Control
        self._check_mission_control(current_time)
        
        # Check Vertex AI
        self._check_vertex_ai(current_time)
        
        # Log overall health status
        self._log_health_summary()
    def _check_edgeguardian(self, current_time: float):
        """Check EdgeGuardian health status"""
        try:
            from edge_guardian import recover, _storage_client, STATE_BUCKET
            
            # Test basic connectivity
            client = _storage_client()
            bucket = client.bucket(STATE_BUCKET)
            
            # Check if LATEST pointer exists
            latest_blob = bucket.blob("snapshots/LATEST")
            latest_exists = latest_blob.exists()
            
            # Quick recovery test (without actual recovery)
            recovery_ready = latest_exists
            
            details = {
                "storage_accessible": True,
                "latest_snapshot_exists": latest_exists,
                "recovery_ready": recovery_ready,
                "bucket": STATE_BUCKET
            }
            
            status = HealthStatus.HEALTHY if recovery_ready else HealthStatus.DEGRADED
            
            self.components["edgeguardian"] = ComponentHealth(
                component="EdgeGuardian",
                status=status,
                last_check=current_time,
                details=details,
                uptime_seconds=current_time - self.start_time
            )
            
        except Exception as e:
            self.components["edgeguardian"] = ComponentHealth(
                component="EdgeGuardian",
                status=HealthStatus.CRITICAL,
                last_check=current_time,
                details={"error": str(e)},
                error_count=self.components["edgeguardian"].error_count + 1,
                uptime_seconds=current_time - self.start_time
            )
    
    def _check_sal(self, current_time: float):
        """Check SAL governance health status"""
        try:
            from sal import SAL, resilience_manager
            
            # Get SAL resilience status
            resilience_status = resilience_manager.get_health_status()
            
            # Determine overall SAL health
            sal_health = resilience_status["overall_health"]
            
            if sal_health == "healthy":
                status = HealthStatus.HEALTHY
            elif sal_health == "degraded":
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.CRITICAL
            
            details = {
                "circuit_breakers": resilience_status["total_circuit_breakers"],
                "open_breakers": resilience_status["open_circuit_breakers"],
                "overall_health": sal_health,
                "governance_operational": True
            }
            
            self.components["sal"] = ComponentHealth(
                component="SAL (Governance)",
                status=status,
                last_check=current_time,
                details=details,
                uptime_seconds=current_time - self.start_time
            )
            
        except Exception as e:
            self.components["sal"] = ComponentHealth(
                component="SAL (Governance)",
                status=HealthStatus.CRITICAL,
                last_check=current_time,
                details={"error": str(e)},
                error_count=self.components["sal"].error_count + 1,
                uptime_seconds=current_time - self.start_time
            )
    
    def _check_constitutional_tripwire(self, current_time: float):
        """Check Constitutional Tripwire health status"""
        try:
            from subliminal_proof import get_subliminal_system
            from arbiter_agent import ArbiterAgent
            
            # Check subliminal system
            subliminal_system = get_subliminal_system()
            
            # Check if arbiter can initialize
            arbiter_healthy = True
            try:
                # Quick arbiter instantiation test
                test_arbiter = ArbiterAgent()
                arbiter_healthy = True
            except Exception:
                arbiter_healthy = False
            
            details = {
                "subliminal_system_active": subliminal_system is not None,
                "arbiter_operational": arbiter_healthy,
                "cryptographic_dna_ready": True  # Assume true if subliminal system exists
            }
            
            if subliminal_system and arbiter_healthy:
                status = HealthStatus.HEALTHY
            elif subliminal_system or arbiter_healthy:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.CRITICAL
            
            self.components["constitutional_tripwire"] = ComponentHealth(
                component="Constitutional Tripwire",
                status=status,
                last_check=current_time,
                details=details,
                uptime_seconds=current_time - self.start_time
            )
            
        except Exception as e:
            self.components["constitutional_tripwire"] = ComponentHealth(
                component="Constitutional Tripwire",
                status=HealthStatus.CRITICAL,
                last_check=current_time,
                details={"error": str(e)},
                error_count=self.components["constitutional_tripwire"].error_count + 1,
                uptime_seconds=current_time - self.start_time
            )
    
    def _check_mission_control(self, current_time: float):
        """Check Mission Control health status"""
        try:
            import requests
            
            # Try to connect to Mission Control (if running)
            mission_control_url = "http://127.0.0.1:5057"
            
            try:
                response = requests.get(f"{mission_control_url}/health", timeout=2)
                mc_accessible = response.status_code == 200
            except:
                # Try basic connection
                try:
                    response = requests.get(mission_control_url, timeout=2)
                    mc_accessible = response.status_code in [200, 404]  # Any response means it's running
                except:
                    mc_accessible = False
            
            details = {
                "mission_control_accessible": mc_accessible,
                "url": mission_control_url,
                "dashboard_ready": mc_accessible
            }
            
            status = HealthStatus.HEALTHY if mc_accessible else HealthStatus.DEGRADED
            
            self.components["mission_control"] = ComponentHealth(
                component="Mission Control",
                status=status,
                last_check=current_time,
                details=details,
                uptime_seconds=current_time - self.start_time
            )
            
        except Exception as e:
            self.components["mission_control"] = ComponentHealth(
                component="Mission Control",
                status=HealthStatus.DEGRADED,  # Not critical since it's UI only
                last_check=current_time,
                details={"error": str(e), "note": "Mission Control may not be running"},
                error_count=self.components["mission_control"].error_count + 1,
                uptime_seconds=current_time - self.start_time
            )

    def _check_vertex_ai(self, current_time: float):
        """Check Vertex AI service health status"""
        try:
            # Check for degraded mode flag first
            degraded_flag = Path("state/vertex_degraded.json")
            if degraded_flag.exists():
                try:
                    with open(degraded_flag) as f:
                        data = json.load(f)
                    details = {
                        "service_available": False,
                        "fallback_active": True,
                        "mode": "template_based",
                        "since": data.get("started", "unknown")
                    }
                    status = HealthStatus.DEGRADED
                except:
                    details = {"error": "Could not read degraded flag"}
                    status = HealthStatus.UNKNOWN
            else:
                # Try to check if Vertex AI is actually available
                try:
                    from intent_engine import IntentEngine
                    # Quick test - this will fail if Vertex AI is down
                    engine = IntentEngine()
                    details = {
                        "service_available": True,
                        "fallback_active": False,
                        "mode": "normal"
                    }
                    status = HealthStatus.HEALTHY
                except Exception as e:
                    if "404" in str(e) or "403" in str(e):
                        details = {
                            "service_available": False,
                            "fallback_active": True,
                            "mode": "template_based",
                            "error": str(e)[:100]
                        }
                        status = HealthStatus.DEGRADED
                    else:
                        details = {"error": str(e)[:100]}
                        status = HealthStatus.CRITICAL
            
            self.components["vertex_ai"] = ComponentHealth(
                component="Vertex AI",
                status=status,
                last_check=current_time,
                details=details,
                uptime_seconds=current_time - self.start_time
            )
            
        except Exception as e:
            self.components["vertex_ai"] = ComponentHealth(
                component="Vertex AI",
                status=HealthStatus.CRITICAL,
                last_check=current_time,
                details={"error": str(e)},
                error_count=self.components.get("vertex_ai", ComponentHealth("", HealthStatus.UNKNOWN, 0, {})).error_count + 1,
                uptime_seconds=current_time - self.start_time
            )
    
    def _log_health_summary(self):
        """Log overall health summary"""
        overall_status = self._calculate_overall_status()
        
        from event_log import emit_event
        emit_event(
            "system-health",
            event="health_check.summary",
            status="info" if overall_status == HealthStatus.HEALTHY else "warning",
            details={
                "overall_status": overall_status.value,
                "component_count": len(self.components),
                "healthy_components": len([c for c in self.components.values() if c.status == HealthStatus.HEALTHY]),
                "degraded_components": len([c for c in self.components.values() if c.status == HealthStatus.DEGRADED]),
                "critical_components": len([c for c in self.components.values() if c.status == HealthStatus.CRITICAL]),
                "system_uptime": time.time() - self.start_time
            }
        )
    
    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall system health status"""
        if not self.components:
            return HealthStatus.UNKNOWN
        
        statuses = [comp.status for comp in self.components.values()]
        
        # Critical if any component is critical
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        
        # Degraded if any component is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # Unknown if any component is unknown
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        
        # Healthy if all components are healthy
        return HealthStatus.HEALTHY
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status"""
        overall_status = self._calculate_overall_status()
        
        # Calculate summary metrics
        total_components = len(self.components)
        healthy_count = len([c for c in self.components.values() if c.status == HealthStatus.HEALTHY])
        degraded_count = len([c for c in self.components.values() if c.status == HealthStatus.DEGRADED])
        critical_count = len([c for c in self.components.values() if c.status == HealthStatus.CRITICAL])
        
        summary = {
            "total_components": total_components,
            "healthy_components": healthy_count,
            "degraded_components": degraded_count,
            "critical_components": critical_count,
            "system_uptime": time.time() - self.start_time,
            "health_percentage": (healthy_count / total_components * 100) if total_components > 0 else 0
        }
        
        return SystemHealth(
            overall_status=overall_status,
            components=self.components.copy(),
            timestamp=time.time(),
            summary=summary
        )
    
    def get_health_report(self) -> str:
        """Generate human-readable health report"""
        health = self.get_system_health()
        
        report = f"Project Resilience System Health Report\n"
        report += f"{'='*50}\n"
        report += f"Overall Status: {health.overall_status.value.upper()}\n"
        report += f"System Uptime: {health.summary['system_uptime']:.0f} seconds\n"
        report += f"Health Score: {health.summary['health_percentage']:.1f}%\n\n"
        
        report += "Component Status:\n"
        report += f"{'-'*30}\n"
        
        for comp_name, comp_health in health.components.items():
            status_symbol = {
                HealthStatus.HEALTHY: "✓",
                HealthStatus.DEGRADED: "⚠",
                HealthStatus.CRITICAL: "✗",
                HealthStatus.UNKNOWN: "?"
            }.get(comp_health.status, "?")
            
            report += f"{status_symbol} {comp_health.component:<20} {comp_health.status.value.upper()}\n"
            
            if comp_health.details:
                for key, value in comp_health.details.items():
                    if key != "error":
                        report += f"    {key}: {value}\n"
                if "error" in comp_health.details:
                    report += f"    Error: {comp_health.details['error']}\n"
            report += "\n"
        
        return report

# Global health monitor instance
health_monitor = HealthMonitor()

def start_health_monitoring():
    """Start system health monitoring"""
    health_monitor.start_monitoring()

def stop_health_monitoring():
    """Stop system health monitoring"""
    health_monitor.stop_monitoring()

def get_current_health() -> SystemHealth:
    """Get current system health status"""
    return health_monitor.get_system_health()

def print_health_report():
    """Print human-readable health report"""
    print(health_monitor.get_health_report())

if __name__ == "__main__":
    # Start monitoring and print initial report
    start_health_monitoring()
    time.sleep(2)  # Allow initial checks
    print_health_report()
    
    # Keep monitoring
    try:
        while True:
            time.sleep(30)
            print("\n" + "="*50)
            print(f"Health Check at {datetime.now().strftime('%H:%M:%S')}")
            print("="*50)
            print_health_report()
    except KeyboardInterrupt:
        stop_health_monitoring()
        print("Health monitoring stopped.")