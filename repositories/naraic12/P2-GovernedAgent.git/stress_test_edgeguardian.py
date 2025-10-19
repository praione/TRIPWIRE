# stress_test_edgeguardian.py
# Comprehensive stress testing for EdgeGuardian recovery under extreme conditions

import os
import json
import time
import random
import hashlib
from pathlib import Path
from google.cloud import storage
from edge_guardian import make_snapshot, seal_and_store, recover

def test_corrupted_shards():
    """Test recovery when shards are corrupted"""
    print("=== Corrupted Shards Test ===")
    
    # Create a snapshot
    snapshot = make_snapshot("stress_test_001", "completed", ["intro", "mentor", "outro"])
    manifest_uri = seal_and_store(snapshot)
    
    if not manifest_uri:
        print("Failed to create initial snapshot")
        return False
    
    # Parse manifest location
    bucket_name = os.getenv("STATE_BUCKET", "project-resilience-agent-state")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Get manifest path
    manifest_path = manifest_uri.split(f"gs://{bucket_name}/")[1]
    base_path = manifest_path.replace("manifest.json", "")
    
    # Load manifest to find shards
    manifest_blob = bucket.blob(manifest_path)
    manifest_data = json.loads(manifest_blob.download_as_text())
    
    # Corrupt 2 shards (should still recover with 3 remaining)
    corrupted_count = 0
    for obj in manifest_data["objects"][:2]:
        shard_path = obj["gcs"]
        shard_blob = bucket.blob(shard_path)
        
        # Corrupt the shard with random data
        corrupt_data = os.urandom(100)
        shard_blob.upload_from_string(corrupt_data)
        corrupted_count += 1
        print(f"Corrupted shard: {shard_path}")
    
    # Attempt recovery
    status, recovered = recover()
    
    if status == "SUCCESS":
        print(f"✓ Recovery succeeded with {corrupted_count} corrupted shards")
        return True
    else:
        print(f"✗ Recovery failed: {status}")
        return False

def test_missing_shards():
    """Test recovery when shards are missing"""
    print("\n=== Missing Shards Test ===")
    
    snapshot = make_snapshot("stress_test_002", "completed", ["intro", "mentor", "outro"])
    manifest_uri = seal_and_store(snapshot)
    
    if not manifest_uri:
        print("Failed to create initial snapshot")
        return False
    
    bucket_name = os.getenv("STATE_BUCKET", "project-resilience-agent-state")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    manifest_path = manifest_uri.split(f"gs://{bucket_name}/")[1]
    manifest_blob = bucket.blob(manifest_path)
    manifest_data = json.loads(manifest_blob.download_as_text())
    
    # Delete 2 shards (should still recover with 3 remaining)
    deleted_count = 0
    for obj in manifest_data["objects"][:2]:
        shard_path = obj["gcs"]
        shard_blob = bucket.blob(shard_path)
        shard_blob.delete()
        deleted_count += 1
        print(f"Deleted shard: {shard_path}")
    
    status, recovered = recover()
    
    if status == "SUCCESS":
        print(f"✓ Recovery succeeded with {deleted_count} missing shards")
        return True
    else:
        print(f"✗ Recovery failed: {status}")
        return False

def test_corrupted_manifest():
    """Test behavior when manifest is corrupted"""
    print("\n=== Corrupted Manifest Test ===")
    
    snapshot = make_snapshot("stress_test_003", "completed", ["intro", "mentor", "outro"])
    manifest_uri = seal_and_store(snapshot)
    
    if not manifest_uri:
        print("Failed to create initial snapshot")
        return False
    
    bucket_name = os.getenv("STATE_BUCKET", "project-resilience-agent-state")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    manifest_path = manifest_uri.split(f"gs://{bucket_name}/")[1]
    manifest_blob = bucket.blob(manifest_path)
    
    # Corrupt manifest with invalid JSON
    corrupt_manifest = '{"invalid": json syntax,,,}'
    manifest_blob.upload_from_string(corrupt_manifest)
    print("Corrupted manifest with invalid JSON")
    
    status, recovered = recover()
    
    if status == "RECONSTRUCTION_FAILED":
        print("✓ Recovery correctly failed with corrupted manifest")
        return True
    else:
        print(f"✗ Unexpected recovery result: {status}")
        return False

def test_threshold_boundary():
    """Test recovery exactly at the K=3 threshold"""
    print("\n=== Threshold Boundary Test ===")
    
    snapshot = make_snapshot("stress_test_004", "completed", ["intro", "mentor", "outro"])
    manifest_uri = seal_and_store(snapshot)
    
    if not manifest_uri:
        print("Failed to create initial snapshot")
        return False
    
    bucket_name = os.getenv("STATE_BUCKET", "project-resilience-agent-state")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    manifest_path = manifest_uri.split(f"gs://{bucket_name}/")[1]
    manifest_blob = bucket.blob(manifest_path)
    manifest_data = json.loads(manifest_blob.download_as_text())
    
    # Delete exactly 2 shards, leaving exactly K=3 for recovery
    for obj in manifest_data["objects"][:2]:
        shard_path = obj["gcs"]
        shard_blob = bucket.blob(shard_path)
        shard_blob.delete()
    
    print("Deleted 2 shards, exactly 3 remaining (threshold boundary)")
    
    status, recovered = recover()
    
    if status == "SUCCESS":
        print("✓ Recovery succeeded exactly at K=3 threshold")
        return True
    else:
        print(f"✗ Recovery failed at threshold: {status}")
        return False

def test_complete_failure():
    """Test behavior when too many shards are missing (should fail gracefully)"""
    print("\n=== Complete Failure Test ===")
    
    snapshot = make_snapshot("stress_test_005", "completed", ["intro", "mentor", "outro"])
    manifest_uri = seal_and_store(snapshot)
    
    if not manifest_uri:
        print("Failed to create initial snapshot")
        return False
    
    bucket_name = os.getenv("STATE_BUCKET", "project-resilience-agent-state")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    manifest_path = manifest_uri.split(f"gs://{bucket_name}/")[1]
    manifest_blob = bucket.blob(manifest_path)
    manifest_data = json.loads(manifest_blob.download_as_text())
    
    # Delete 4 shards, leaving only 1 (below K=3 threshold)
    for obj in manifest_data["objects"][:4]:
        shard_path = obj["gcs"]
        shard_blob = bucket.blob(shard_path)
        shard_blob.delete()
    
    print("Deleted 4 shards, only 1 remaining (below threshold)")
    
    status, recovered = recover()
    
    if status == "RECONSTRUCTION_FAILED":
        print("✓ Recovery correctly failed with insufficient shards")
        return True
    else:
        print(f"✗ Unexpected recovery result: {status}")
        return False

def test_rapid_recovery_cycles():
    """Test multiple rapid recovery cycles to check for resource leaks"""
    print("\n=== Rapid Recovery Cycles Test ===")
    
    success_count = 0
    total_cycles = 5
    
    for i in range(total_cycles):
        print(f"Recovery cycle {i+1}/{total_cycles}")
        
        # Create snapshot
        snapshot = make_snapshot(f"rapid_test_{i}", "completed", ["intro", "mentor", "outro"])
        manifest_uri = seal_and_store(snapshot)
        
        if manifest_uri:
            # Immediate recovery
            status, recovered = recover()
            if status == "SUCCESS":
                success_count += 1
            
            # Brief pause
            time.sleep(0.5)
    
    success_rate = (success_count / total_cycles) * 100
    print(f"Recovery success rate: {success_rate}% ({success_count}/{total_cycles})")
    
    return success_rate >= 80  # Allow for some variance in cloud operations

def run_stress_tests():
    """Run comprehensive EdgeGuardian stress test suite"""
    print("EdgeGuardian Stress Test Suite")
    print("=" * 50)
    
    tests = [
        ("Corrupted Shards", test_corrupted_shards),
        ("Missing Shards", test_missing_shards),
        ("Corrupted Manifest", test_corrupted_manifest),
        ("Threshold Boundary", test_threshold_boundary),
        ("Complete Failure", test_complete_failure),
        ("Rapid Recovery Cycles", test_rapid_recovery_cycles)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("STRESS TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("✓ EdgeGuardian is stress-test ready for demo")
    else:
        print("✗ EdgeGuardian needs hardening before demo")
    
    return passed == len(results)

if __name__ == "__main__":
    run_stress_tests()