import yaml
from pathlib import Path

try:
    with open('config/governance_resilience.yaml', 'r', encoding='utf-8-sig') as f:
        data = yaml.safe_load(f)
    print("✅ YAML loaded successfully!")
    print("Keys found:", list(data.keys()))
except Exception as e:
    print(f"❌ Error: {e}")
