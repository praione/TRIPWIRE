# apis.py
# This file centralizes the definitions for interacting with subsystems,
# making the main orchestrator cleaner and easier to maintain.

"""
Central registry + light helpers for DynamicAgentFactory entrypoints.
Edit this list (order matters) instead of hardcoding names in the orchestrator.
"""

# --- Factory + return types ---
# Canonical success type (dataclass)
LIVETEAM_IMPORT = "dynamic_agent_factory.LiveTeam"

# Fallback when all factory APIs fail
FALLBACK_FUNC = "_fabricate_team_from_blueprint"

# Behavior flags
STRICT_RETURN_TYPES = True      # Only allow LiveTeam or dict
PRINT_STACK_ON_ERROR = True     # Print traceback when an API fails

# --- Ordered list of factory APIs to try ---
# Each item: (method_name, parameter_name_to_use_for_blueprint)
FACTORY_APIS = [
    ("instantiate_team_from_blueprint", "institution_blueprint"),  # primary
    ("create_team",                       "blueprint"),              # fallback 1
    ("instantiate_team",                  "blueprint"),              # fallback 2
    ("build_team",                        "blueprint"),              # fallback 3
]


# --- Minimal helper to import a dotted symbol like "pkg.mod.Class" ---
def load_symbol(dotted: str):
    import importlib
    mod, _, attr = dotted.rpartition(".")
    if not mod:
        raise ImportError(f"Invalid import path: {dotted}")
    return getattr(importlib.import_module(mod), attr)

