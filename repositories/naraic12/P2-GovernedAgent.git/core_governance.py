# --- core_governance.py ---
# V5.3: Added Simulacrum class for intent-based philosophical alignment checks.
# - New Simulacrum class encapsulates the logic for evaluating the "spirit of the law".
# - It uses an LLM to reason about an agent's proposed action against its core ethos.

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import random
import time
import json
import re

# ---------------------------
# Vertex AI initialization
# ---------------------------
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationResponse, GenerationConfig
    
    PROJECT_ID = "project-resilience-ai-one"
    LOCATION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    MODEL = GenerativeModel("gemini-2.5-pro")
    print("Vertex AI and Gemini 2.5 Pro model initialized successfully.")
except Exception as e:
    raise RuntimeError(
        "Vertex AI init failed. Ensure google-cloud-aiplatform is installed and ADC is configured."
    ) from e


# ===============================
# Validation contracts & logic
# ===============================

# --- Retry backoff (Week 4) ---
def _retry_delay_seconds(next_attempt: int) -> float:
    # next_attempt: 2 -> ~2s, 3 -> ~6s (±10% jitter).
    schedule = {2: 2.0, 3: 6.0}
    base = schedule.get(next_attempt, 12.0)
    jitter = base * 0.10
    return base + random.uniform(-jitter, jitter)


@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None
    reason_code: Optional[str] = None
    adjusted_text: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class ConstitutionValidator:
    """Validates text and tool access against rules in a loaded dictionary."""
    def __init__(self, rules: Dict[str, Any]):
        self._raw = rules or {}
        # Schema-tolerant loading
        self.old_immutable = self._raw.get("immutable_laws", []) or []
        self.old_min = int(self._raw.get("min_words", 0) or 0)
        self.old_max = int(self._raw.get("max_words", 100000) or 100000)
        self.old_grade = float(self._raw.get("readability_grade", 20) or 20)
        self.disallow_emojis = bool(self._raw.get("disallow_emojis", False))
        self.disallow_hashtags = bool(self._raw.get("disallow_hashtags", False))
        self.forbidden_phrases = [str(p) for p in (self._raw.get("forbidden_phrases") or [])]
        self.constitution = self._raw.get("constitution") or {}
        self.quality = self._raw.get("quality_gates") or {}
        # NEW: Load agent permissions for tool access validation
        self.agent_permissions = self._raw.get("agent_permissions", {})

    def validate(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        res = self.validate_structured(text)
        return res.is_valid, res.reason, res.adjusted_text

    def validate_tool_access(self, agent_name: str, tool_name: str) -> ValidationResult:
        """
        Checks if an agent has permission to use a specific tool based on rules_core.yaml.
        """
        agent_rules = self.agent_permissions.get(agent_name)
        
        # If the agent has no specific rules, fall back to the Default policy
        if not agent_rules:
            agent_rules = self.agent_permissions.get("Default", {})

        allowed_tools: List[str] = agent_rules.get("allowed_tools", [])

        if tool_name in allowed_tools:
            return ValidationResult(True, f"Tool '{tool_name}' is permitted for agent '{agent_name}'.")
        else:
            return ValidationResult(
                is_valid=False,
                reason=f"Tool '{tool_name}' is not in the list of allowed tools for agent '{agent_name}'.",
                reason_code="TOOL_ACCESS_DENIED"
            )

    def validate_structured(self, text: str) -> ValidationResult:
        t = (text or "").strip()

        # Check immutable laws (new and old schemas)
        immutable_laws = self.constitution.get("immutable_laws", []) or []
        for rule in immutable_laws + self.old_immutable:
            if isinstance(rule, str) and rule and rule in t:
                return ValidationResult(False, f"Violated immutable law: '{rule}'", "CONSTITUTION_VIOLATION")
            elif isinstance(rule, dict):
                pat = str(rule.get("pattern", "") or "")
                desc = str(rule.get("description", f"Violated immutable law: '{pat}'"))
                if pat and re.search(pat, t):
                    return ValidationResult(False, desc, "CONSTITUTION_VIOLATION")

        # Content filters
        if self.disallow_hashtags and re.search(r"#\w+", t):
            cleaned = re.sub(r"#\w+\s*", "", t).strip()
            return ValidationResult(False, "Disallowed hashtags.", "HASHTAG_DISALLOWED", adjusted_text=cleaned)
        if self.disallow_emojis and re.search(r"[\U0001F300-\U0001FAFF]", t):
            return ValidationResult(False, "Disallowed emojis.", "EMOJI_DISALLOWED")
        for phrase in self.forbidden_phrases:
            if phrase and phrase.lower() in t.lower():
                return ValidationResult(False, f"Forbidden phrase: '{phrase}'", "FORBIDDEN_PHRASE")

        # Quality gates (word count, readability)
        words = len(t.split())
        min_words = int(self.quality.get("min_words", self.old_min))
        max_words = int(self.quality.get("max_words", self.old_max))
        if words < min_words: return ValidationResult(False, f"Word count ({words}) < {min_words}.", "QUALITY_BELOW_MIN_WORDS")
        if words > max_words: return ValidationResult(False, f"Word count ({words}) > {max_words}.", "QUALITY_ABOVE_MAX_WORDS")
        
        try:
            import textstat
            max_grade = float(self.quality.get("max_readability_grade", self.old_grade))
            grade = float(textstat.flesch_kincaid_grade(t))
            if grade > max_grade:
                return ValidationResult(False, f"Readability score ({grade:.2f}) > target ({max_grade}).", "QUALITY_POOR_READABILITY", metrics={"readability_score": grade})
            metrics = {"readability_score": f"{grade:.2f}/{max_grade}"}
        except Exception:
            metrics = {"readability_score": "not_checked"}

        return ValidationResult(True, "All checks passed.", metrics=metrics)


class SALAwareConstitutionValidator(ConstitutionValidator):
    def validate_structured(self, text: str) -> ValidationResult:
        print("--- Running Validation ---")
        res = super().validate_structured(text)
        if not res.is_valid: print(f"VALIDATION FAILED: {res.reason}")
        else: print("--- Validation PASSED. Finalizing response. ---")
        return res

def get_validator_for_sal(rules: Dict[str, Any]) -> SALAwareConstitutionValidator:
    return SALAwareConstitutionValidator(rules)

def make_sal_feedback_prompt(result: ValidationResult, task_hint: str) -> str:
    return "\n".join([
        "IMPERATIVE: Your previous response failed a mandatory validation check.",
        f"REASON: {result.reason}",
        "You MUST correct this. Rewrite your response to address the issue directly.",
        "---",
        f"Original Task Context: {task_hint or 'Rewrite to satisfy all rules.'}",
    ])


# ===============================
# Simulacrum of Spirit (NEW: Task 1.3)
# ===============================
class Simulacrum:
    """
    Evaluates the philosophical alignment of a proposed agent action.
    This is the "spirit of the law" check.
    """
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        if MODEL is None: raise ConnectionError("Gemini model is not available for Simulacrum.")
        self.model = MODEL
        print("Simulacrum initialized successfully.")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"[Simulacrum WARN] Spirit file not found: {path}. Cannot evaluate intent.")
            return {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"Invalid YAML at {path}: {e}")

    def _create_simulacrum_prompt(self, agent_name: str, spirit_data: Dict[str, Any], task_payload: Dict[str, Any]) -> str:
        """Constructs the system prompt for the intent evaluation."""
        spirit_text = json.dumps(spirit_data, indent=2)
        task_text = json.dumps(task_payload, indent=2)

        return f"""
You are the Simulacrum of Spirit, a core component of a governed AI agent colony.
Your function is to determine if a proposed action is philosophically aligned with the agent's core identity and ethos. You are not checking for rules, but for *intent*.

**Agent's Core Identity (Spirit):**
```json
{spirit_text}
```

**Proposed Action (Task Payload):**
```json
{task_text}
```

**Your Task:**
Based on the agent's spirit, evaluate the proposed action. Is the action a direct and coherent consequence of the agent's stated ethos and directives? Consider the second-order effects and potential for drift.

Respond with a single, valid JSON object with your verdict. Do NOT include markdown formatting or any other text.
The JSON object must have two keys: 'verdict' ('allow' or 'deny') and 'rationale' (a brief explanation).

Example:
{{"verdict": "allow", "rationale": "The action to summarize the document aligns with the agent's directive to clarify complex information."}}
"""

    def evaluate_intent(self, agent_name: str, task_payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Performs the "spirit of the law" check using an LLM.
        """
        # Search multiple locations for spirit files
        from pathlib import Path
        spirit_file = None
        search_locations = [
            self.config_dir / f"{agent_name.lower()}_spirit.yaml",
            Path("generated_spirits") / f"{agent_name}_spirit.yaml",
            Path("deployed_agents/spirits") / f"agent_{agent_name}_spirit.yaml",
            Path("deployed_agents/spirits") / f"{agent_name}_spirit.yaml",
        ]
        
        for path in search_locations:
            if path.exists():
                spirit_file = path
                print(f"[Simulacrum] Found spirit at: {path}")
                break
        
        if not spirit_file:
            return {"verdict": "allow", "rationale": f"No spirit file found for {agent_name} in any location, skipping intent check."}
        
        spirit_data = self._load_yaml(spirit_file)

        if not spirit_data:
            return {"verdict": "allow", "rationale": f"Spirit file {spirit_file} is empty or invalid, skipping intent check."}

        prompt = self._create_simulacrum_prompt(agent_name, spirit_data, task_payload)

        for attempt in range(1, max_retries + 1):
            print(f"--- Simulacrum Intent Evaluation --- Attempt {attempt}/{max_retries}")
            try:
                gen_config = GenerationConfig(response_mime_type="application/json")
                response = self.model.generate_content(prompt, generation_config=gen_config)
                raw_text = response.text or ""
                
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        parsed = json.loads(json_str)
                        if "verdict" in parsed and "rationale" in parsed:
                             print(f"[Simulacrum OK] Successfully parsed intent verdict.")
                             return parsed
                        else:
                            failure_reason = "Model returned valid JSON but with missing keys."
                    except json.JSONDecodeError:
                        failure_reason = "Model returned invalid JSON structure."
                else:
                    failure_reason = "Model did not return a JSON object."
                
                print(f"[Simulacrum WARN] {failure_reason} Retrying...")

            except Exception as e:
                print(f"[Simulacrum ERROR] An exception occurred: {e}. Retrying...")
            
            if attempt < max_retries:
                time.sleep(_retry_delay_seconds(attempt + 1))
        
        print("[Simulacrum FAIL] Max retries reached. Returning default deny verdict.")
        return {
            "verdict": "deny",
            "rationale": "Simulacrum failed to produce a valid verdict after multiple retries."
        }


# ===============================
# ArchitectAgent (meta-agent)
# ===============================
class ArchitectAgent:
    def __init__(
        self,
        voice_config: Dict[str, Any],
        spirit_file: Path,
        rules_file: Path,
        prompt_file: Optional[Path] = None,
    ):
        self.voice = voice_config or {}
        self.spirit_file = Path(spirit_file)
        self.rules_file = Path(rules_file)
        self.agent_name = self.spirit_file.stem.replace('_spirit', '').lower()
        
        self.prompt_file = Path(prompt_file) if prompt_file else None
        self.spirit_data = self._load_yaml(self.spirit_file)
        self.rules_data = self._load_yaml(self.rules_file)
        self.prompt_template = self._load_text(self.prompt_file) if self.prompt_file else ""

        self.validator = ConstitutionValidator(self.rules_data)
        if MODEL is None: raise ConnectionError("Gemini model is not available.")
        self.model = MODEL
        print(f"Gemini model assigned successfully for agent: {self.agent_name}")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f) or {}
        except FileNotFoundError: raise FileNotFoundError(f"Config file not found: {path}")
        except yaml.YAMLError as e: raise RuntimeError(f"Invalid YAML at {path}: {e}")

    def _load_text(self, path: Optional[Path]) -> str:
        if not path: return ""
        try: return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError: return ""

    def _create_system_prompt(self, user_prompt: str) -> str:
        spirit = self.spirit_data or {}
        rules = self.rules_data or {}
        voice = self.voice.get("authorial_voice", {}) or self.voice

        # Voice components
        av_desc  = voice.get("description", "Clear, practical guidance.")
        av_tone  = voice.get("tone", "direct, pragmatic")
        av_motto = voice.get("motto", "Show the map. Then move.")
        av_style = voice.get("style_rules", [])
        av_banned = voice.get("banned", [])

        # Spirit components
        identity = spirit.get("identity", {})
        identity_title   = identity.get("title", "Agent")
        identity_persona = identity.get("persona", "helpful specialist")
        ethos_list = spirit.get("ethos", [])
        directives_list = spirit.get("directives", [])

        # Build prompt sections
        sections = [
            f"You are the {identity_title}, a {identity_persona}.",
            f"Global Voice: {av_desc} Tone: {av_tone}. Motto: {av_motto}.",
        ]
        if av_style: sections.extend([f"- Style: {s}" for s in av_style])
        if av_banned: sections.extend([f"- Do not use: {b}" for b in av_banned])
        if ethos_list:
            sections.append("\n**Core Ethos:**")
            sections.extend([f"- {e}" for e in ethos_list])
        if directives_list:
            sections.append("\n**Operational Directives:**")
            sections.extend([f"- {d}" for d in directives_list])
        
        # Guardian-specific output contract
        if self.agent_name == "guardian":
            sections.append("\n**Output Contract (CRITICAL):**")
            sections.append("- You MUST respond with a single, valid JSON object and nothing else.")
            sections.append("- Do NOT include markdown formatting like ```json or any conversational text.")
            sections.append("Example: {\"verdict\": \"deny\", \"rule_id\": \"RISK-002\", \"rationale\": \"The prompt conflicts with the core authorial voice.\"}")

        sections.append(f"\n**User Request:**\n{user_prompt}")
        if self.prompt_template:
            sections.append("\n**Task Hints:**")
            sections.append(self.prompt_template)

        return "\n".join(sections)

    def _generate_guardian_verdict(self, user_prompt: str, max_retries: int = 3) -> str:
        """
        A dedicated, resilient method for the Guardian agent.
        It will retry until a valid JSON object is extracted.
        """
        for attempt in range(1, max_retries + 1):
            print(f"--- Guardian Verdict Generation --- Attempt {attempt}/{max_retries}")
            try:
                gen_config = GenerationConfig(response_mime_type="application/json")
                system_prompt = self._create_system_prompt(user_prompt)
                
                response = self.model.generate_content(system_prompt, generation_config=gen_config)
                raw_text = response.text or ""
                
                # Resiliently find the first JSON object in the response
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        parsed = json.loads(json_str)
                        print(f"[Guardian OK] Successfully parsed JSON verdict.")
                        return json.dumps(parsed) # Return clean, re-serialized JSON
                    except json.JSONDecodeError:
                        failure_reason = "Model returned invalid JSON structure."
                else:
                    failure_reason = "Model did not return a JSON object."
                
                print(f"[Guardian WARN] {failure_reason} Retrying...")

            except Exception as e:
                print(f"[Guardian ERROR] An exception occurred: {e}. Retrying...")
            
            if attempt < max_retries:
                time.sleep(_retry_delay_seconds(attempt + 1))

        print("[Guardian FAIL] Max retries reached. Returning default deny verdict.")
        return json.dumps({
            "verdict": "deny",
            "rule_id": "GEN-001",
            "rationale": "Guardian agent failed to produce a valid JSON response after multiple retries."
        })

    def _generate_doer_text(self, user_prompt: str, max_retries: int = 3) -> str:
        """
        A dedicated method for Doer agents to generate and validate text output.
        """
        failure_reason = ""
        last_response_text = ""
        current_prompt = user_prompt

        for attempt in range(1, max_retries + 1):
            print(f"--- Doer Agent Generation ({self.agent_name}) --- Attempt {attempt}/{max_retries}")
            
            system_prompt = self._create_system_prompt(current_prompt)

            try:
                response = self.model.generate_content(system_prompt)
                last_response_text = response.text or ""
            except Exception as e:
                print(f"[LLM Generation Error] {e}")
                last_response_text = ""

            result = self.validator.validate_structured(last_response_text)
            if result.is_valid:
                return result.adjusted_text or last_response_text

            # Prepare for retry
            current_prompt = make_sal_feedback_prompt(result, task_hint=self.prompt_template or "")
            if attempt < max_retries:
                time.sleep(_retry_delay_seconds(attempt + 1))
        
        print("--- MAX RETRIES REACHED. Returning last failed attempt. ---")
        return last_response_text

    def generate_rational_output(self, user_prompt: str, max_retries: int = 3) -> str:
        """
        Main entry point. Routes to the appropriate specialized helper function.
        """
        if self.agent_name == "guardian":
            return self._generate_guardian_verdict(user_prompt, max_retries)
        else:
            return self._generate_doer_text(user_prompt, max_retries)

