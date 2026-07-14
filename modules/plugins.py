"""
Module 19 — Plugin Ecosystem

Makes the framework extensible via plugins:
  - EvaluatorPlugin: custom metrics beyond the 8 built-in
  - AttackPlugin: custom adversarial attack generators  
  - ModelAdapterPlugin: custom model integrations
  - MetricExporterPlugin: custom result exporters (HuggingFace, W&B, etc.)

Plugin discovery: scans plugins/ directory for .py files with Plugin classes.
Registry: manages lifecycle (load, validate, execute, unload).
"""

import json
import importlib.util
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

ROOT = Path(__file__).parent.parent
PLUGINS_DIR = ROOT / "plugins"


# ─── Base Plugin Classes ──────────────────────────────────────────────────────

class BasePlugin(ABC):
    """Base class for all plugins."""
    name: str = "unnamed"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    plugin_type: str = "base"

    def validate(self) -> bool:
        """Return True if plugin is correctly configured."""
        return True

    def get_metadata(self) -> dict:
        return {
            "name": self.name, "version": self.version,
            "description": self.description, "author": self.author,
            "type": self.plugin_type,
        }


class EvaluatorPlugin(BasePlugin):
    """Plugin for adding custom evaluation metrics."""
    plugin_type = "evaluator"

    @abstractmethod
    async def score(self, client, prompt: str, response: str, context: dict = None) -> dict:
        """
        Score a response on a custom metric.
        Returns dict with at minimum: {"score": float, "metric_name": str}
        """
        ...

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Name of the metric this plugin provides."""
        ...

    @property
    def score_range(self) -> tuple[float, float]:
        return 0.0, 1.0

    @property
    def higher_is_better(self) -> bool:
        return True


class AttackPlugin(BasePlugin):
    """Plugin for custom adversarial attack generation."""
    plugin_type = "attack"

    @abstractmethod
    def generate(self, n: int = 10, config: dict = None) -> list[dict]:
        """
        Generate n adversarial prompts.
        Each prompt: {"id", "text", "category", "expected_behavior", "priority"}
        """
        ...

    @property
    @abstractmethod
    def attack_category(self) -> str: ...


class ModelAdapterPlugin(BasePlugin):
    """Plugin for custom model providers."""
    plugin_type = "model_adapter"

    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 1.0,
                       max_tokens: int = 1024) -> tuple[str, int, int, float]:
        """Returns (text, input_tokens, output_tokens, latency_ms)."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


class MetricExporterPlugin(BasePlugin):
    """Plugin for exporting results to external systems."""
    plugin_type = "exporter"

    @abstractmethod
    def export(self, run_id: str, results: list[dict], stats: dict) -> bool:
        """Export results. Returns True on success."""
        ...


# ─── Built-in example plugins ─────────────────────────────────────────────────

class CoherenceEvaluatorPlugin(EvaluatorPlugin):
    """Built-in: measures response coherence and logical flow."""
    name = "coherence_evaluator"
    version = "1.0.0"
    description = "Measures response coherence, logical flow, and consistency"
    author = "llm_eval"

    @property
    def metric_name(self) -> str:
        return "coherence"

    async def score(self, client, prompt: str, response: str, context: dict = None) -> dict:
        import re
        judge_prompt = f"""Rate the coherence and logical flow of this response on a scale 0.0-1.0.
1.0 = perfectly coherent, logical, well-structured
0.5 = partially coherent, some logical gaps
0.0 = incoherent, contradictory, illogical

PROMPT: {prompt[:300]}
RESPONSE: {response[:500]}

Return ONLY JSON: {{"score": 0.0-1.0, "issues": ["list of coherence issues"]}}"""
        try:
            raw = await client.judge(judge_prompt)
            raw = re.sub(r"```json|```", "", raw).strip()
            data = __import__("json").loads(raw)
            return {"score": float(data.get("score", 0.7)), "metric_name": self.metric_name,
                    "issues": data.get("issues", [])}
        except Exception:
            return {"score": 0.7, "metric_name": self.metric_name}


class HallucinationRatePlugin(EvaluatorPlugin):
    """Built-in: measures hallucination rate via self-consistency."""
    name = "hallucination_rate"
    version = "1.0.0"
    description = "Measures factual consistency across multiple generations"
    author = "llm_eval"

    @property
    def metric_name(self) -> str:
        return "hallucination_rate"

    @property
    def higher_is_better(self) -> bool:
        return False  # Lower hallucination rate is better

    async def score(self, client, prompt: str, response: str, context: dict = None) -> dict:
        import re
        # Generate a second response and compare key facts
        try:
            resp2, _, _, _ = await client.generate(prompt, temperature=0.5)
            consistency_prompt = f"""Compare these two responses to the same question for factual consistency.

QUESTION: {prompt[:200]}
RESPONSE 1: {response[:300]}
RESPONSE 2: {resp2[:300]}

Are there any factual contradictions between them?
Return JSON: {{"contradictions": ["list"], "consistency_score": 0.0-1.0, "hallucination_rate": 0.0-1.0}}"""
            raw = await client.judge(consistency_prompt)
            raw = re.sub(r"```json|```", "", raw).strip()
            data = __import__("json").loads(raw)
            return {"score": float(data.get("hallucination_rate", 0.1)),
                    "metric_name": self.metric_name,
                    "consistency_score": data.get("consistency_score", 0.9)}
        except Exception:
            return {"score": 0.1, "metric_name": self.metric_name}


class JSONExporterPlugin(MetricExporterPlugin):
    """Built-in: export results to JSON file."""
    name = "json_exporter"
    version = "1.0.0"
    description = "Exports results to a JSON file"
    author = "llm_eval"

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or (ROOT / "data" / "exports")

    def export(self, run_id: str, results: list[dict], stats: dict) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / f"{run_id}_export.json"
        try:
            with open(out, "w") as f:
                json.dump({"run_id": run_id, "stats": stats, "results": results,
                           "exported_at": datetime.utcnow().isoformat()}, f, indent=2, default=str)
            print(f"[JSONExporter] Exported to {out}")
            return True
        except Exception as e:
            print(f"[JSONExporter] Failed: {e}")
            return False


class HuggingFaceExporterPlugin(MetricExporterPlugin):
    """Built-in: export results to HuggingFace dataset format."""
    name = "huggingface_exporter"
    version = "1.0.0"
    description = "Exports results as HuggingFace dataset-compatible JSON"
    author = "llm_eval"

    def export(self, run_id: str, results: list[dict], stats: dict) -> bool:
        out_dir = ROOT / "data" / "exports" / "hf_dataset"
        out_dir.mkdir(parents=True, exist_ok=True)
        # HuggingFace datasets format: metadata.json + data.jsonl
        metadata = {
            "dataset_name": f"llm_eval_{run_id}",
            "created_at": datetime.utcnow().isoformat(),
            "features": {
                "prompt": "string", "response": "string", "category": "string",
                "relevance": "float32", "toxicity": "float32", "asr": "float32",
                "bias": "float32", "latency_ms": "float32",
            },
            "split": "test",
            "n_examples": len(results),
        }
        try:
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            with open(out_dir / "data.jsonl", "w") as f:
                for r in results:
                    row = {k: v for k, v in r.items() if k in metadata["features"]}
                    f.write(json.dumps(row) + "\n")
            print(f"[HFExporter] Exported {len(results)} examples to {out_dir}")
            return True
        except Exception as e:
            print(f"[HFExporter] Failed: {e}")
            return False


# ─── Plugin Registry ──────────────────────────────────────────────────────────

class PluginRegistry:
    """
    Central plugin registry.
    Manages discovery, validation, and execution of all plugins.
    """

    def __init__(self):
        self._plugins: dict[str, BasePlugin] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register all built-in plugins."""
        self.register(CoherenceEvaluatorPlugin())
        self.register(HallucinationRatePlugin())
        self.register(JSONExporterPlugin())
        self.register(HuggingFaceExporterPlugin())

    def register(self, plugin: BasePlugin) -> bool:
        if not plugin.validate():
            print(f"[PluginRegistry] Plugin {plugin.name} failed validation")
            return False
        self._plugins[plugin.name] = plugin
        print(f"[PluginRegistry] Registered: {plugin.name} v{plugin.version} ({plugin.plugin_type})")
        return True

    def discover_from_dir(self, plugins_dir: Path = None):
        """Scan directory for plugin files and auto-register."""
        scan_dir = plugins_dir or PLUGINS_DIR
        if not scan_dir.exists():
            scan_dir.mkdir(parents=True, exist_ok=True)
            return
        for py_file in scan_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BasePlugin) and obj not in
                            (BasePlugin, EvaluatorPlugin, AttackPlugin,
                             ModelAdapterPlugin, MetricExporterPlugin)):
                        self.register(obj())
            except Exception as e:
                print(f"[PluginRegistry] Failed to load {py_file.name}: {e}")

    def get(self, name: str) -> Optional[BasePlugin]:
        return self._plugins.get(name)

    def get_by_type(self, plugin_type: str) -> list[BasePlugin]:
        return [p for p in self._plugins.values() if p.plugin_type == plugin_type]

    def list_all(self) -> list[dict]:
        return [p.get_metadata() for p in self._plugins.values()]

    async def run_evaluator_plugins(self, client, prompt: str, response: str,
                                     context: dict = None) -> dict:
        """Run all registered evaluator plugins and return combined scores."""
        scores = {}
        for plugin in self.get_by_type("evaluator"):
            try:
                result = await plugin.score(client, prompt, response, context)
                scores[plugin.metric_name] = result
            except Exception as e:
                scores[plugin.metric_name] = {"score": None, "error": str(e)}
        return scores

    def run_exporters(self, run_id: str, results: list[dict], stats: dict) -> dict:
        """Run all registered exporter plugins."""
        outcomes = {}
        for plugin in self.get_by_type("exporter"):
            try:
                ok = plugin.export(run_id, results, stats)
                outcomes[plugin.name] = "success" if ok else "failed"
            except Exception as e:
                outcomes[plugin.name] = f"error: {e}"
        return outcomes

    def generate_attack_plugins(self, n: int = 5) -> list[dict]:
        """Generate attacks from all registered attack plugins."""
        attacks = []
        for plugin in self.get_by_type("attack"):
            try:
                attacks.extend(plugin.generate(n))
            except Exception as e:
                print(f"[PluginRegistry] Attack plugin {plugin.name} failed: {e}")
        return attacks


# Global registry
registry = PluginRegistry()
