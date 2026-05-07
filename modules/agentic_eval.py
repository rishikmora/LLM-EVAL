"""
Module 09 — Agentic Evaluation (Tier 2 Enhancement)

Evaluates LLMs in agentic/tool-use contexts:
  - Tool selection accuracy (function calling)
  - Planning success rate
  - Hallucinated tool usage
  - Chain-of-thought consistency
  - Memory retention across turns
"""

import json
import re
import asyncio
from pathlib import Path
from typing import Optional
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "eval_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─── Tool definitions (mock function registry) ────────────────────────────────

AVAILABLE_TOOLS = {
    "search_web": {"description": "Search the internet for current information", "params": ["query"]},
    "get_weather": {"description": "Get current weather for a location", "params": ["location"]},
    "calculate": {"description": "Perform mathematical calculations", "params": ["expression"]},
    "get_stock_price": {"description": "Get current stock price", "params": ["ticker"]},
    "send_email": {"description": "Send an email", "params": ["to", "subject", "body"]},
    "read_file": {"description": "Read contents of a file", "params": ["filepath"]},
    "write_file": {"description": "Write content to a file", "params": ["filepath", "content"]},
    "run_sql": {"description": "Execute a SQL query", "params": ["query", "database"]},
}

HALLUCINATED_TOOLS = [
    "magic_summarize", "instant_translate", "perfect_classify",
    "auto_debug", "smart_fix", "universal_api",
]

# ─── Agentic test cases ───────────────────────────────────────────────────────

AGENTIC_TEST_CASES = [
    {
        "id": "AGENT_001",
        "name": "correct_tool_selection_search",
        "description": "Model should choose search_web for a current events question",
        "user_request": "What is the current weather in Tokyo?",
        "expected_tool": "get_weather",
        "expected_params_contain": ["Tokyo"],
        "category": "tool_selection",
    },
    {
        "id": "AGENT_002",
        "name": "correct_tool_selection_math",
        "description": "Model should choose calculate for arithmetic",
        "user_request": "What is 15% of 847?",
        "expected_tool": "calculate",
        "expected_params_contain": ["847", "0.15"],
        "category": "tool_selection",
    },
    {
        "id": "AGENT_003",
        "name": "no_hallucinated_tools",
        "description": "Model should not invent tools not in the registry",
        "user_request": "Summarize this document for me: [long document here]",
        "expected_tool": None,  # should use available tools or decline
        "should_not_use": HALLUCINATED_TOOLS,
        "category": "hallucination_prevention",
    },
    {
        "id": "AGENT_004",
        "name": "multi_step_planning",
        "description": "Model should break down a complex task into correct tool sequence",
        "user_request": "Research the current stock price of AAPL, then send me an email with the result.",
        "expected_tool_sequence": ["get_stock_price", "send_email"],
        "category": "planning",
    },
    {
        "id": "AGENT_005",
        "name": "memory_retention",
        "description": "Model should remember a fact stated earlier in conversation",
        "turns": [
            "My name is Alice and I live in Boston.",
            "What is 2 + 2?",
            "What city do I live in?",
        ],
        "expected_final_response_contains": ["Boston"],
        "category": "memory_retention",
    },
    {
        "id": "AGENT_006",
        "name": "cot_consistency",
        "description": "Model chain-of-thought should be consistent with its conclusion",
        "user_request": (
            "Think step by step: If a train travels at 60mph and needs to cover 180 miles, "
            "how long will the journey take? Show your reasoning."
        ),
        "expected_answer_contains": ["3 hours", "3h", "180/60"],
        "category": "cot_consistency",
    },
    {
        "id": "AGENT_007",
        "name": "tool_param_correctness",
        "description": "Model should pass correct parameter types to tools",
        "user_request": "Check the stock price for Microsoft",
        "expected_tool": "get_stock_price",
        "expected_params_contain": ["MSFT"],
        "category": "tool_selection",
    },
]


# ─── Scoring functions ────────────────────────────────────────────────────────

def _build_tool_prompt(test_case: dict) -> str:
    tools_desc = "\n".join(
        f"- {name}: {info['description']} (params: {', '.join(info['params'])})"
        for name, info in AVAILABLE_TOOLS.items()
    )
    prompt = f"""You are an AI assistant with access to these tools:
{tools_desc}

When a task requires a tool, respond with JSON in this format:
{{"tool": "tool_name", "params": {{"param_name": "value"}}, "reasoning": "why you chose this tool"}}

If no tool is needed, respond normally in plain text.

User request: {test_case['user_request']}"""
    return prompt


async def score_tool_selection(client, test_case: dict) -> dict:
    """Score whether the model selected the correct tool."""
    prompt = _build_tool_prompt(test_case)

    try:
        response, _, _, latency = await client.generate(prompt)

        # Try to parse tool selection from response
        tool_used = None
        params_used = {}
        raw_json = None

        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                raw_json = json.loads(json_match.group())
                tool_used = raw_json.get("tool", "")
                params_used = raw_json.get("params", {})
            except Exception:
                pass

        expected_tool = test_case.get("expected_tool")
        should_not_use = test_case.get("should_not_use", [])

        # Tool selection accuracy
        tool_correct = (
            tool_used == expected_tool if expected_tool else
            tool_used not in should_not_use
        )

        # Parameter correctness
        expected_contains = test_case.get("expected_params_contain", [])
        params_str = json.dumps(params_used).lower()
        params_correct = all(
            p.lower() in params_str
            for p in expected_contains
        ) if expected_contains else True

        # Hallucination check
        hallucinated = tool_used in HALLUCINATED_TOOLS if tool_used else False

        return {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "tool_used": tool_used,
            "expected_tool": expected_tool,
            "tool_correct": tool_correct,
            "params_correct": params_correct,
            "hallucinated_tool": hallucinated,
            "latency_ms": round(latency, 1),
            "raw_response": response[:300],
        }
    except Exception as e:
        return {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "error": str(e),
            "tool_correct": False,
            "params_correct": False,
        }


async def score_memory_retention(client, test_case: dict) -> dict:
    """Score whether the model retains key facts across conversation turns."""
    turns = test_case.get("turns", [])
    expected_contains = test_case.get("expected_final_response_contains", [])
    last_response = ""

    try:
        for turn in turns:
            last_response, _, _, _ = await client.generate(turn)

        remembered = all(
            term.lower() in last_response.lower()
            for term in expected_contains
        )

        return {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "memory_retained": remembered,
            "expected_terms": expected_contains,
            "final_response": last_response[:200],
        }
    except Exception as e:
        return {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "error": str(e),
            "memory_retained": False,
        }


async def score_cot_consistency(client, test_case: dict) -> dict:
    """Score chain-of-thought reasoning consistency."""
    expected_contains = test_case.get("expected_answer_contains", [])

    try:
        response, _, _, latency = await client.generate(test_case["user_request"])
        response_lower = response.lower()

        answer_correct = any(
            term.lower() in response_lower
            for term in expected_contains
        )

        # Check if reasoning is present (CoT)
        has_reasoning = any(
            marker in response_lower
            for marker in ["step 1", "first,", "therefore", "so,", "thus", "because"]
        )

        return {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "answer_correct": answer_correct,
            "has_reasoning": has_reasoning,
            "cot_consistent": answer_correct and has_reasoning,
            "latency_ms": round(latency, 1),
            "response_snippet": response[:200],
        }
    except Exception as e:
        return {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "error": str(e),
            "answer_correct": False,
        }


async def score_planning(client, test_case: dict) -> dict:
    """Score multi-step planning accuracy."""
    expected_sequence = test_case.get("expected_tool_sequence", [])
    prompt = f"""You are given a complex task that requires multiple steps. List the tools you would use in order.
Available tools: {', '.join(AVAILABLE_TOOLS.keys())}

Task: {test_case['user_request']}

Respond with JSON: {{"plan": ["tool1", "tool2", ...], "reasoning": "explanation"}}"""

    try:
        response, _, _, latency = await client.generate(prompt)
        plan = []
        json_match = re.search(r'\{[^{}]*"plan"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                plan = data.get("plan", [])
            except Exception:
                pass

        # Check if all expected tools appear in the plan (order-flexible)
        tools_present = all(t in plan for t in expected_sequence)
        sequence_correct = plan[:len(expected_sequence)] == expected_sequence if plan else False

        return {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "planned_sequence": plan,
            "expected_sequence": expected_sequence,
            "tools_present": tools_present,
            "sequence_correct": sequence_correct,
            "planning_score": 1.0 if sequence_correct else 0.5 if tools_present else 0.0,
            "latency_ms": round(latency, 1),
        }
    except Exception as e:
        return {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "error": str(e),
            "planning_score": 0.0,
        }


# ─── Run full agentic evaluation ─────────────────────────────────────────────

async def run_agentic_eval(client, config: Optional[dict] = None) -> dict:
    """Run all agentic test cases and return aggregate report."""
    print("\n[Agentic Eval] Running agentic evaluation suite...")
    results = []

    for tc in AGENTIC_TEST_CASES:
        cat = tc["category"]
        try:
            if cat in ("tool_selection", "hallucination_prevention"):
                r = await score_tool_selection(client, tc)
            elif cat == "memory_retention":
                r = await score_memory_retention(client, tc)
            elif cat == "cot_consistency":
                r = await score_cot_consistency(client, tc)
            elif cat == "planning":
                r = await score_planning(client, tc)
            else:
                continue
            results.append(r)
            status = "✓" if not r.get("error") else "✗"
            print(f"  {status} {tc['id']}: {tc['name']}")
        except Exception as e:
            results.append({"test_id": tc["id"], "category": cat, "error": str(e)})
            print(f"  ✗ {tc['id']}: ERROR — {e}")

    # Aggregate metrics
    tool_results = [r for r in results if r.get("category") == "tool_selection"]
    planning_results = [r for r in results if r.get("category") == "planning"]
    memory_results = [r for r in results if r.get("category") == "memory_retention"]
    cot_results = [r for r in results if r.get("category") == "cot_consistency"]
    halluc_results = [r for r in results if r.get("category") == "hallucination_prevention"]

    def safe_mean(lst, key):
        vals = [r[key] for r in lst if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "tool_selection_accuracy": safe_mean(tool_results, "tool_correct"),
        "param_correctness": safe_mean(tool_results, "params_correct"),
        "planning_success_rate": safe_mean(planning_results, "planning_score"),
        "memory_retention_rate": safe_mean(memory_results, "memory_retained"),
        "cot_consistency_rate": safe_mean(cot_results, "cot_consistent"),
        "hallucination_free_rate": (
            sum(1 for r in halluc_results if not r.get("hallucinated_tool", True)) /
            max(len(halluc_results), 1)
        ),
        "total_tests": len(AGENTIC_TEST_CASES),
        "errors": sum(1 for r in results if r.get("error")),
    }

    print(f"\n[Agentic Eval] Summary:")
    for k, v in summary.items():
        if v is not None and isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")

    return {"summary": summary, "results": results}
