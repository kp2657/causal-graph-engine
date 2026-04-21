"""
agent_messages.py — Inter-agent communication contracts for Phase O.

Defines structured message types for:
  - ReDelegationInstruction: reviewer → specific agent, with issues + targeted instruction
  - AgentFeedback: full feedback package from reviewer, keyed by agent
  - DownstreamAdjustment: chemistry/clinical → target_prioritization score signals

Design principles:
  - Plain dataclasses (not Pydantic) — fast, no validation overhead
  - Serialisable to dict for inclusion in AgentInput.upstream_results
  - AgentFeedback.for_agent() returns the instruction most relevant to an agent
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReDelegationInstruction:
    """
    Targeted instruction from the scientific reviewer to a specific agent.

    An agent receiving this as `_reviewer_feedback` in its upstream_results
    should prioritise resolving the listed issues before returning.
    """
    agent_name:  str
    priority:    str               # "CRITICAL" | "MAJOR"
    issues:      list[str]         # human-readable issue descriptions
    instruction: str               # single actionable sentence for the agent
    context:     dict = field(default_factory=dict)  # gene names, check ids, etc.

    def to_dict(self) -> dict:
        return {
            "agent_name":  self.agent_name,
            "priority":    self.priority,
            "issues":      self.issues,
            "instruction": self.instruction,
            "context":     self.context,
        }


@dataclass
class AgentFeedback:
    """
    Full feedback package assembled after the reviewer runs.

    Attached to re-delegation AgentInput calls so the target agent
    knows exactly what to fix.
    """
    run_id:           str
    reviewer_verdict: str                         # "APPROVE" | "REVISE"
    instructions:     list[ReDelegationInstruction] = field(default_factory=list)

    def for_agent(self, agent_name: str) -> ReDelegationInstruction | None:
        """Return the instruction for a specific agent, or None."""
        for instr in self.instructions:
            if instr.agent_name == agent_name:
                return instr
        return None

    def has_critical(self) -> bool:
        return any(i.priority == "CRITICAL" for i in self.instructions)

    def agents_to_revisit(self) -> list[str]:
        """Ordered list of agents with CRITICAL first, then MAJOR."""
        critical = [i.agent_name for i in self.instructions if i.priority == "CRITICAL"]
        major    = [i.agent_name for i in self.instructions if i.priority == "MAJOR"]
        seen: set[str] = set()
        result: list[str] = []
        for name in critical + major:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def to_dict(self) -> dict:
        return {
            "run_id":           self.run_id,
            "reviewer_verdict": self.reviewer_verdict,
            "instructions":     [i.to_dict() for i in self.instructions],
        }


def build_feedback_from_reviewer(reviewer_result: dict, run_id: str) -> AgentFeedback:
    """
    Parse a scientific_reviewer_agent result into a structured AgentFeedback.

    Groups issues by agent_to_revisit and generates a per-agent instruction
    summarising what needs to be fixed.
    """
    issues = reviewer_result.get("issues", [])
    verdict = reviewer_result.get("verdict", "APPROVE")

    # Group CRITICAL + MAJOR issues by responsible agent
    agent_issues: dict[str, list[dict]] = {}
    for issue in issues:
        if issue.get("severity") not in ("CRITICAL", "MAJOR"):
            continue
        agent = issue.get("agent_to_revisit")
        if not agent:
            continue
        agent_issues.setdefault(agent, []).append(issue)

    instructions: list[ReDelegationInstruction] = []
    for agent_name, agent_issue_list in agent_issues.items():
        priority = (
            "CRITICAL"
            if any(i["severity"] == "CRITICAL" for i in agent_issue_list)
            else "MAJOR"
        )
        issue_texts = [i["description"] for i in agent_issue_list]
        checks = [i["check"] for i in agent_issue_list]
        genes  = [i["gene"] for i in agent_issue_list if i.get("gene")]

        instruction = _generate_instruction(agent_name, checks, genes, agent_issue_list)

        instructions.append(ReDelegationInstruction(
            agent_name=agent_name,
            priority=priority,
            issues=issue_texts,
            instruction=instruction,
            context={"checks": checks, "genes": genes},
        ))

    return AgentFeedback(
        run_id=run_id,
        reviewer_verdict=verdict,
        instructions=instructions,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_instruction(
    agent_name: str,
    checks: list[str],
    genes: list[str],
    issues: list[dict],
) -> str:
    """Generate a targeted, actionable instruction for a specific agent."""
    gene_str = f" for genes: {', '.join(genes[:5])}" if genes else ""

    if agent_name == "causal_discovery_agent":
        if any("anchor_recovery" in c for c in checks):
            return (
                f"Anchor recovery is below 80%. Investigate why required anchor edges "
                f"are missing{gene_str}. "
                "Use run_python to sweep tissues for missing β values and retry with "
                "lower gamma threshold for anchor genes only."
            )
        return (
            f"Re-examine edge selection{gene_str}. "
            "Check that all SCONE-rejected edges were correctly excluded and "
            "that zero-gamma edges are filtered."
        )

    if agent_name == "perturbation_genomics_agent":
        return (
            f"Provisional_virtual β found for top targets{gene_str}. "
            "Retry Tier 1 and Tier 2 lookups with additional Perturb-seq datasets "
            "and alternative tissues. Document what you tried."
        )

    if agent_name == "chemistry_agent":
        return (
            f"GPS reversal / putative-target normalization needs revision{gene_str}. "
            "Re-run GPS disease/program screen and ensure inferred putative targets are "
            "normalized to HGNC; document any unmapped labels."
        )

    if agent_name == "scientific_writer_agent":
        return (
            "Revise the report to address reviewer issues: "
            + "; ".join(i["description"][:80] for i in issues[:3])
        )

    return (
        f"Address the following issues{gene_str}: "
        + "; ".join(i["description"][:80] for i in issues[:2])
    )
