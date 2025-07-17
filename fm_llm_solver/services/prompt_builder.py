"""
Prompt builder for FM-LLM Solver.

Constructs prompts for LLM generation and verification tasks.
"""

from typing import List, Optional, Dict, Any

from fm_llm_solver.core.types import SystemDescription, RAGDocument, SystemType


class PromptBuilder:
    """Builds prompts for various tasks."""

    def build_generation_prompt(
        self,
        system: SystemDescription,
        context: Optional[List[RAGDocument]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a prompt for barrier certificate generation.

        Args:
            system: System description
            context: Optional RAG context documents
            examples: Optional few-shot examples

        Returns:
            Complete prompt string
        """
        parts = []

        # System instruction
        parts.append(self._get_system_instruction(system.system_type))

        # Add context if available
        if context:
            parts.append(self._format_context(context))

        # Add examples if available
        if examples:
            parts.append(self._format_examples(examples))

        # Add the actual system
        parts.append(self._format_system(system))

        # Add generation instruction
        parts.append(self._get_generation_instruction(system))

        return "\n\n".join(parts)

    def _get_system_instruction(self, system_type: SystemType) -> str:
        """Get the system instruction based on system type."""
        base = """You are an expert in dynamical systems and control theory, specializing in barrier certificate generation.
Your task is to generate a barrier certificate that proves safety for the given dynamical system."""

        if system_type == SystemType.CONTINUOUS:
            return (
                base
                + """
For continuous-time systems, the barrier certificate B(x) must satisfy:
1. B(x) > 0 for all x in the initial set
2. B(x) ≤ 0 for all x in the unsafe set
3. The Lie derivative L_f B(x) ≤ 0 for all x where B(x) = 0"""
            )

        elif system_type == SystemType.DISCRETE:
            return (
                base
                + """
For discrete-time systems, the barrier certificate B(x) must satisfy:
1. B(x) > 0 for all x in the initial set
2. B(x) ≤ 0 for all x in the unsafe set
3. B(f(x)) - B(x) ≤ 0 for all x in the state space"""
            )

        elif system_type == SystemType.STOCHASTIC:
            return (
                base
                + """
For stochastic systems, the barrier certificate B(x) must satisfy:
1. B(x) > c for all x in the initial set (c > 0)
2. B(x) ≤ 0 for all x in the unsafe set
3. The infinitesimal generator L B(x) ≤ -αB(x) for some α > 0"""
            )

        else:
            return base

    def _format_context(self, context: List[RAGDocument]) -> str:
        """Format RAG context documents."""
        lines = ["## Relevant Context\n"]

        for i, doc in enumerate(context, 1):
            lines.append(f"### Context {i} (Relevance: {doc.score:.2f})")
            lines.append(f"Source: {doc.source}")
            lines.append(f"{doc.content}")
            lines.append("")

        return "\n".join(lines)

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format few-shot examples."""
        lines = ["## Examples\n"]

        for i, example in enumerate(examples, 1):
            lines.append(f"### Example {i}")
            lines.append(f"System: {example.get('system', 'N/A')}")
            lines.append(f"Certificate: {example.get('certificate', 'N/A')}")
            lines.append(f"Explanation: {example.get('explanation', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    def _format_system(self, system: SystemDescription) -> str:
        """Format the system description."""
        lines = ["## System to Analyze\n"]

        # Dynamics
        if system.system_type == SystemType.CONTINUOUS:
            lines.append("### Dynamics (Continuous-time)")
            for var, expr in system.dynamics.items():
                lines.append(f"d{var}/dt = {expr}")
        elif system.system_type == SystemType.DISCRETE:
            lines.append("### Dynamics (Discrete-time)")
            for var, expr in system.dynamics.items():
                lines.append(f"{var}[k+1] = {expr}")
        else:
            lines.append("### Dynamics")
            for var, expr in system.dynamics.items():
                lines.append(f"{var}: {expr}")

        lines.append("")

        # Sets
        lines.append(f"### Initial Set: {system.initial_set}")
        lines.append(f"### Unsafe Set: {system.unsafe_set}")

        # Domain bounds
        if system.domain_bounds:
            lines.append("\n### Domain Bounds")
            for var, (low, high) in system.domain_bounds.bounds.items():
                lines.append(f"{var} ∈ [{low}, {high}]")

        return "\n".join(lines)

    def _get_generation_instruction(self, system: SystemDescription) -> str:
        """Get the generation instruction."""
        variables = list(system.dynamics.keys())
        var_str = ", ".join(variables)

        return """## Task

Generate a barrier certificate B({var_str}) for the system above that proves safety (the system starting in the initial set will never reach the unsafe set).

Requirements:
1. The certificate should be a polynomial function of the state variables
2. Provide ONLY the mathematical expression, no explanation
3. Use standard mathematical notation
4. The expression should be valid Python/SymPy syntax

Your response should be in the format:
B({var_str}) = <expression>"""

    def build_verification_prompt(
        self, system: SystemDescription, certificate: str, check_type: str
    ) -> str:
        """
        Build a prompt for verification assistance.

        Args:
            system: System description
            certificate: Barrier certificate expression
            check_type: Type of check being performed

        Returns:
            Verification prompt
        """
        parts = [
            "You are an expert in dynamical systems verification.",
            f"System: {self._format_system(system)}",
            f"Proposed Certificate: B = {certificate}",
            f"\nVerify that this certificate satisfies the {check_type} condition.",
            "Provide a step-by-step analysis.",
        ]

        return "\n\n".join(parts)
