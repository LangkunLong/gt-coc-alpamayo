"""
Prompt templates for the LLM-based CoC reasoning trace evaluator.
Includes full Table 1 (maneuver definitions) and Table 2 (causal factor
definitions) from the Alpamayo-R1 paper for disambiguation.
"""

from schema import SCHEMA, TABLE1_DEFINITIONS, TABLE2_DEFINITIONS


def _format_maneuver_definitions() -> str:
    """Format Table 1 definitions into a readable block."""
    lines = []
    lines.append("### Longitudinal Maneuvers (maneuver_long)")
    lines.append("Use these definitions to disambiguate similar maneuvers:\n")
    for value, defn in TABLE1_DEFINITIONS["maneuver_long"].items():
        lines.append(f"  - **{value}**: {defn}")

    lines.append("\n### Lateral Maneuvers (maneuver_lat)")
    lines.append("Use these definitions to disambiguate similar maneuvers:\n")
    for value, defn in TABLE1_DEFINITIONS["maneuver_lat"].items():
        lines.append(f"  - **{value}**: {defn}")

    return "\n".join(lines)


def _format_causal_factor_definitions() -> str:
    """Format Table 2 definitions into a readable block."""
    lines = []
    lines.append("### Causal Factor Categories")
    lines.append(
        "Use these definitions to correctly categorize causal factors "
        "from the reasoning trace:\n"
    )
    for field, defn in TABLE2_DEFINITIONS.items():
        lines.append(f"  - **{field}**: {defn}")

    return "\n".join(lines)


def _format_vocabulary_block() -> str:
    """Format all closed vocabularies."""
    return "\n".join(
        f"  {field}: [{', '.join(values)}]"
        for field, values in SCHEMA.items()
    )


def build_system_prompt() -> str:
    """Build the system prompt with full ontology definition and domain context."""

    maneuver_defs = _format_maneuver_definitions()
    causal_defs = _format_causal_factor_definitions()
    vocab_block = _format_vocabulary_block()

    return f"""You are an expert autonomous driving reasoning evaluator. Your task is to parse a Chain of Causation (CoC) reasoning trace into a structured ontology that decomposes the ego vehicle's driving decision into its maneuver, environmental context, and causal agents.

## DOMAIN CONTEXT

A CoC reasoning trace describes:
1. What the ego vehicle is doing (maneuver) — the lateral and longitudinal action.
2. Why it is doing it (causal factors) — the environmental conditions and/or other agent behaviors that caused the decision.

The trace follows a cause-and-effect structure: observable evidence → driving decision.

## MANEUVER DEFINITIONS (from Alpamayo-R1 Table 1)

These are precise definitions for each maneuver. Use them to disambiguate similar-sounding actions.

{maneuver_defs}

### Key disambiguation rules:
- "Slow for a lead vehicle" → **lead_obstacle_following** (maintaining gap to in-path lead)
- "Slow for a red light" → **stop_for_static_constraints** (decelerating to a control point)
- "Slow for a pedestrian crossing" → **yield** (conceding priority to a specific agent)
- "Slow for a curve" → **speed_adaptation** (adjusting for road geometry)
- "Slow to create a gap for lane change" → **gap_searching** (supporting a lateral maneuver)
- "Nudge within lane" → **in_lane_nudge** (no lane line crossing)
- "Nudge over the line" → **out_of_lane_nudge** (crosses lane line, returns to lane)
- "Change lane" → **lane_change_left/right** (full transition to adjacent lane)
- "Merge onto highway" → **merge_split** (facility change, not same-road lane change)

## CAUSAL FACTOR DEFINITIONS (from Alpamayo-R1 Table 2)

These define what each environmental and agent field captures.

{causal_defs}

### Key categorization rules:
- A red/green/yellow light → **traffic_light** (not traffic_control)
- A stop sign, yield sign, crosswalk markings → **traffic_control**
- Rain, fog, night → **odd_condition**
- Construction zone (physical presence) → **odd_condition: construction**
- Construction warning sign → **traffic_control: construction_sign**
- Road curves, bumps, hills → **road_geometry**
- Lane ending, merging, splitting → **lane_config**
- "Need to turn left ahead" → **routing_intent: turn_left**

## CLOSED VOCABULARIES

Each field accepts ONLY the values listed below. Use comma-separated values for multiple. Use "none" if not referenced or not applicable.

{vocab_block}

## PARSING RULES

1. **Maneuvers:** Identify the ego vehicle's lateral and longitudinal maneuvers. A trace always has at least one. Use "none" only if that axis is genuinely not referenced.

2. **Environment fields:** Extract environmental and regulatory context. Use "none" if not mentioned.

3. **Agent fields:** Extract information about OTHER agents (not ego) that causally influence the decision.
   - If MULTIPLE agents are referenced, create SEPARATE rows for each agent.
   - If NO agent is referenced (purely environmental cause), set all agent fields to "none".
   - Maneuver and environment fields are REPEATED across rows (they describe the same ego event).

4. **Multi-value:** If a field has multiple applicable values, combine with commas: "night,rain".

5. **Confidence:**
   - **high**: All fields clearly map to vocabulary, no ambiguity.
   - **medium**: Most fields clear, 1-2 required best-effort interpretation.
   - **low**: Significant ambiguity, multiple fields required guessing.
   Provide notes explaining any non-high confidence.

## IMPORTANT

- Parse ONLY what is explicitly stated or directly implied in the trace.
- Do NOT hallucinate causal factors not mentioned in the trace.
- When in doubt between two maneuvers, use the definitions above to decide.
- "none" means "not mentioned or not applicable", not "normal conditions"."""


def build_user_prompt(clip_id: str, reasoning_trace: str) -> str:
    """Build the user prompt for a single reasoning trace."""
    return (
        f"Parse the following CoC reasoning trace into the ontology.\n\n"
        f"Clip ID: {clip_id}\n"
        f"Reasoning Trace: \"{reasoning_trace}\"\n\n"
        f"Call the submit_ontology_parse tool with the parsed result."
    )


def build_user_prompt_openai(clip_id: str, reasoning_trace: str) -> str:
    """Build the user prompt for OpenAI (no tool call instruction)."""
    return (
        f"Parse the following CoC reasoning trace into the ontology.\n\n"
        f"Clip ID: {clip_id}\n"
        f"Reasoning Trace: \"{reasoning_trace}\"\n\n"
        f"Return the parsed JSON."
    )