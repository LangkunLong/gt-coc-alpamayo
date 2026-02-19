"""
Closed ontology schema for CoC reasoning trace evaluation.
All field values are defined here as the single source of truth.

Includes JSON Schema definitions for structured output enforcement
via Anthropic tool_use and OpenAI json_schema response_format.
"""

# based on table 1 and table 2 from alapamayo paper
SCHEMA = {
    "maneuver_lat": [
        "lane_keeping", "merge_split", "out_of_lane_nudge", "in_lane_nudge",
        "lane_change_left", "lane_change_right", "pull_over",
        "turn_left", "turn_right", "u_turn", "lateral_maneuver_abort", "none"
    ],
    "maneuver_long": [
        "set_speed_tracking", "lead_obstacle_following", "speed_adaptation",
        "gap_searching", "acceleration_for_passing", "yield",
        "stop_for_static_constraints", "none"
    ],
    "traffic_light": [
        "green", "yellow", "red", "flashing",
        "arrow_green", "arrow_red", "none"
    ],
    "traffic_control": [
        "stop_sign", "yield_sign", "pedestrian_crossing", "school_zone",
        "rail_crossing", "speed_limit", "no_entry", "one_way",
        "right_of_way", "construction_sign", "lane_control_sign", "none"
    ],
    "road_geometry": [
        "straight", "curve", "grade", "speed_bump", "narrowing",
        "roundabout", "ramp", "intersection", "none"
    ],
    "lane_config": [
        "lane_merge", "lane_split", "lane_exit", "lane_end",
        "bike_lane", "shoulder", "none"
    ],
    "odd_condition": [
        "rain", "snow", "fog", "night", "construction",
        "emergency_vehicle", "school_bus", "none"
    ],
    "routing_intent": [
        "go_straight", "turn_left", "turn_right", "lane_change_needed",
        "merge_needed", "exit_needed", "none"
    ],
    # default to vehilce, if no cars present, categorize to none
    "agent_type": [
        "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian",
        "scooter", "emergency_vehicle", "vehicle", "none"
    ],
    "relative_position": [
        "in_path_ahead", "in_path_behind", "left_adjacent", "right_adjacent",
        "oncoming", "crossing_path", "roadside", "none"
    ],
    "lane_relation": [
        "same_lane", "left_lane", "right_lane", "oncoming_lane",
        "cross_traffic", "off_road", "none"
    ],
    "agent_behavior": [
        "stationary", "decelerating", "accelerating", "constant_speed",
        "crossing_path", "cutting_in", "lane_changing", "weaving",
        "backing_out", "encroaching", "yielding_to_ego", "blocking_lane",
        "unknown", "none"
    ],
}

# --------------------------------------------------------------------------
# Table 1 & Table 2 definitions for richer prompt context
# --------------------------------------------------------------------------

TABLE1_DEFINITIONS = {
    "maneuver_long": {
        "set_speed_tracking": "Maintain or reach a target speed when unconstrained; excludes follow/yield/stop logic.",
        "lead_obstacle_following": "Maintain a safe time gap to the lead entity (closest in-path entity moves in the same traffic flow); excludes geometry-based slowing, gap-matching, and yielding to non-lead entity.",
        "speed_adaptation": "Adjust speed for roadway features (curves, grades, bumps, ramps, roundabouts, turns); independent of a lead.",
        "gap_searching": "Adjust speed to match the target stream or create a usable gap to support a planned lateral maneuver.",
        "acceleration_for_passing": "Increase speed to pass a slower lead with an associated lateral plan.",
        "yield": "Slow/stop to concede priority to specific agents (pedestrians, cross-traffic, emergency vehicles, cut-ins).",
        "stop_for_static_constraints": "Decelerate to and hold at control points (stop/yield lines, red light, school bus/rail rules).",
    },
    "maneuver_lat": {
        "lane_keeping": "Maintain position within lane boundaries; minor in-lane offsets allowed; never cross lane lines.",
        "merge_split": "Transition between facilities (e.g., on-ramp to mainline, weave segments); not a same-road lane change.",
        "out_of_lane_nudge": "Brief, intentional lane-line crossing to increase clearance around a blockage/hazard; return to original lane.",
        "in_lane_nudge": "Temporary offset within the lane (no line crossing) to increase clearance around a blockage/hazard.",
        "lane_change_left": "Full adjacent-lane transition with gap negotiation to the left.",
        "lane_change_right": "Full adjacent-lane transition with gap negotiation to the right.",
        "pull_over": "Move toward edge/shoulder or a designated stop area (pickup, emergency stop, parking approach).",
        "turn_left": "Planned path onto a different road segment with a significant heading change to the left.",
        "turn_right": "Planned path onto a different road segment with a significant heading change to the right.",
        "u_turn": "Planned path reversing the direction of travel.",
        "lateral_maneuver_abort": "Cancel an ongoing lateral maneuver (nudge, lane change, merge/split, pull-over) and re-center when safe.",
    },
}

TABLE2_DEFINITIONS = {
    "traffic_light": "Traffic signal state including current color (R/Y/G), arrow state, and visibility/occlusion.",
    "traffic_control": "Regulatory signage and infrastructure: stop/yield signs, pedestrian crossings, school zones, rail crossings, speed limits, lane control signs, construction signs.",
    "road_geometry": "Roadway geometric features: curvature/grade, speed bumps, narrowing, roundabouts, ramps, intersections.",
    "lane_config": "Lane configuration: lane count changes, merges, splits, exits, lane ends, bike lanes, shoulders.",
    "odd_condition": "Operational Design Domain constraints: weather/visibility (rain, snow, fog), time of day (night), construction zones, presence of emergency vehicles or school buses.",
    "routing_intent": "Ego vehicle's navigational goal: target lane/turn direction, near-term split/merge requirement.",
    "agent_type": "Type of other road agent: vehicle (car, truck, bus, motorcycle), vulnerable road user (pedestrian, cyclist, scooter), or emergency vehicle.",
    "relative_position": "Spatial position of the agent relative to the ego vehicle: in-path ahead/behind, adjacent left/right, oncoming, crossing path, or roadside.",
    "lane_relation": "Lane relationship of the agent relative to the ego vehicle: same lane, adjacent lanes, oncoming lane, cross traffic, or off-road.",
    "agent_behavior": "Observable motion behavior of the agent: stationary, decelerating, accelerating, constant speed, crossing path, cutting in, lane changing, weaving, backing out, encroaching into ego's lane, yielding to ego, or blocking the lane.",
}

# --------------------------------------------------------------------------
# Field groupings
# --------------------------------------------------------------------------

ENVIRONMENT_FIELDS = [
    "traffic_light", "traffic_control", "road_geometry",
    "lane_config", "odd_condition", "routing_intent"
]

AGENT_FIELDS = [
    "agent_type", "relative_position", "lane_relation", "agent_behavior"
]

MANEUVER_FIELDS = [
    "maneuver_lat", "maneuver_long"
]

ALL_FIELDS = (
    ["clip_id"]
    + MANEUVER_FIELDS
    + ENVIRONMENT_FIELDS
    + AGENT_FIELDS
    + ["confidence", "confidence_notes"]
)


# --------------------------------------------------------------------------
# JSON Schema for structured output enforcement
# --------------------------------------------------------------------------

def _build_row_schema() -> dict:
    """Build JSON Schema for a single ontology row."""
    properties = {}
    required = []
    for field, values in SCHEMA.items():
        properties[field] = {
            "type": "string",
            "description": (
                f"Comma-separated values from: {', '.join(values)}. "
                f"Use 'none' if not applicable."
            ),
        }
        required.append(field)

    # Confidence fields
    properties["confidence"] = {
        "type": "string",
        "enum": ["high", "medium", "low"],
        "description": "Parsing confidence level.",
    }
    required.append("confidence")

    properties["confidence_notes"] = {
        "type": "string",
        "description": "Explanation for medium/low confidence. Empty string if high.",
    }
    required.append("confidence_notes")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def get_output_json_schema() -> dict:
    """Full JSON Schema for the LLM output (array of rows)."""
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": _build_row_schema(),
                "minItems": 1,
                "description": (
                    "One row per causal agent. If no agent is involved, "
                    "return one row with agent fields set to 'none'."
                ),
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    }


def get_anthropic_tool_definition() -> dict:
    """
    Anthropic tool_use definition. By defining the output as a 'tool call',
    we force the model to return structured JSON matching our schema.
    """
    return {
        "name": "submit_ontology_parse",
        "description": (
            "Submit the parsed ontology for a CoC reasoning trace. "
            "Call this tool exactly once with the parsed result."
        ),
        "input_schema": get_output_json_schema(),
    }


def get_openai_response_format() -> dict:
    """
    OpenAI structured output response_format definition.
    Forces the model to return JSON conforming to the schema.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ontology_parse",
            "strict": True,
            "schema": get_output_json_schema(),
        },
    }