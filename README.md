# Game-Theoretic Analysis of Chain of Causation Reasoning in Nvidia Alpamayo 
## Parsing CoC Reasoning Trace → Specified Ontology

Parses Alpamayo-R1 Chain of Causation (CoC) reasoning traces into a structured
driving ontology using an LLM with guaranteed structured output.

## Structured Output

**OpenAI (json_schema):** Uses `response_format` with `strict: True` to constrain
token generation so only valid schema-conforming JSON is produced.

## Setup

```bash
sudo apt update && sudo apt install python3-venv -y
python3 -m venv openai-env
source openai-env/bin/activate
pip install openai  
```

## Usage

```bash
python run_pipeline.py 
```

## Config file arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Input CSV path |
| `--output` | `parsed_ontology.csv` | Output CSV path |
| `--provider` | `anthropic` | `anthropic` or `openai` |
| `--model` | `claude-sonnet-4-20250514` | Model name |
| `--api-key` | env var | API key |
| `--id-column` | `clip_id` | Clip ID column name |
| `--trace-column` | `reasoning_trace` | Trace text column name |
| `--concurrency` | `5` | Max parallel API requests |
| `--max-retries` | `3` | Retries per trace on failure |

## Input Format

CSV with at minimum two columns:

```csv
clip_id,reasoning_trace
001,"Keep distance to the lead vehicle because it is slowing for the red light."
002,"Nudge left over the line to increase clearance from the cyclists on the right."
```

## Output Schema

Each trace produces one or more rows (one per causal agent, or one row with
agent fields set to `none` for purely environmental causes).

| Field | Description |
|-------|-------------|
| `clip_id` | Clip identifier |
| `maneuver_lat` | Ego lateral maneuver |
| `maneuver_long` | Ego longitudinal maneuver |
| `traffic_light` | Traffic light state |
| `traffic_control` | Regulatory signage/infrastructure |
| `road_geometry` | Road geometry features |
| `lane_config` | Lane configuration |
| `odd_condition` | ODD conditions |
| `routing_intent` | Ego routing intent |
| `agent_type` | Other agent type |
| `relative_position` | Agent position relative to ego |
| `lane_relation` | Agent lane relative to ego |
| `agent_behavior` | Observed agent behavior |
| `confidence` | high / medium / low |
| `confidence_notes` | Explanation for non-high confidence |

All enum fields accept comma-separated values. See `schema.py` for vocabularies.

## Files

```
schema.py          Closed vocabularies, Table 1/2 definitions, JSON schema for tool_use
rag.py         System prompt with maneuver definitions and disambiguation rules
run.py    Main pipeline: async concurrency, structured output, validation
```
