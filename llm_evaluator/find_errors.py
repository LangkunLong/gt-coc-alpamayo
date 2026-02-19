import pandas as pd

# group all the failed clips with PARSE_ERROR in the parsed_ontology.csv and get their corresponding reasoning traces from physical_ai_inference_results.csv
parsed_ontology = pd.read_csv('/home/larry/alpamayo/coc_ontology/gt-coc-alpamayo/llm_evaluator/parsed_ontology.csv')
inference_results = pd.read_csv('/home/larry/alpamayo/coc_ontology/gt-coc-alpamayo/coc_traces/physical_ai_inference_results.csv')
parse_error_rows = parsed_ontology[parsed_ontology.apply(lambda row: row.astype(str).str.contains('PARSE_ERROR').any(), axis=1)]


error_clip_ids = parse_error_rows['clip_id'].tolist()
error_traces = inference_results[inference_results['clip_id'].isin(error_clip_ids)][['clip_id', 'reasoning_trace']]
result = parse_error_rows[['clip_id']].merge(error_traces, on='clip_id', how='left')

output_path = '/home/larry/alpamayo/coc_ontology/gt-coc-alpamayo/llm_evaluator/parse_errors_with_traces.csv'
result.to_csv(output_path, index=False)

print(f"Found {len(result)} clips with parse errors")
print(f"Results saved to: {output_path}")
print("\nClips with parse errors:")
print(result)