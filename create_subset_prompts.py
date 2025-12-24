import json

# Load runs
with open('creative-writing-bench/creative_bench_runs.json', 'r') as f:
    runs = json.load(f)

# Find tinker run
tinker_run = next((v for k,v in runs.items() if 'tinker' in k), None)
if not tinker_run:
    print("Tinker run not found")
    exit(1)

completed_ids = list(tinker_run['creative_tasks']['1'].keys())
print(f"Found {len(completed_ids)} completed prompts")

# Load all prompts
with open('creative-writing-bench/data/creative_writing_prompts_v3.json', 'r') as f:
    all_prompts = json.load(f)

# Filter
subset_prompts = {k: v for k, v in all_prompts.items() if k in completed_ids}
print(f"Subset has {len(subset_prompts)} prompts")

# Save
with open('creative-writing-bench/data/temp_prompts.json', 'w') as f:
    json.dump(subset_prompts, f, indent=2)
