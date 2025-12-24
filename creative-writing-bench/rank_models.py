import json

with open('elo_results.json', 'r') as f:
    data = json.load(f)

# Extract model names and elo_norm scores
models = []
for model_name, stats in data.items():
    if 'elo_norm' in stats:
        models.append({
            'name': model_name,
            'elo_norm': stats['elo_norm'],
            'elo': stats.get('elo', 0),
            'score': stats.get('creative_writing_rubric_score_agg', 0),
            'ci_low': stats.get('ci_low_norm', 0),
            'ci_high': stats.get('ci_high_norm', 0)
        })

# Sort by normalized ELO (descending)
models.sort(key=lambda x: x['elo_norm'], reverse=True)

# Print ranking
print(f"{'Rank':<6} {'Model':<65} {'ELO Norm':<12} {'95% CI':<20} {'Score':<10}")
print('=' * 115)
for i, model in enumerate(models, 1):
    ci_range = f"[{model['ci_low']:.1f}, {model['ci_high']:.1f}]"
    print(f"{i:<6} {model['name']:<65} {model['elo_norm']:<12.2f} {ci_range:<20} {model['score']:<10.2f}")
