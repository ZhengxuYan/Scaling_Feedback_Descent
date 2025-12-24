import json
import matplotlib.pyplot as plt
import numpy as np

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Read the data
with open('elo_results.json', 'r') as f:
    data = json.load(f)

# Extract model names and elo_norm scores
models = []
for model_name, stats in data.items():
    if 'elo_norm' in stats:
        models.append({
            'name': model_name,
            'elo_norm': stats['elo_norm'],
            'ci_low': stats.get('ci_low_norm', 0),
            'ci_high': stats.get('ci_high_norm', 0)
        })

# Sort by normalized ELO (descending)
models.sort(key=lambda x: x['elo_norm'], reverse=True)

# Take top 10 models to avoid overlapping
top_n = 10
top_models = models[:top_n]

# Prepare data for plotting
names = [m['name'] for m in top_models]
elo_scores = [m['elo_norm'] for m in top_models]
ci_lows = [m['ci_low'] for m in top_models]
ci_highs = [m['ci_high'] for m in top_models]

# Calculate error bars
errors_low = [elo - low if low > 0 else 0 for elo, low in zip(elo_scores, ci_lows)]
errors_high = [high - elo if high > 0 else 0 for elo, high in zip(elo_scores, ci_highs)]

# Clean model names
clean_names = []
for name in names:
    # Remove common prefixes
    short = name.replace('openrouter/', '').replace('anthropic/', '').replace('google/', '')
    short = short.replace('deepseek/', '').replace('mistralai/', '').replace('meta-llama/', '')
    short = short.replace('openai/', '').replace('moonshotai/', '').replace('tinker:', '').replace('z-ai/', 'ZhipuAI ')
    
    # Special handling for common models
    if 'claude-sonnet-4.5' in short:
        short = 'Claude Sonnet 4.5'
    elif 'claude-opus-4' in short:
        short = 'Claude Opus 4'
    elif 'gpt-5-' in short:
        short = short.replace('gpt-5-2025-08-07', 'GPT-5')
    elif 'qwen-235b-critic' in short or 'qwen3-235b' in short:
        short = 'Qwen 235B Critic' if 'critic' in short else 'Qwen 235B'
    
    # Truncate if still too long
    if len(short) > 40:
        short = short[:37] + '...'
    clean_names.append(short)

# Create figure with much larger height to prevent overlap
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f8f9fa')

# Define colors
tinker_color = '#e74c3c'  # Red for Tinker
top3_color = '#3498db'     # Blue for top 3
other_color = '#95a5a6'    # Gray for others

# Create colors list
colors = []
for i, name in enumerate(names):
    if 'tinker' in name.lower() or 'qwen-235b-critic' in name.lower():
        colors.append(tinker_color)
    elif i < 3:
        colors.append(top3_color)
    else:
        colors.append(other_color)

# Create horizontal bar chart with larger bars
y_pos = np.arange(len(clean_names))
bars = ax.barh(y_pos, elo_scores, 
               height=0.85,
               color=colors, 
               alpha=0.85,
               edgecolor='white',
               linewidth=2)

# Add error bars separately for better control
ax.errorbar(elo_scores, y_pos, 
            xerr=[errors_low, errors_high],
            fmt='none',
            ecolor='black',
            alpha=0.4,
            capsize=4,
            capthick=1.5,
            zorder=10)

# Customize plot
ax.set_yticks(y_pos)
ax.set_yticklabels(clean_names, fontsize=15, fontweight='500')
ax.invert_yaxis()  # Highest score at the top
ax.set_xlabel('Normalized ELO Rating', fontsize=17, fontweight='bold', labelpad=10)
ax.set_title('Top 10 Models - Creative Writing Benchmark (Normalized ELO)', 
             fontsize=20, fontweight='bold', pad=25, color='#2c3e50')

# Customize grid
ax.grid(axis='x', alpha=0.25, linestyle='-', linewidth=0.8, color='#7f8c8d')
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Add value labels on bars with better formatting
for i, (score, name, color, ci_high) in enumerate(zip(elo_scores, names, colors, ci_highs)):
    rank = i + 1
    
    # Position text intelligently
    text_x = max(score, ci_high) + 15
    
    # Choose text color based on background
    text_color = color if color == tinker_color else '#2c3e50'
    
    # Format label
    label = f"#{rank}  {score:.0f}"
    
    # Make tinker model stand out
    if 'tinker' in name.lower() or 'qwen-235b-critic' in name.lower():
        ax.text(text_x, i, label, va='center', ha='left',
                fontweight='bold', color=text_color, fontsize=13,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=tinker_color, linewidth=2, alpha=0.9))
    else:
        ax.text(text_x, i, label, va='center', ha='left',
                fontweight='600', color=text_color, fontsize=12)

# Add legend with better styling
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=tinker_color, alpha=0.85, edgecolor='white', linewidth=2, 
          label='🎯 Tinker Model (Your Model)'),
    Patch(facecolor=top3_color, alpha=0.85, edgecolor='white', linewidth=2,
          label='Top 3 Frontier Models'),
    Patch(facecolor=other_color, alpha=0.85, edgecolor='white', linewidth=2,
          label='Other Models')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=12, 
          frameon=True, fancybox=True, shadow=True, framealpha=0.95)

# Set x-axis limits for better spacing
max_val = max([max(s, c) for s, c in zip(elo_scores, ci_highs)])
ax.set_xlim([0, max_val + 150])

plt.tight_layout()
plt.savefig('elo_rankings.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Graph saved as 'elo_rankings.png'")
plt.show()
