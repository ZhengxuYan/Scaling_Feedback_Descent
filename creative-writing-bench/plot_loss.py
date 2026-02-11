
import json
import matplotlib.pyplot as plt
import os

metrics_file = "/Users/jasonyan/Desktop/Scaling_Feedback_Descent/logs/sft_creative_writing_critic_freeform_Qwen3-4B-Instruct-2507/metrics.jsonl"
output_plot = "loss_plot.png"

steps = []
losses = []

with open(metrics_file, 'r') as f:
    for line in f:
        if line.strip():
            try:
                data = json.loads(line)
                steps.append(data['step'])
                losses.append(data['train_mean_nll'])
            except:
                pass

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, alpha=0.6, label='Train Mean NLL')
plt.xlabel('Step')
plt.ylabel('Negative Log Likelihood')
plt.title('Training Loss Convergence')
plt.legend()
plt.grid(True)
plt.savefig(output_plot)
print(f"Plot saved to {os.path.abspath(output_plot)}")
