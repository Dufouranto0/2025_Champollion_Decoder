# evaluation/compare_test_losses.py

import json
import matplotlib.pyplot as plt

with open("all_results.json") as f:
    results = json.load(f)

# Extract (region, test_loss) pairs
region_losses = [(r, results[r]["test_loss"]) for r in results]

# Sort by test_loss (ascending)
region_losses.sort(key=lambda x: x[1])

# Unpack sorted values
regions, test_losses = zip(*region_losses)

# Plot
plt.figure(figsize=(12, 8))
bars = plt.bar(regions, test_losses)
plt.xticks(rotation=90)
plt.ylabel("Test BCE Loss (after 10 epochs)")
plt.title("Decoder Test Loss per Region (sorted), for models from Champollion_V1_after_ablation_latent_256")

# Extend y-axis limit by 10% so labels fit
plt.ylim(0, max(test_losses) * 1.15)

# Add numerical labels above each bar
for bar, loss in zip(bars, test_losses):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height()+0.002,
        f"{loss:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90
    )
plt.tight_layout()
plt.savefig("test_loss_comparison.png")
