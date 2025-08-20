# evaluation/compare_test_losses.py

import json, matplotlib.pyplot as plt

with open("all_results.json") as f:
    results = json.load(f)

regions = list(results.keys())
test_losses = [results[r]["test_loss"] for r in regions]

plt.figure(figsize=(12,6))
plt.bar(regions, test_losses)
plt.xticks(rotation=90)
plt.ylabel("Test BCE Loss (after 10 epochs)")
plt.tight_layout()
plt.savefig("test_loss_comparison.png")
