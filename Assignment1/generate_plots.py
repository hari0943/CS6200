#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('simulation_results.csv')

data.set_index('Experiments', inplace=True)

metrics = [
    ('Serial Time (s)', 'Average Serial Time (s)'),
    ('Serial Pi Estimate', 'Average Serial Pi Estimate'),
    ('Serial Pi Estimation Error (%)', 'Average Serial Pi Estimation Error (%)'),
    ('Parallel Time (s)', 'Average Parallel Time (s)'),
    ('Parallel Pi Estimate', 'Average Parallel Pi Estimate'),
    ('Parallel Pi Estimation Error (%)', 'Average Parallel Pi Estimation Error (%)'),
    ('Average Speedup', 'Average Speedup'),
]

for metric, title in metrics:
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[metric], marker='o', linestyle='-', color='b')
    plt.xlabel('Experiments')
    plt.ylabel(title)
    plt.grid()
    plt.xticks(data.index, rotation=90)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

print("Plots have been generated and saved as PNG files.")
