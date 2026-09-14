import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Provide the data
markdown_table = """
| Layer                         |   Greedy Final Rank |   Retained SV Weight |
|-------------------------------|---------------------|----------------------|
| Block 0.Sa.Q Projection       |                  84 |               0.2347 |
| Block 0.Sa.K Projection       |                 134 |               0.3486 |
| Block 0.Sa.V Projection       |                 134 |               0.3469 |
| Block 0.Sa.Output Projection  |                  84 |               0.2316 |
| Block 0.Ffwd.Net.0            |                 384 |               0.6108 |
| Block 0.Ffwd.Net.2            |                 334 |               0.5449 |
| Block 1.Sa.Q Projection       |                  84 |               0.2336 |
| Block 1.Sa.K Projection       |                 134 |               0.348  |
| Block 1.Sa.V Projection       |                 134 |               0.3452 |
| Block 1.Sa.Output Projection  |                  84 |               0.2306 |
| Block 1.Ffwd.Net.0            |                 384 |               0.611  |
| Block 1.Ffwd.Net.2            |                 384 |               0.6112 |
| Block 2.Sa.Q Projection       |                  84 |               0.2345 |
| Block 2.Sa.K Projection       |                  84 |               0.2336 |
| Block 2.Sa.V Projection       |                 134 |               0.344  |
| Block 2.Sa.Output Projection  |                  34 |               0.1011 |
| Block 2.Ffwd.Net.0            |                 334 |               0.5451 |
| Block 2.Ffwd.Net.2            |                 284 |               0.475  |
| Block 3.Sa.Q Projection       |                  84 |               0.2345 |
| Block 3.Sa.K Projection       |                  84 |               0.2337 |
| Block 3.Sa.V Projection       |                 184 |               0.4477 |
| Block 3.Sa.Output Projection  |                 134 |               0.344  |
| Block 3.Ffwd.Net.0            |                 334 |               0.545  |
| Block 3.Ffwd.Net.2            |                 334 |               0.545  |
| Block 4.Sa.Q Projection       |                  34 |               0.1067 |
| Block 4.Sa.K Projection       |                  84 |               0.2351 |
| Block 4.Sa.V Projection       |                 184 |               0.4466 |
| Block 4.Sa.Output Projection  |                 184 |               0.4484 |
| Block 4.Ffwd.Net.0            |                 284 |               0.4751 |
| Block 4.Ffwd.Net.2            |                 284 |               0.4753 |
| Block 5.Sa.Q Projection       |                  34 |               0.1075 |
| Block 5.Sa.K Projection       |                  34 |               0.1063 |
| Block 5.Sa.V Projection       |                 184 |               0.4474 |
| Block 5.Sa.Output Projection  |                 134 |               0.3441 |
| Block 5.Ffwd.Net.0            |                 134 |               0.2429 |
| Block 5.Ffwd.Net.2            |                 184 |               0.3248 |
| Block 6.Sa.Q Projection       |                  34 |               0.1074 |
| Block 6.Sa.K Projection       |                  34 |               0.1071 |
| Block 6.Sa.V Projection       |                 134 |               0.3439 |
| Block 6.Sa.Output Projection  |                 134 |               0.3445 |
| Block 6.Ffwd.Net.0            |                 234 |               0.4016 |
| Block 6.Ffwd.Net.2            |                 234 |               0.402  |
| Block 7.Sa.Q Projection       |                  34 |               0.1087 |
| Block 7.Sa.K Projection       |                  34 |               0.109  |
| Block 7.Sa.V Projection       |                  84 |               0.228  |
| Block 7.Sa.Output Projection  |                  84 |               0.2286 |
| Block 7.Ffwd.Net.0            |                  84 |               0.1573 |
| Block 7.Ffwd.Net.2            |                  34 |               0.0664 |
| Block 8.Sa.Q Projection       |                  34 |               0.1079 |
| Block 8.Sa.K Projection       |                  34 |               0.1078 |
| Block 8.Sa.V Projection       |                 134 |               0.3433 |
| Block 8.Sa.Output Projection  |                 134 |               0.3434 |
| Block 8.Ffwd.Net.0            |                 134 |               0.2436 |
| Block 8.Ffwd.Net.2            |                 134 |               0.2443 |
| Block 9.Sa.Q Projection       |                  34 |               0.1031 |
| Block 9.Sa.K Projection       |                  34 |               0.104  |
| Block 9.Sa.V Projection       |                  34 |               0.0981 |
| Block 9.Sa.Output Projection  |                  34 |               0.0978 |
| Block 9.Ffwd.Net.0            |                 134 |               0.244  |
| Block 9.Ffwd.Net.2            |                 134 |               0.2448 |
| Block 10.Sa.Q Projection      |                  34 |               0.1005 |
| Block 10.Sa.K Projection      |                  34 |               0.1021 |
| Block 10.Sa.V Projection      |                  34 |               0.0975 |
| Block 10.Sa.Output Projection |                  34 |               0.0973 |
| Block 10.Ffwd.Net.0           |                  84 |               0.1587 |
| Block 10.Ffwd.Net.2           |                  84 |               0.16   |
| Block 11.Sa.Q Projection      |                  34 |               0.1    |
| Block 11.Sa.K Projection      |                  34 |               0.1008 |
| Block 11.Sa.V Projection      |                  34 |               0.0977 |
| Block 11.Sa.Output Projection |                  34 |               0.0975 |
| Block 11.Ffwd.Net.0           |                  34 |               0.0683 |
| Block 11.Ffwd.Net.2           |                  34 |               0.07   |
"""

# 2. Parse the markdown table into a list of rows
lines = markdown_table.strip().split('\n')
data = []
for line in lines[2:]:
    parts = [p.strip() for p in line.split('|') if p.strip()]
    if len(parts) == 3:
        data.append([parts[0], int(parts[1]), float(parts[2])])

df = pd.DataFrame(data, columns=['Layer', 'Rank', 'Weight'])

# 3. Map the extracted component names to match your HTML column headers
comp_map = {
    'Sa.Q Projection': 'Q projection',
    'Sa.K Projection': 'K projection',
    'Sa.V Projection': 'V projection',
    'Sa.Output Projection': 'self attention output',
    'Ffwd.Net.0': 'ffwd 0',
    'Ffwd.Net.2': 'ffwd 2'
}

df['Block'] = df['Layer'].apply(lambda x: x.split('.')[0])
df['ComponentRaw'] = df['Layer'].apply(lambda x: '.'.join(x.split('.')[1:]))
df['Component'] = df['ComponentRaw'].map(comp_map)

cols_order = ['Q projection', 'K projection', 'V projection', 'self attention output', 'ffwd 0', 'ffwd 2']
blocks_order = [f"Block {i}" for i in range(12)]

# 4. Create pivot tables for Annotations (Rank) and Values (Weight)
rank_pivot = df.pivot(index='Block', columns='Component', values='Rank').reindex(index=blocks_order, columns=cols_order)
weight_pivot = df.pivot(index='Block', columns='Component', values='Weight').reindex(index=blocks_order, columns=cols_order)

# 5. Plot the heatmap using matplotlib's imshow
fig, ax = plt.subplots(figsize=(10, 8))

# aspect="auto" lets the cells stretch rather than forcing square proportions
cax = ax.imshow(weight_pivot.values, cmap="viridis", aspect="auto")

# Set the ticks and labels based on dataframe index and columns
ax.set_xticks(np.arange(len(cols_order)))
ax.set_yticks(np.arange(len(blocks_order)))
ax.set_xticklabels(cols_order, rotation=45, ha="right")
ax.set_yticklabels(blocks_order)

# 6. Loop over data dimensions and create text annotations for the Rank
for i in range(len(blocks_order)):
    for j in range(len(cols_order)):
        # Calculate dynamic text color based on background weight to maintain readability
        text_color = "white" if weight_pivot.values[i, j] < 0.5 else "black"
        ax.text(j, i, int(rank_pivot.values[i, j]),
                ha="center", va="center", color=text_color)

# 7. Add colorbar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Retained SV Weight')

ax.set_title("Matrix Rank and Retained Singular Values Weight for Greedy98_50_15", pad=20, fontsize=14)

# Remove the default framing around the plot to match traditional heatmaps (optional)
ax.spines[:].set_visible(False)

fig.tight_layout()
plt.savefig("img/heatmap.png", dpi=600, bbox_inches='tight')
plt.show()