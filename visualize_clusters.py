import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.express as px

# ---------------- Load cluster summary ----------------
with open("clusters.pkl", "rb") as f:
    cluster_summary = pickle.load(f)

df_plot = pd.DataFrame(cluster_summary)

# ---------------- Scale positions to [0,1] for visualization ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
df_plot[["x", "y"]] = scaler.fit_transform(df_plot[["x", "y"]])

# ---------------- Scale bubble size for visibility ----------------
df_plot["size_scaled"] = np.sqrt(df_plot["size"])

# ---------------- Assign random color per cluster ----------------
unique_clusters = df_plot["cluster"].unique()
colors = px.colors.sample_colorscale(
    "Rainbow", [i / (len(unique_clusters) - 1) for i in range(len(unique_clusters))]
)
color_map = dict(zip(unique_clusters, colors))
df_plot["color"] = df_plot["cluster"].map(color_map)

# ---------------- Add readable label column ----------------
df_plot["label"] = df_plot["cluster"].apply(lambda c: f"Cluster {c}")

# ---------------- Plot with summaries overlayed ----------------
fig = px.scatter(
    df_plot,
    x="x",
    y="y",
    size="size_scaled",
    color="color",
    text="summary",  # overlay AI-generated summaries
    hover_data={"label": True, "size": True, "cluster": True},
    title="Clusters of Posts",
    width=1000,
    height=700,
)

# ---------------- Improve visualization ----------------
fig.update_traces(textposition="middle center", textfont_size=10)
fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    showlegend=False  # hide legend for clarity
)

fig.show()


