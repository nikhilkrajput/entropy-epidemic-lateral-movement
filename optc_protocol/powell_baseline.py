"""
Powell-style graph-role baseline (after Powell 2020, "Detecting malicious
logins as graph anomalies"), adapted from per-user login graphs to the
network-wide host graph of each time unit, because the OpTC flows carry no
user and the comparison must be the same on both datasets.

For each unit u (a day on LANL, an hour on OpTC) the directed host graph
G_u has an edge s -> d for every (source, destination) pair with traffic.
Every vertex gets six features: log1p of out-degree, in-degree, out-volume
and in-volume, the local clustering coefficient of the undirected graph,
and PageRank (times the vertex count, so that it does not shrink with the
graph).  Features are min-max scaled on the history window, a rank-R NMF is
fitted on the vertex-feature rows of the previous W units, and the score of
a source vertex in unit u is the Euclidean reconstruction error of its
feature row (NNLS projection on the fitted roles).  Causal: unit u is
scored with roles fitted on units u-W .. u-1 only; the first unit has no
score (NaN).

Usage: python powell_baseline.py edges.csv.gz unit_col out.csv.gz [--window W] [--rank R]
  edges: columns s, <unit_col>, d, count
"""
import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

ap = argparse.ArgumentParser()
ap.add_argument("edges")
ap.add_argument("unit")
ap.add_argument("out")
ap.add_argument("--window", type=int, default=7)
ap.add_argument("--rank", type=int, default=5)
args = ap.parse_args()

e = pd.read_csv(args.edges) if args.edges.endswith((".csv", ".csv.gz")) else pd.read_parquet(args.edges)
units = np.sort(e[args.unit].unique())
feats = {}
t0 = time.time()
for u in units:
    x = e[e[args.unit] == u]
    g = nx.DiGraph()
    g.add_weighted_edges_from(zip(x.s, x.d, x["count"]))
    outd = x.groupby("s")["d"].nunique()
    ind = x.groupby("d")["s"].nunique()
    outv = x.groupby("s")["count"].sum()
    inv = x.groupby("d")["count"].sum()
    nodes = pd.Index(sorted(g.nodes()))
    cl = nx.clustering(g.to_undirected())
    pr = nx.pagerank(g, weight="weight")
    f = pd.DataFrame({"outd": np.log1p(outd.reindex(nodes).fillna(0)), "ind": np.log1p(ind.reindex(nodes).fillna(0)),
                      "outv": np.log1p(outv.reindex(nodes).fillna(0)), "inv": np.log1p(inv.reindex(nodes).fillna(0)),
                      "clust": pd.Series(cl).reindex(nodes).fillna(0),
                      "pr": pd.Series(pr).reindex(nodes).fillna(0) * len(nodes)}, index=nodes)
    feats[u] = f
    print(f"features unit {u}: {len(nodes)} vertices, {g.number_of_edges()} edges, {time.time() - t0:.0f}s", flush=True)

rows = []
for i, u in enumerate(units):
    src = e.loc[e[args.unit] == u, "s"].unique()
    if i == 0:
        rows.append(pd.DataFrame({"s": src, args.unit: u, "powell": np.nan}))
        continue
    hist = pd.concat([feats[v] for v in units[max(0, i - args.window):i]])
    lo, hi = hist.min(), hist.max()
    span = (hi - lo).replace(0, 1.0)
    H = ((hist - lo) / span).clip(lower=0).to_numpy()
    nmf = NMF(n_components=min(args.rank, H.shape[1]), init="nndsvda", max_iter=500, random_state=0).fit(H)
    cur = ((feats[u].loc[src] - lo) / span).clip(lower=0).to_numpy()
    W = nmf.transform(cur)
    err = np.linalg.norm(cur - W @ nmf.components_, axis=1)
    rows.append(pd.DataFrame({"s": src, args.unit: u, "powell": err}))
    print(f"scored unit {u}", flush=True)
pd.concat(rows).to_csv(args.out, index=False)
