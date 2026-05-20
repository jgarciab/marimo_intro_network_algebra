# marimo_intro_network_algebra

A standalone [marimo](https://marimo.io) app that gives visual intuition
for how matrix multiplication on an adjacency matrix encodes paths,
triangles, and the major centrality measures.

Built as a companion to the *Network Representation, Algebra, and
Centrality* lecture of the Network Science Summer School (Utrecht
University, 2026 edition).

## What's inside

Eight short sections, all driven by the same active network:

1. **The adjacency matrix as a picture** — node-link view alongside a
   heatmap of $A$, with the transpose toggle and the trace.
2. **$Ax$ — ask each node a question about its neighbours** — pick the
   vector $x$ (ones, indicator, Gaussian) and watch $A x$ light up.
3. **The friendship paradox** — degree vs. average-neighbour-degree.
4. **$A^k$ counts length-$k$ walks** — slider over $k$, click a cell to
   walk through the dot product.
5. **Triangles live on the diagonal of $A^3$** — bar chart of triangles
   per node and the same on the graph.
6. **Power iteration → eigenvector centrality** — repeat $x \leftarrow A x$
   and watch the bars converge.
7. **Random walks → PageRank** — drop a unit mass, iterate
   $x \leftarrow P^\top x$, and compare with `g.pagerank()`.
8. **Same network, different questions** — five centralities side by
   side on the same graph.

Default network: Florentine families (Padgett 1994, 16 nodes). Also
bundled: Zachary's karate club, the Krackhardt Kite, and a small
directed toy. You can also upload your own — CSV edge list with
`source`/`target` columns, or a GraphML/GML file (cap: 60 nodes).

## Running locally

```bash
./run.sh --setup   # first time only
./run.sh           # launches marimo in run mode
```

The setup step creates a virtualenv at `~/.uv_envs/day1b_matrix_paths`
because pCloud Drive (where this is developed) doesn't support symlinks
and uv's default `.venv` layout breaks on it.

## Building a static WASM bundle

```bash
./export_wasm.sh   # writes ./build/index.html plus assets
cd build && python -m http.server 8000
```

Everything in `app.py` is Pyodide-friendly — `numpy`, `pandas`,
`matplotlib`, and `python-igraph`. No NetworkX, no SciPy, no torch.

## License

MIT.
