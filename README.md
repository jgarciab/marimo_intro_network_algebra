# marimo_intro_network_algebra

A standalone [marimo](https://marimo.io) app that builds intuition for
one idea: **matrix multiplication on an adjacency matrix is counting
paths** — and path-counting powers half of network science.

Built as a companion to the *Network Representation, Algebra, and
Centrality* lecture of the Network Science Summer School (Utrecht
University, 2026 edition).

**Live (WASM build):**
<https://javier.science/marimo_intro_network_algebra/>

Every push to `main` rebuilds the WASM bundle via GitHub Actions
(`.github/workflows/deploy.yml`) and republishes it to GitHub Pages.

## What's inside

Seven short sections, all driven by the same active network. The default
is a five-person toy network, small enough that every number in every
multiplication is visible and checkable by hand:

1. **The network and its matrix** — pick a node, see its row highlighted
   in the numeric matrix and its edges highlighted in the drawing.
2. **`A @ x` — ask every node about its neighbours** — the two matrices
   drawn side by side, with the selected row coloured and the products
   written under the columns they come from.
3. **Average of friends, and the friendship paradox** — the network plus
   a table you can check by hand; histograms on bigger networks.
4. **`A @ A` — a row times a column** — both adjacency matrices side by
   side, row of the first and column of the second coloured, the result
   cell boxed: common neighbours, computed in front of you.
5. **Keep multiplying** — walks of length k drawn on the network and
   colour-matched to the terms that count them; reachability rings in
   sequential colours; triangles on the diagonal of `A @ A @ A`.
6. **Multiply again and again** — power iteration settles on a ranking
   (a teaser for eigenvector centrality and PageRank).
7. **Where you will meet this again** — one line per day of the summer
   school: graph models, community detection, link prediction,
   embeddings, graphical models, social contagion.

Default network: "Five friends" (5 nodes). Also bundled: a small
directed toy, the Krackhardt Kite, Florentine families (Padgett 1994),
and Zachary's karate club. You can also upload your own — CSV edge list
with `source`/`target` columns, or a GraphML/GML file (cap: 60 nodes).
Full numeric matrices and spelled-out sums appear for networks up to
16 nodes; bigger networks fall back to heatmaps.

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
