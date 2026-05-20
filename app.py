import marimo

__generated_with = "0.23.6"
app = marimo.App(
    width="medium",
    app_title="Matrix multiplication, paths, triangles & centrality",
)


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import igraph as ig
    import io
    import random as rnd_mod

    rnd_mod.seed(1)
    np.random.seed(1)

    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#cccccc",
    })

    PALETTE = (
        "#0b789d", "#e07b00", "#7a9e3b", "#b04a6f", "#5e548e", "#c9a227",
        "#3aa6a0", "#9a4f86", "#d1495b", "#3a7d44", "#2b6cb0", "#8c6b3f",
    )
    ACCENT = "#0b789d"
    NEUTRAL_NODE = "#0b789d"
    EDGE_COLOR = (0.85, 0.85, 0.85, 0.35)

    # Tighter cap than the day1 intuition app — matshow on A^k doesn't
    # scale much past ~60 nodes before the heatmap becomes unreadable.
    MAX_NODES = 60
    return (
        ACCENT, EDGE_COLOR, MAX_NODES, NEUTRAL_NODE, PALETTE,
        ig, io, mo, np, pd, plt, rnd_mod,
    )


# -----------------------------------------------------------------------------
# Inlined network catalogue
# -----------------------------------------------------------------------------


@app.cell
def network_catalogue(ig):
    # Florentine families (Padgett 1994). The default — 16 nodes fits any
    # matrix display, and Medici-vs-Strozzi is a story worth telling.
    _flor_names = (
        "ACCIAIUOL", "ALBIZZI", "BARBADORI", "BISCHERI", "CASTELLAN",
        "GINORI", "GUADAGNI", "LAMBERTES", "MEDICI", "PAZZI", "PERUZZI",
        "PUCCI", "RIDOLFI", "SALVIATI", "STROZZI", "TORNABUON",
    )
    _flor_edges = (
        (0, 8), (1, 5), (1, 6), (1, 8), (2, 4), (2, 5), (2, 8), (2, 10),
        (3, 6), (3, 7), (3, 10), (3, 14), (4, 7), (4, 10), (4, 14),
        (5, 8), (6, 7), (6, 15), (7, 10), (8, 9), (8, 12), (8, 13),
        (8, 15), (9, 13), (10, 14), (12, 14), (12, 15),
    )

    # Zachary karate factions (Zachary 1977).
    _zachary_factions = (
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    )

    # A small directed toy so that the transpose / asymmetry view in
    # Section 1 actually shows something. Six nodes, citation-like.
    _toy_names = ("A", "B", "C", "D", "E", "F")
    _toy_edges = (
        (0, 1), (0, 2), (1, 3), (2, 3), (3, 4),
        (4, 1), (4, 5), (5, 0),
    )

    def _build(names, edges, attr=None, attr_name=None,
               value_names=None, directed=False):
        g_ = ig.Graph(n=len(names), edges=list(edges), directed=directed)
        g_.vs["name"] = list(names)
        if attr is not None:
            g_.vs[attr_name] = list(attr)
            g_["_attr_name"] = attr_name
            g_["_attr_value_names"] = (
                list(value_names) if value_names else None
            )
        return g_

    def _florentine():
        return _build(_flor_names, _flor_edges)

    def _karate():
        g_ = ig.Graph.Famous("Zachary")
        g_.vs["name"] = [str(i) for i in range(g_.vcount())]
        g_.vs["Faction"] = list(_zachary_factions)
        g_["_attr_name"] = "Faction"
        g_["_attr_value_names"] = ["Mr Hi", "Officer"]
        return g_

    def _kite():
        g_ = ig.Graph.Famous("Krackhardt_Kite")
        g_.vs["name"] = [str(i) for i in range(g_.vcount())]
        return g_

    def _toy_directed():
        return _build(_toy_names, _toy_edges, directed=True)

    BUNDLED = {
        "Florentine families (16 nodes)": _florentine,
        "Karate club (34 nodes, 2 factions)": _karate,
        "Krackhardt Kite (10 nodes)": _kite,
        "Directed toy (6 nodes)": _toy_directed,
    }
    return (BUNDLED,)


# -----------------------------------------------------------------------------
# Sidebar: network chooser (visible from every section)
# -----------------------------------------------------------------------------


@app.cell
def chooser_widgets(BUNDLED, mo):
    bundled_choice = mo.ui.dropdown(
        options=list(BUNDLED.keys()),
        value="Florentine families (16 nodes)",
        label="Bundled network",
    )
    file_upload = mo.ui.file(
        kind="button",
        filetypes=[".csv", ".tsv", ".txt", ".graphml", ".gml", ".xml"],
        label="Upload edge list or GraphML/GML",
        multiple=False,
    )
    return bundled_choice, file_upload


@app.cell
def build_active_graph(BUNDLED, MAX_NODES, bundled_choice, file_upload, ig, io, pd):
    import os
    import tempfile

    def _read_uploaded(file_obj):
        name = file_obj.name
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        raw = file_obj.contents
        if isinstance(raw, str):
            raw_bytes = raw.encode("utf-8")
        else:
            raw_bytes = raw

        if ext in ("graphml", "xml"):
            with tempfile.NamedTemporaryFile(
                suffix=".graphml", delete=False
            ) as tf:
                tf.write(raw_bytes)
                tf_path = tf.name
            try:
                return ig.Graph.Read_GraphML(tf_path)
            finally:
                try:
                    os.unlink(tf_path)
                except OSError:
                    pass

        if ext == "gml":
            with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tf:
                tf.write(raw_bytes)
                tf_path = tf.name
            try:
                return ig.Graph.Read_GML(tf_path)
            finally:
                try:
                    os.unlink(tf_path)
                except OSError:
                    pass

        text = raw_bytes.decode("utf-8", errors="replace")
        sep = "\t" if text.count("\t") > text.count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
        df.columns = [c.lower().strip() for c in df.columns]
        if "source" not in df.columns or "target" not in df.columns:
            raise ValueError(
                "Edge list must have columns named 'source' and 'target'."
            )
        edges = list(zip(df["source"].astype(str), df["target"].astype(str)))
        return ig.Graph.TupleList(edges, directed=False)

    upload_warning = None
    source_label = ""
    g_active = None

    if file_upload.value and len(file_upload.value) > 0:
        _f = file_upload.value[0]
        try:
            _g = _read_uploaded(_f)
            _g.simplify(multiple=True, loops=True)
            if _g.vcount() > MAX_NODES:
                _g = _g.connected_components().giant()
                if _g.vcount() > MAX_NODES:
                    _keep = sorted(
                        range(_g.vcount()), key=lambda i: -_g.degree(i)
                    )[:MAX_NODES]
                    _g = _g.subgraph(_keep)
                upload_warning = (
                    f"Upload had more than {MAX_NODES} nodes; trimmed to "
                    f"the largest component ({_g.vcount()} nodes)."
                )
            g_active = _g
            source_label = f"upload: {_f.name}"
        except Exception as _e:
            upload_warning = f"Upload error: {_e}. Falling back to bundled."
            g_active = None

    if g_active is None:
        g_active = BUNDLED[bundled_choice.value]()
        source_label = bundled_choice.value

    g_active.simplify(multiple=True, loops=True)
    if "name" not in g_active.vs.attributes() or any(
        v["name"] is None for v in g_active.vs
    ):
        g_active.vs["name"] = [str(i) for i in range(g_active.vcount())]

    g = g_active
    is_directed = g.is_directed()
    return g, is_directed, source_label, upload_warning


@app.cell
def sidebar_cell(bundled_choice, file_upload, g, is_directed, mo, source_label, upload_warning):
    _items = [
        mo.md("### Active network"),
        bundled_choice,
        mo.md("_Or upload your own (CSV with `source`/`target`, or GraphML/GML):_"),
        file_upload,
        mo.md(
            f"**Currently:**  \n{source_label}  \n"
            f"_n = {g.vcount()}, m = {g.ecount()}, "
            f"{'directed' if is_directed else 'undirected'}_"
        ),
    ]
    if upload_warning is not None:
        _items.append(mo.md(f"> {upload_warning}"))
    _items.append(mo.md("---"))
    _items.append(mo.md(
        "_The active network drives every section below — pick once "
        "here, then scroll._"
    ))
    mo.sidebar(_items)
    return


# -----------------------------------------------------------------------------
# Shared computations: adjacency matrix and a fixed layout
# -----------------------------------------------------------------------------


@app.cell
def shared_A_and_layout(g, np):
    # Adjacency matrix as a dense float array. With MAX_NODES = 60 this
    # is at most 60*60 = 3600 floats — cheap to recompute every time the
    # network changes, no caching needed.
    A = np.array(g.get_adjacency().data, dtype=float)

    # One layout per active network. Kamada-Kawai works well for the
    # small bundled graphs; for anything bigger or non-symmetric we fall
    # back to Fruchterman-Reingold with a fixed seed so that node
    # positions are stable across sections.
    try:
        _layout = g.layout_kamada_kawai()
    except Exception:
        _layout = g.layout_fruchterman_reingold(niter=500, seed=1)
    layout_coords = [tuple(row) for row in _layout.coords]
    return A, layout_coords


# -----------------------------------------------------------------------------
# Title
# -----------------------------------------------------------------------------


@app.cell
def title(mo):
    mo.md(r"""
    # Matrix multiplication, paths, triangles & centrality

    Matrix multiplication on an adjacency matrix isn't an abstract
    algebraic trick — it's *the* way networks answer questions about
    themselves.

    - $A x$ asks every node a question about its neighbours.
    - $A^2$ counts length-2 paths, and the diagonal counts common
      neighbours.
    - $A^3$ has triangles on its diagonal.
    - Iterating $x \leftarrow A x$ converges to eigenvector centrality.
    - Iterating $x \leftarrow P^\top x$ on the row-normalised $P$
      converges to PageRank.

    **One matrix, many questions.** Pick a network in the sidebar (the
    default is the Florentine families, 16 nodes) and scroll through.
    """)
    return


# -----------------------------------------------------------------------------
# Section 1 — The adjacency matrix as a picture
# -----------------------------------------------------------------------------


@app.cell
def section1_header(mo):
    mo.md(r"""
    ---
    ## 1. The adjacency matrix as a picture

    The same network, two pictures: a node-link drawing on the left, a
    heatmap of the adjacency matrix $A$ on the right.

    - **A blue cell** at row $i$, column $j$ means there is an edge from
      $i$ to $j$.
    - **The diagonal** (top-left to bottom-right) is zero for a simple
      graph — no self-loops. The *trace* of $A$ (the sum of the
      diagonal) is therefore zero too.
    - **Symmetry across the diagonal** means undirected. If you flip
      the matrix across the diagonal ($A^\top$, the *transpose*) and
      get back the same picture, the graph is undirected.
    """)
    return


@app.cell
def s1_controls(is_directed, mo):
    show_transpose = mo.ui.checkbox(
        value=False,
        label="Show the transpose $A^\\top$ instead",
    )
    _note = mo.md(
        "_The active graph is directed — toggle the transpose to see the "
        "asymmetry: $A \\neq A^\\top$._"
        if is_directed else
        "_The active graph is undirected, so $A = A^\\top$ — toggling the "
        "transpose makes no visible difference._"
    )
    mo.vstack([show_transpose, _note], gap=0.4)
    return (show_transpose,)


@app.cell
def s1_plot(A, EDGE_COLOR, NEUTRAL_NODE, g, ig, layout_coords, mo, np, plt, show_transpose):
    _M = A.T if show_transpose.value else A
    _label = "A^T" if show_transpose.value else "A"
    _n = g.vcount()
    _names = g.vs["name"]

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5.0))

    # Left: node-link
    _axL = _axes[0]
    _axL.set_facecolor("white")
    _axL.grid(False)
    _axL.set_xticks([])
    _axL.set_yticks([])
    for _side in ("top", "right", "bottom", "left"):
        _axL.spines[_side].set_visible(False)
    ig.plot(
        g,
        target=_axL,
        layout=layout_coords,
        vertex_color=NEUTRAL_NODE,
        vertex_size=22 if _n <= 30 else 14,
        vertex_frame_width=0,
        vertex_label=_names if _n <= 30 else [""] * _n,
        vertex_label_size=9,
        vertex_label_color="#222222",
        edge_color=EDGE_COLOR,
        edge_width=1.0,
        edge_arrow_size=0.8 if g.is_directed() else 0.0,
    )
    _axL.set_title(f"Node-link view  ·  n = {_n}, m = {g.ecount()}")

    # Right: heatmap of A (or A^T)
    _axR = _axes[1]
    _axR.imshow(_M, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    # diagonal highlight
    _axR.plot(
        [-0.5, _n - 0.5], [-0.5, _n - 0.5],
        color="#c0223b", linestyle="--", linewidth=1.2, alpha=0.8,
    )
    if _n <= 30:
        _axR.set_xticks(range(_n))
        _axR.set_yticks(range(_n))
        _axR.set_xticklabels(_names, rotation=90, fontsize=8)
        _axR.set_yticklabels(_names, fontsize=8)
    else:
        _axR.set_xticks([])
        _axR.set_yticks([])
    _trace = float(np.trace(A))
    _axR.set_title(f"{_label}   ·   trace = {int(_trace)}")
    _axR.grid(False)
    plt.tight_layout()
    mo.vstack([_fig])
    return


# -----------------------------------------------------------------------------
# Section 2 — A @ x = ask each node a question about its neighbours
# -----------------------------------------------------------------------------


@app.cell
def section2_header(mo):
    mo.md(r"""
    ---
    ## 2. $Ax$ — ask each node a question about its neighbours

    Multiplying $A$ by a vector $x$ produces a new vector $y = Ax$ where

    $$
    y_i = \sum_{j} A_{ij}\, x_j = \sum_{j \in N(i)} x_j
    $$

    In plain words: **$y_i$ is the sum of $x$ over node $i$'s
    neighbours**. So choose $x$ and you choose the question:

    - $x = (1, 1, \dots, 1)$ &nbsp; → &nbsp; $y$ = each node's **degree**.
    - $x$ = an indicator for one node &nbsp; → &nbsp; $y$ marks that
      node's neighbours.
    - $x$ = a random Gaussian &nbsp; → &nbsp; $y$ is the average
      neighbour signal at each node (an unweighted *graph smoothing*).
    """)
    return


@app.cell
def s2_widgets(g, mo):
    x_choice = mo.ui.dropdown(
        options=[
            "All ones (gives degree)",
            "Indicator for one node",
            "Random Gaussian",
        ],
        value="All ones (gives degree)",
        label="Vector x",
    )
    seed_choice = mo.ui.slider(
        start=1, stop=20, step=1, value=1,
        label="Random seed (Gaussian only)",
        show_value=True,
    )
    node_choice = mo.ui.dropdown(
        options=list(g.vs["name"]),
        value=g.vs["name"][0],
        label="Indicator node",
    )
    mo.hstack([x_choice, node_choice, seed_choice], gap=1.0, widths="equal")
    return node_choice, seed_choice, x_choice


@app.cell
def s2_plot(A, ACCENT, g, mo, node_choice, np, plt, seed_choice, x_choice):
    _n = g.vcount()
    _names = g.vs["name"]

    if x_choice.value == "All ones (gives degree)":
        _x = np.ones(_n)
        _xlabel = "x = 1"
    elif x_choice.value == "Indicator for one node":
        _x = np.zeros(_n)
        _idx = _names.index(node_choice.value)
        _x[_idx] = 1.0
        _xlabel = f"x = e_{node_choice.value}"
    else:
        _rng = np.random.default_rng(int(seed_choice.value))
        _x = _rng.standard_normal(_n)
        _xlabel = f"x ~ N(0, 1), seed = {int(seed_choice.value)}"

    _y = A @ _x

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    _ax1, _ax2 = _axes
    _show_ticks = _n <= 30

    _ax1.bar(range(_n), _x, color="#888888", edgecolor="white")
    _ax1.axhline(0, color="#444444", linewidth=0.6)
    _ax1.set_title(f"Input vector  ·  {_xlabel}")
    if _show_ticks:
        _ax1.set_xticks(range(_n))
        _ax1.set_xticklabels(_names, rotation=90, fontsize=8)
    else:
        _ax1.set_xticks([])
    _ax1.set_xlabel("node")

    _ax2.bar(range(_n), _y, color=ACCENT, edgecolor="white")
    _ax2.axhline(0, color="#444444", linewidth=0.6)
    _ax2.set_title("Output  y = A x  (sum of neighbours' x)")
    if _show_ticks:
        _ax2.set_xticks(range(_n))
        _ax2.set_xticklabels(_names, rotation=90, fontsize=8)
    else:
        _ax2.set_xticks([])
    _ax2.set_xlabel("node")

    for _ax in _axes:
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if x_choice.value == "All ones (gives degree)":
        _note = (
            "The output is exactly the degree sequence. "
            "$A \\mathbf{1}$ = row-sums of $A$ = degree."
        )
    elif x_choice.value == "Indicator for one node":
        _nb = [_names[j] for j in range(_n) if A[_idx, j] > 0]
        _note = (
            f"Only **{node_choice.value}**'s neighbours light up in $y$: "
            f"{', '.join(_nb) if _nb else '(none — isolated node)'}."
        )
    else:
        _note = (
            "The output is a smoothed version of the input — each node's "
            "value is replaced by the sum of its neighbours' values. "
            "High-degree nodes amplify; isolated nodes go to zero."
        )

    mo.vstack([_fig, mo.md(_note)])
    return


# -----------------------------------------------------------------------------
# Section 3 — Average of neighbours and the friendship paradox
# -----------------------------------------------------------------------------


@app.cell
def section3_header(mo):
    mo.md(r"""
    ---
    ## 3. Average of neighbours, and the friendship paradox

    Divide $Ax$ by the row sums (degree) and you get the **average** of
    $x$ over each node's neighbours:

    $$
    \bar y_i = \frac{1}{k_i} \sum_{j \in N(i)} x_j.
    $$

    A famous special case: take $x$ = degree itself. Then $\bar y_i$ is
    the **average degree of node $i$'s neighbours**. Comparing the
    distributions of $k_i$ and $\bar y_i$ gives the friendship paradox:
    *on average, your friends have more friends than you do*. The shift
    is purely structural — it comes from the fact that high-degree
    nodes appear in many neighbour lists.
    """)
    return


@app.cell
def s3_plot(A, ACCENT, g, mo, np, plt):
    _deg = np.array(g.degree(), dtype=float)
    _safe_deg = np.where(_deg > 0, _deg, 1.0)
    _avg_nbr_deg = (A @ _deg) / _safe_deg
    _avg_nbr_deg[_deg == 0] = np.nan

    _valid = ~np.isnan(_avg_nbr_deg)
    _mean_self = float(_deg[_valid].mean()) if _valid.any() else 0.0
    _mean_nbr = float(np.nanmean(_avg_nbr_deg)) if _valid.any() else 0.0

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
    _ax1, _ax2 = _axes

    _kmax = int(max(_deg.max(), np.nanmax(_avg_nbr_deg))) if _valid.any() else int(_deg.max())
    _bins = np.arange(0, _kmax + 2) - 0.5

    _ax1.hist(_deg, bins=_bins, color="#888888", edgecolor="white")
    _ax1.axvline(_mean_self, color="#c0223b", linestyle="--", linewidth=1.2,
                 label=f"mean = {_mean_self:.2f}")
    _ax1.set_title("Your degree  k_i")
    _ax1.set_xlabel("k")
    _ax1.legend(frameon=False, fontsize=9)

    _ax2.hist(_avg_nbr_deg[_valid], bins=_bins, color=ACCENT, edgecolor="white")
    _ax2.axvline(_mean_nbr, color="#c0223b", linestyle="--", linewidth=1.2,
                 label=f"mean = {_mean_nbr:.2f}")
    _ax2.set_title("Your friends' average degree  (A k) / k")
    _ax2.set_xlabel("avg. neighbour k")
    _ax2.legend(frameon=False, fontsize=9)

    for _ax in _axes:
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    plt.tight_layout()

    _diff = _mean_nbr - _mean_self
    _verdict = (
        "friends have a higher mean degree" if _diff > 1e-9
        else "friends have a lower mean degree (unusual!)"
        if _diff < -1e-9
        else "the two means are essentially equal (degree-regular graph)"
    )
    _note = mo.md(
        f"On the active network: mean degree = **{_mean_self:.2f}**, mean "
        f"of average-neighbour-degree = **{_mean_nbr:.2f}** — i.e. "
        f"{_verdict}. The right-hand histogram is the same data reweighted "
        "by who-appears-in-whose-friend-list."
    )
    mo.vstack([_fig, _note])
    return


# -----------------------------------------------------------------------------
# Section 4 — A^k counts length-k paths
# -----------------------------------------------------------------------------


@app.cell
def section4_header(mo):
    mo.md(r"""
    ---
    ## 4. $A^k$ counts length-$k$ paths

    The $(i, j)$ entry of $A^2$ counts **the number of length-2 walks
    from $i$ to $j$** — equivalently, **the number of common
    neighbours** of $i$ and $j$:

    $$
    (A^2)_{ij} = \sum_{k} A_{ik}\, A_{kj} = |N(i) \cap N(j)|.
    $$

    More generally, $(A^k)_{ij}$ counts length-$k$ walks from $i$ to
    $j$. Pick a power and watch the heatmap fill in — entries that were
    zero in $A$ become positive in $A^2$ wherever a length-2 walk
    exists. Pick a specific $(i, j)$ cell to walk through the dot
    product: which intermediate nodes contribute?
    """)
    return


@app.cell
def s4_widgets(g, mo):
    power_k = mo.ui.slider(
        start=1, stop=4, step=1, value=2,
        label="Power k", show_value=True, full_width=True,
    )
    _names = list(g.vs["name"])
    cell_i = mo.ui.dropdown(
        options=_names, value=_names[0], label="Row i",
    )
    cell_j = mo.ui.dropdown(
        options=_names, value=_names[min(1, len(_names) - 1)], label="Column j",
    )
    mo.hstack([power_k, cell_i, cell_j], gap=1.0, widths="equal")
    return cell_i, cell_j, power_k


@app.cell
def s4_compute(A, np, power_k):
    _k = int(power_k.value)
    _Ak = np.eye(A.shape[0])
    for _ in range(_k):
        _Ak = _Ak @ A
    Ak = _Ak
    return (Ak,)


@app.cell
def s4_plot(A, Ak, cell_i, cell_j, g, mo, np, plt, power_k):
    _n = g.vcount()
    _names = list(g.vs["name"])
    _k = int(power_k.value)

    # Cap the displayed matrix to the top-30 by degree if larger.
    if _n > 30:
        _idx = sorted(range(_n), key=lambda i: -g.degree(i))[:30]
        _idx = sorted(_idx)
        _M_disp = Ak[np.ix_(_idx, _idx)]
        _disp_names = [_names[i] for i in _idx]
        _i_in_disp = (
            _disp_names.index(cell_i.value)
            if cell_i.value in _disp_names else None
        )
        _j_in_disp = (
            _disp_names.index(cell_j.value)
            if cell_j.value in _disp_names else None
        )
        _trim_note = (
            " · displaying top-30 nodes by degree"
        )
    else:
        _M_disp = Ak
        _disp_names = _names
        _i_in_disp = _disp_names.index(cell_i.value)
        _j_in_disp = _disp_names.index(cell_j.value)
        _trim_note = ""

    _i_full = _names.index(cell_i.value)
    _j_full = _names.index(cell_j.value)

    _fig, _ax = plt.subplots(figsize=(7.5, 6.5))
    _vmax = max(1.0, float(_M_disp.max()))
    _im = _ax.imshow(_M_disp, cmap="magma", vmin=0, vmax=_vmax, aspect="equal")
    if len(_disp_names) <= 30:
        _ax.set_xticks(range(len(_disp_names)))
        _ax.set_yticks(range(len(_disp_names)))
        _ax.set_xticklabels(_disp_names, rotation=90, fontsize=8)
        _ax.set_yticklabels(_disp_names, fontsize=8)
    if _i_in_disp is not None and _j_in_disp is not None:
        _ax.scatter(
            [_j_in_disp], [_i_in_disp], s=180,
            facecolors="none", edgecolors="#1ed8a3", linewidths=2.5,
        )
    _ax.set_title(f"A^{_k}   ·   max entry = {int(Ak.max())}{_trim_note}")
    _ax.grid(False)
    _fig.colorbar(_im, ax=_ax, fraction=0.045)
    plt.tight_layout()

    # Walk through the dot product for the highlighted (i, j).
    _val = float(Ak[_i_full, _j_full])
    if _k == 1:
        _explain = mo.md(
            f"**Cell ($i$={cell_i.value}, $j$={cell_j.value}) of $A^1$** "
            f"= **{int(_val)}** — i.e. is there a direct edge $i \\to j$?"
        )
    elif _k == 2:
        # Contributions: nodes m with A[i, m] * A[m, j] > 0
        _contrib = []
        for _m in range(_n):
            _c = A[_i_full, _m] * A[_m, _j_full]
            if _c > 0:
                _contrib.append(_names[_m])
        _explain = mo.md(
            f"**Cell ($i$={cell_i.value}, $j$={cell_j.value}) of $A^2$** "
            f"= **{int(_val)}**.\n\n"
            "The dot product $(A^2)_{ij} = \\sum_m A_{im} A_{mj}$ counts "
            "the common neighbours of $i$ and $j$. On this network "
            "those are: "
            f"{', '.join('`' + c + '`' for c in _contrib) if _contrib else 'none'}."
        )
    else:
        _explain = mo.md(
            f"**Cell ($i$={cell_i.value}, $j$={cell_j.value}) of $A^{_k}$** "
            f"= **{int(_val)}**. Each unit counts one length-{_k} walk "
            f"from $i$ to $j$ (walks may revisit nodes)."
        )

    # Diagonal note
    if _k == 2:
        _diag_note = mo.md(
            "_Note the diagonal of $A^2$ — entry $(i, i)$ is the number "
            "of length-2 walks from $i$ back to $i$, which on an "
            "undirected graph equals **degree($i$)**._"
        )
    elif _k == 3:
        _diag_note = mo.md(
            "_The diagonal of $A^3$ counts closed length-3 walks "
            "through each node — i.e. twice the number of triangles "
            "containing that node. We use this in Section 5._"
        )
    else:
        _diag_note = mo.md("")

    mo.vstack([_fig, _explain, _diag_note])
    return


# -----------------------------------------------------------------------------
# Section 5 — Triangles live on the diagonal of A^3
# -----------------------------------------------------------------------------


@app.cell
def section5_header(mo):
    mo.md(r"""
    ---
    ## 5. Triangles live on the diagonal of $A^3$

    A length-3 walk that returns to its starting node $i$ visits two
    other nodes and closes a triangle. On an undirected simple graph,
    each triangle through $i$ is counted twice in $(A^3)_{ii}$ (once in
    each direction), so

    $$
    \text{triangles through } i \;=\; \tfrac{1}{2}\,(A^3)_{ii},
    \qquad
    \text{total triangles} \;=\; \tfrac{1}{6}\,\text{tr}(A^3).
    $$

    The bar chart shows triangles-through-node on the left; the
    node-link plot on the right scales each node by that count. On the
    Florentine network watch where the Medici sit.
    """)
    return


@app.cell
def s5_plot(A, ACCENT, EDGE_COLOR, NEUTRAL_NODE, g, ig, is_directed, layout_coords, mo, np, plt):
    _n = g.vcount()
    _names = g.vs["name"]
    _A3 = A @ A @ A
    _diag = np.diag(_A3)
    if is_directed:
        # The factor-of-2 only applies undirected; for directed graphs
        # the diagonal of A^3 already counts directed 3-cycles once.
        _tri_node = _diag.astype(float)
        _total_tri = float(np.trace(_A3) / 3.0)
        _formula_note = (
            "(directed graph: triangles per node = $(A^3)_{ii}$, "
            "total = $\\text{tr}(A^3)/3$)"
        )
    else:
        _tri_node = _diag / 2.0
        _total_tri = float(np.trace(_A3) / 6.0)
        _formula_note = (
            "(undirected: triangles per node = $(A^3)_{ii}/2$, "
            "total = $\\text{tr}(A^3)/6$)"
        )

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5.0))
    _ax1, _ax2 = _axes

    _order = np.argsort(-_tri_node)
    _ax1.bar(
        range(_n),
        [_tri_node[i] for i in _order],
        color=ACCENT, edgecolor="white",
    )
    if _n <= 30:
        _ax1.set_xticks(range(_n))
        _ax1.set_xticklabels([_names[i] for i in _order], rotation=90, fontsize=8)
    else:
        _ax1.set_xticks([])
    _ax1.set_title(f"Triangles through node   {_formula_note}")
    _ax1.set_ylabel("count")
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)

    # Right: node-link with size proportional to triangle count
    _ax2.set_facecolor("white")
    _ax2.grid(False)
    _ax2.set_xticks([])
    _ax2.set_yticks([])
    for _side in ("top", "right", "bottom", "left"):
        _ax2.spines[_side].set_visible(False)
    if _tri_node.max() > 0:
        _sizes = 10 + 35 * (_tri_node / _tri_node.max())
    else:
        _sizes = np.full(_n, 14.0)
    ig.plot(
        g,
        target=_ax2,
        layout=layout_coords,
        vertex_color=NEUTRAL_NODE,
        vertex_size=list(_sizes),
        vertex_frame_width=0,
        vertex_label=_names if _n <= 30 else [""] * _n,
        vertex_label_size=8,
        vertex_label_color="#222222",
        edge_color=EDGE_COLOR,
        edge_width=1.0,
        edge_arrow_size=0.7 if g.is_directed() else 0.0,
    )
    _ax2.set_title(
        f"Node size = triangles through node   ·   total = {_total_tri:.1f}"
    )
    plt.tight_layout()

    _top = int(np.argmax(_tri_node))
    _msg = mo.md(
        f"**Most triangular node:** {_names[_top]} with "
        f"{_tri_node[_top]:.1f} triangles. Total triangles in the graph: "
        f"**{_total_tri:.1f}**."
    )
    mo.vstack([_fig, _msg])
    return


# -----------------------------------------------------------------------------
# Section 6 — Power iteration converges to eigenvector centrality
# -----------------------------------------------------------------------------


@app.cell
def section6_header(mo):
    mo.md(r"""
    ---
    ## 6. Power iteration → eigenvector centrality

    Eigenvector centrality says: *you are important if you are
    connected to important people*. Formally, it is the leading
    eigenvector of $A$, i.e. the vector $x$ that satisfies

    $$ A x = \lambda x $$

    for the largest eigenvalue $\lambda$. You can find it without ever
    diagonalising $A$: just repeat $x \leftarrow A x$ and renormalise.
    Each step pushes mass toward the leading eigenvector; after a few
    iterations the bars stop moving.

    Move the slider and watch the bars converge. The dashed grey line
    shows the answer that igraph's `eigenvector_centrality()` gives —
    same idea, just iterated further.
    """)
    return


@app.cell
def s6_widgets(mo):
    iter_k = mo.ui.slider(
        start=1, stop=30, step=1, value=1,
        label="Iterations", show_value=True, full_width=True,
    )
    return (iter_k,)


@app.cell
def s6_plot(A, ACCENT, g, iter_k, mo, np, plt):
    _n = g.vcount()
    _names = g.vs["name"]
    _k = int(iter_k.value)

    _x = np.ones(_n) / _n
    _history = [_x.copy()]
    for _ in range(_k):
        _x = A @ _x
        _norm = np.linalg.norm(_x)
        if _norm > 0:
            _x = _x / _norm
        _history.append(_x.copy())

    try:
        _true_ev = np.array(g.eigenvector_centrality(directed=g.is_directed()))
        _true_ev = _true_ev / max(1e-12, np.linalg.norm(_true_ev))
        _true_ev = np.abs(_true_ev)
    except Exception:
        _true_ev = np.zeros(_n)

    _x = np.abs(_history[-1])
    _order = np.argsort(-_true_ev) if _true_ev.sum() > 0 else np.argsort(-_x)

    _fig, _ax = plt.subplots(figsize=(11, 3.8))
    _xs = np.arange(_n)
    _ax.bar(
        _xs - 0.2, [_x[i] for i in _order], width=0.4,
        color=ACCENT, edgecolor="white",
        label=f"x after {_k} iterations",
    )
    _ax.bar(
        _xs + 0.2, [_true_ev[i] for i in _order], width=0.4,
        color="#bbbbbb", edgecolor="white",
        label="igraph eigenvector_centrality()",
    )
    if _n <= 30:
        _ax.set_xticks(range(_n))
        _ax.set_xticklabels([_names[i] for i in _order], rotation=90, fontsize=8)
    else:
        _ax.set_xticks([])
    _ax.set_title("Power iteration vs. igraph eigenvector centrality")
    _ax.legend(frameon=False, fontsize=9)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    plt.tight_layout()

    _gap = float(np.linalg.norm(np.abs(_history[-1]) - _true_ev))
    _note = mo.md(
        f"After **{_k}** iterations the L2 distance between the iterate "
        f"and igraph's reference vector is **{_gap:.4f}**. Increase the "
        "slider and it should approach zero.\n\n"
        "_Same machinery, different questions:_ Katz centrality adds a "
        "constant $\\beta \\mathbf{1}$ at every step "
        "($x \\leftarrow \\alpha A x + \\beta \\mathbf{1}$); PageRank "
        "replaces $A$ with a row-normalised transition matrix and adds "
        "a teleport term."
    )
    mo.vstack([_fig, _note])
    return


# -----------------------------------------------------------------------------
# Section 7 — Random walks and PageRank
# -----------------------------------------------------------------------------


@app.cell
def section7_header(mo):
    mo.md(r"""
    ---
    ## 7. Random walks → PageRank

    Row-normalise the adjacency matrix to get a transition matrix
    $P = D^{-1} A$. Each row of $P$ sums to 1 — entry $P_{ij}$ is the
    probability of stepping from $i$ to $j$ if you pick a neighbour
    uniformly at random.

    Start a random walker at one node (mass = 1 there, 0 elsewhere) and
    iterate $x \leftarrow P^\top x$. After enough steps, $x$ converges
    to the stationary distribution. For undirected connected graphs
    it's proportional to degree; for directed graphs (try the *Directed
    toy*) it can get stuck in sinks — which is exactly what PageRank's
    teleport term fixes.
    """)
    return


@app.cell
def s7_widgets(g, mo):
    start_node = mo.ui.dropdown(
        options=list(g.vs["name"]),
        value=g.vs["name"][0],
        label="Walker starts at",
    )
    walk_steps = mo.ui.slider(
        start=0, stop=50, step=1, value=5,
        label="Steps t", show_value=True, full_width=True,
    )
    mo.hstack([start_node, walk_steps], gap=1.0, widths="equal")
    return start_node, walk_steps


@app.cell
def s7_plot(
    A, EDGE_COLOR, g, ig, is_directed, layout_coords, mo, np, plt,
    start_node, walk_steps,
):
    _n = g.vcount()
    _names = g.vs["name"]

    # Row-normalised transition matrix.  Handle dangling nodes (degree 0
    # in the out-direction) by leaving their row all zero — mass that
    # lands there gets stuck, which is the point of the section.
    _row_sums = A.sum(axis=1)
    _safe = np.where(_row_sums > 0, _row_sums, 1.0)
    P = A / _safe[:, None]
    P[_row_sums == 0] = 0.0

    _idx = _names.index(start_node.value)
    _x = np.zeros(_n)
    _x[_idx] = 1.0
    _t = int(walk_steps.value)
    for _ in range(_t):
        _x = P.T @ _x

    try:
        _pr = np.array(g.pagerank(directed=g.is_directed(), damping=0.85))
    except Exception:
        _pr = np.zeros(_n)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5.0))
    _axL, _axR = _axes

    # Left: node-link coloured by current mass
    _axL.set_facecolor("white")
    _axL.grid(False)
    _axL.set_xticks([])
    _axL.set_yticks([])
    for _side in ("top", "right", "bottom", "left"):
        _axL.spines[_side].set_visible(False)
    _vmax = max(1e-9, float(_x.max()))
    _cmap = plt.get_cmap("coolwarm")
    _colors = [_cmap(min(1.0, _x[i] / _vmax)) for i in range(_n)]
    _sizes = 12 + 30 * (_x / max(_vmax, 1e-9))
    ig.plot(
        g,
        target=_axL,
        layout=layout_coords,
        vertex_color=_colors,
        vertex_size=list(_sizes),
        vertex_frame_width=0,
        vertex_label=_names if _n <= 30 else [""] * _n,
        vertex_label_size=8,
        vertex_label_color="#222222",
        edge_color=EDGE_COLOR,
        edge_width=1.0,
        edge_arrow_size=0.7 if g.is_directed() else 0.0,
    )
    _axL.set_title(
        f"Mass at step t = {_t}  ·  total = {_x.sum():.3f}  "
        f"(start: {start_node.value})"
    )

    # Right: bar chart vs. PageRank
    _order = np.argsort(-_pr) if _pr.sum() > 0 else np.argsort(-_x)
    _xs = np.arange(_n)
    _axR.bar(
        _xs - 0.2, [_x[i] for i in _order], width=0.4,
        color="#0b789d", edgecolor="white", label=f"walker at t = {_t}",
    )
    _axR.bar(
        _xs + 0.2, [_pr[i] for i in _order], width=0.4,
        color="#bbbbbb", edgecolor="white", label="igraph pagerank()",
    )
    if _n <= 30:
        _axR.set_xticks(range(_n))
        _axR.set_xticklabels([_names[i] for i in _order], rotation=90, fontsize=8)
    else:
        _axR.set_xticks([])
    _axR.set_title("Walker mass vs. PageRank (teleport = 0.15)")
    _axR.legend(frameon=False, fontsize=9)
    _axR.spines["top"].set_visible(False)
    _axR.spines["right"].set_visible(False)
    plt.tight_layout()

    if is_directed:
        _note = (
            "On directed graphs the pure random walk can leak mass into "
            "sinks (rows of $A$ that sum to zero), so $\\sum x$ drops "
            "below 1 over time. PageRank prevents this by *teleporting* "
            "with probability 0.15 to a uniform node — that's the "
            "constant offset between the blue and grey bars."
        )
    else:
        _note = (
            "On a connected undirected graph the stationary distribution "
            "is **proportional to degree** — high-degree nodes get visited "
            "more often. PageRank applies a small uniform teleport on "
            "top, but for undirected graphs the two are nearly identical."
        )
    mo.vstack([_fig, mo.md(_note)])
    return


# -----------------------------------------------------------------------------
# Section 8 — Same network, different questions
# -----------------------------------------------------------------------------


@app.cell
def section8_header(mo):
    mo.md(r"""
    ---
    ## 8. Same network, different questions

    Each centrality below is a different question asked of the same
    adjacency matrix — and the matrix answers each one with a different
    ranking.

    - **Degree** = $A \mathbf{1}$. *How many friends do I have?*
    - **Eigenvector** = leading eigenvector of $A$. *Are my friends
      important?*
    - **PageRank** = stationary distribution of $P^\top$ with teleport.
      *Where does a random walker spend its time?*
    - **Betweenness** = fraction of shortest paths through me. *Am I a
      bridge?*
    - **Closeness** = inverse of average shortest-path distance. *Am I
      central in the small-world sense?*

    Five panels, one network. Where the rankings agree, you're seeing a
    structurally obvious centre. Where they disagree, you're seeing the
    *question* doing the work, not the network.
    """)
    return


@app.cell
def s8_grid(EDGE_COLOR, PALETTE, g, ig, layout_coords, mo, np, plt):
    _n = g.vcount()
    _names = g.vs["name"]

    def _safe(fn, fallback):
        try:
            v = np.array(fn(), dtype=float)
            if not np.all(np.isfinite(v)):
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return v
        except Exception:
            return np.array(fallback, dtype=float)

    def _ev():
        if g.is_connected(mode="weak") or _n == 0:
            return g.eigenvector_centrality(directed=g.is_directed())
        _comps = g.connected_components(mode="weak")
        _gc_ids = max(_comps, key=len)
        _gc = g.subgraph(_gc_ids)
        _vals = _gc.eigenvector_centrality(directed=g.is_directed())
        _out = [0.0] * _n
        for _i, _node in enumerate(_gc_ids):
            _out[_node] = _vals[_i]
        return _out

    centralities = {
        "Degree": np.array(g.degree(), dtype=float),
        "Eigenvector": _safe(_ev, [0] * _n),
        "PageRank": _safe(
            lambda: g.pagerank(directed=g.is_directed(), damping=0.85),
            [0] * _n,
        ),
        "Betweenness": _safe(lambda: g.betweenness(), [0] * _n),
        "Closeness": _safe(lambda: g.closeness(), [0] * _n),
    }

    _measures = list(centralities)
    _fig, _axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    _show_labels = _n <= 30
    _cmap = plt.get_cmap("coolwarm")

    for _ax, _name in zip(_axes.flat, _measures + [None]):
        _ax.set_facecolor("white")
        _ax.grid(False)
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _side in ("top", "right", "bottom", "left"):
            _ax.spines[_side].set_visible(False)
        if _name is None:
            _ax.set_visible(False)
            continue
        _c = np.asarray(centralities[_name], dtype=float)
        _rng = _c.max() - _c.min()
        if _rng > 0:
            _t = (_c - _c.min()) / _rng
        else:
            _t = np.zeros_like(_c)
        _colors = [_cmap(float(v)) for v in _t]
        ig.plot(
            g,
            target=_ax,
            layout=layout_coords,
            vertex_color=_colors,
            vertex_size=14,
            vertex_frame_width=0,
            vertex_label=_names if _show_labels else [""] * _n,
            vertex_label_size=8,
            edge_color=EDGE_COLOR,
            edge_width=0.9,
            edge_arrow_size=0.6 if g.is_directed() else 0.0,
        )
        _top = int(np.argmax(_c))
        _x, _y = layout_coords[_top]
        _ax.scatter(
            [_x], [_y], s=380, facecolors="none",
            edgecolors="#c0223b", linewidths=2.2, zorder=10,
        )
        _ax.set_title(
            f"{_name}  ·  top: {_names[_top]} ({_c[_top]:.3g})",
            fontsize=11,
        )

    _lines = ["| Centrality | 1st | 2nd | 3rd |", "|---|---|---|---|"]
    for _name in _measures:
        _c = centralities[_name]
        _order = np.argsort(-_c)[:3]
        _top = " | ".join(
            f"{_names[i]} ({_c[i]:.3g})" for i in _order
        )
        _lines.append(f"| {_name} | {_top} |")
    _table = mo.md("\n".join(_lines))

    mo.vstack([_fig, _table])
    return


@app.cell
def footer(mo):
    mo.md(r"""
    ---
    Network Science Summer School 2026 · Utrecht University ·
    standalone marimo companion app for Day 1b.
    """)
    return


if __name__ == "__main__":
    app.run()
