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
    EDGE_COLOR = (0.66, 0.66, 0.66, 0.8)

    # Tighter cap than the day1 intuition app — matrix displays do not
    # scale much past ~60 nodes.
    MAX_NODES = 60
    # Up to this size we print the actual numbers inside the matrix and
    # spell out the multiplications term by term. Florentine (16) is the
    # largest network that still gets the full numeric treatment.
    NUMBER_THRESHOLD = 16
    return (
        ACCENT, EDGE_COLOR, MAX_NODES, NEUTRAL_NODE, NUMBER_THRESHOLD,
        PALETTE, ig, io, mo, np, pd, plt, rnd_mod,
    )


# -----------------------------------------------------------------------------
# Inlined network catalogue
# -----------------------------------------------------------------------------


@app.cell
def network_catalogue(ig):
    # Five friends — the default. Small enough that every entry of A,
    # every term of every product, and every walk can be read directly.
    # A triangle (Alice, Bob, Carol) plus a tail (Carol-David-Emma).
    def _five_friends():
        g_ = ig.Graph(
            n=5,
            edges=[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)],
            directed=False,
        )
        g_.vs["name"] = ["Alice", "Bob", "Carol", "David", "Emma"]
        g_.vs["children"] = [2, 0, 3, 1, 4]
        return g_

    # A small directed toy so the asymmetric-matrix story is visible.
    def _toy_directed():
        g_ = ig.Graph(
            n=6,
            edges=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 1),
                   (4, 5), (5, 0)],
            directed=True,
        )
        g_.vs["name"] = ["A", "B", "C", "D", "E", "F"]
        return g_

    def _kite():
        g_ = ig.Graph.Famous("Krackhardt_Kite")
        g_.vs["name"] = [str(i) for i in range(g_.vcount())]
        return g_

    # Florentine families (Padgett 1994).
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

    def _florentine():
        g_ = ig.Graph(n=len(_flor_names), edges=list(_flor_edges),
                      directed=False)
        g_.vs["name"] = list(_flor_names)
        return g_

    def _karate():
        g_ = ig.Graph.Famous("Zachary")
        g_.vs["name"] = [str(i) for i in range(g_.vcount())]
        return g_

    BUNDLED = {
        "Five friends (5 nodes)": _five_friends,
        "Directed toy (6 nodes)": _toy_directed,
        "Krackhardt Kite (10 nodes)": _kite,
        "Florentine families (16 nodes)": _florentine,
        "Karate club (34 nodes)": _karate,
    }
    return (BUNDLED,)


# -----------------------------------------------------------------------------
# Sidebar: network chooser (visible from every section)
# -----------------------------------------------------------------------------


@app.cell
def chooser_widgets(BUNDLED, mo):
    bundled_choice = mo.ui.dropdown(
        options=list(BUNDLED.keys()),
        value="Five friends (5 nodes)",
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
def sidebar_cell(NUMBER_THRESHOLD, bundled_choice, file_upload, g, is_directed, mo, source_label, upload_warning):
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
        "_The active network drives every section below. Multiplications "
        f"are spelled out with real numbers for networks up to "
        f"{NUMBER_THRESHOLD} nodes — start with the tiny default, then "
        "switch to a bigger one to see that nothing changes except "
        "the size._"
    ))
    mo.sidebar(_items)
    return


# -----------------------------------------------------------------------------
# Shared computations: adjacency matrix and a fixed layout
# -----------------------------------------------------------------------------


@app.cell
def shared_A_and_layout(g, np):
    # Dense adjacency matrix. With MAX_NODES = 60 this is at most 3600
    # floats — cheap to recompute whenever the network changes.
    A = np.array(g.get_adjacency().data, dtype=float)

    # One layout per active network, reused by every section so nodes
    # never jump when a slider moves.
    try:
        _layout = g.layout_kamada_kawai()
    except Exception:
        _layout = g.layout_fruchterman_reingold(niter=500, seed=1)
    layout_coords = [tuple(row) for row in _layout.coords]
    return A, layout_coords


# -----------------------------------------------------------------------------
# Shared plotting helpers
# -----------------------------------------------------------------------------


@app.cell
def plot_helpers(EDGE_COLOR, NUMBER_THRESHOLD, ig, np, plt):
    def clean_axis(ax):
        ax.set_facecolor("white")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    def draw_graph(ax, g_, coords, vertex_color="#0b789d", vertex_size=None,
                   labels=None, edge_color=None, edge_width=1.2,
                   label_size=12):
        clean_axis(ax)
        n_ = g_.vcount()
        if vertex_size is None:
            vertex_size = 30 if n_ <= 16 else (18 if n_ <= 34 else 10)
        if labels is None:
            labels = g_.vs["name"] if n_ <= 30 else [""] * n_
        ig.plot(
            g_,
            target=ax,
            layout=coords,
            vertex_color=vertex_color,
            vertex_size=vertex_size,
            vertex_frame_width=0,
            vertex_label=labels,
            vertex_label_size=label_size,
            vertex_label_color="#222222",
            edge_color=edge_color if edge_color is not None else EDGE_COLOR,
            edge_width=edge_width,
            edge_arrow_size=0.8 if g_.is_directed() else 0.0,
        )

    def matrix_with_numbers(ax, M, names_, highlight_row=None,
                            highlight_cell=None, cmap="Blues", title=None):
        """Heatmap of M; prints the integers in the cells for small n."""
        n_ = M.shape[0]
        vmax = max(1.0, float(M.max()))
        cm = plt.get_cmap(cmap)
        ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        ax.grid(False)
        if n_ <= 30:
            ax.set_xticks(range(n_))
            ax.set_yticks(range(n_))
            ax.set_xticklabels(names_, rotation=90, fontsize=9)
            ax.set_yticklabels(names_, fontsize=9)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        if n_ <= NUMBER_THRESHOLD:
            fs = 12 if n_ <= 8 else 9
            for ri in range(n_):
                for ci in range(n_):
                    v = M[ri, ci]
                    red, grn, blu, _ = cm(v / vmax)
                    lum = 0.299 * red + 0.587 * grn + 0.114 * blu
                    if v == 0:
                        col = "#bbbbbb" if lum > 0.5 else "#888888"
                    else:
                        col = "white" if lum < 0.5 else "#333333"
                    ax.text(ci, ri, f"{int(v)}", ha="center", va="center",
                            fontsize=fs, color=col)
        if highlight_row is not None:
            ax.add_patch(plt.Rectangle(
                (-0.5, highlight_row - 0.5), n_, 1,
                fill=False, edgecolor="#c0223b", linewidth=2.0, zorder=5,
            ))
        if highlight_cell is not None:
            hri, hci = highlight_cell
            ax.add_patch(plt.Rectangle(
                (hci - 0.5, hri - 0.5), 1, 1,
                fill=False, edgecolor="#1ed8a3", linewidth=2.5, zorder=6,
            ))
        if title:
            ax.set_title(title)

    def draw_block(ax, M, x0, names_, fs, row_names=False, col_names=False,
                   hl_row=None, hl_col=None, hl_cell=None,
                   hl_row_color="#c0223b", hl_col_color="#0b789d",
                   hl_cell_color="#1ed8a3"):
        """Draw matrix M as a numeric grid starting at x = x0.

        Row 0 is at the top (caller must invert the y axis). Returns the
        x coordinate of the right edge of the block.
        """
        nr, nc = M.shape
        for r in range(nr + 1):
            ax.plot([x0, x0 + nc], [r, r], color="#cccccc",
                    linewidth=0.6, zorder=1)
        for c in range(nc + 1):
            ax.plot([x0 + c, x0 + c], [0, nr], color="#cccccc",
                    linewidth=0.6, zorder=1)
        if hl_row is not None:
            ax.add_patch(plt.Rectangle(
                (x0, hl_row), nc, 1, facecolor=hl_row_color, alpha=0.12,
                edgecolor=hl_row_color, linewidth=1.8, zorder=2,
            ))
        if hl_col is not None:
            ax.add_patch(plt.Rectangle(
                (x0 + hl_col, 0), 1, nr, facecolor=hl_col_color,
                alpha=0.12, edgecolor=hl_col_color, linewidth=1.8,
                zorder=2,
            ))
        if hl_cell is not None:
            cr, cc = hl_cell
            ax.add_patch(plt.Rectangle(
                (x0 + cc, cr), 1, 1, fill=False,
                edgecolor=hl_cell_color, linewidth=2.5, zorder=4,
            ))
        for r in range(nr):
            for c in range(nc):
                v = M[r, c]
                ax.text(x0 + c + 0.5, r + 0.5, f"{v:g}",
                        ha="center", va="center", fontsize=fs,
                        color="#aaaaaa" if v == 0 else "#222222", zorder=3)
        if row_names:
            for r in range(nr):
                ax.text(x0 - 0.3, r + 0.5, names_[r], ha="right",
                        va="center", fontsize=fs, color="#333333")
        if col_names:
            for c in range(nc):
                ax.text(x0 + c + 0.5, -0.3, names_[c], ha="center",
                        va="bottom", rotation=90, fontsize=fs,
                        color="#333333")
        return x0 + nc

    def enumerate_walks(A_, start, end, k, cap=200):
        """All walks of length k from start to end (node index lists)."""
        n_ = A_.shape[0]
        out = []

        def dfs(path):
            if len(out) >= cap:
                return
            if len(path) == k + 1:
                if path[-1] == end:
                    out.append(list(path))
                return
            for nxt in range(n_):
                if A_[path[-1], nxt] > 0:
                    path.append(nxt)
                    dfs(path)
                    path.pop()

        dfs([start])
        return out

    return (
        clean_axis, draw_block, draw_graph, enumerate_walks,
        matrix_with_numbers,
    )


# -----------------------------------------------------------------------------
# Title
# -----------------------------------------------------------------------------


@app.cell
def title(mo):
    mo.md(r"""
    # Matrix multiplication is counting paths

    Javier Garcia-Bernardo — ODISSEI Social Data Science team (SoDa) &
    Department of Methodology and Statistics, Utrecht University

    A network can be written as a matrix $A$: one row and one column per
    node, with $A_{ij} = 1$ when $i$ and $j$ are connected. That is just
    storage. The magic starts when you **multiply**:

    - `A @ x` asks every node a question about its **neighbours** (1 step).
    - `A @ A` counts **walks of length 2** — friends of friends.
    - The diagonal of `A @ A @ A` counts **triangles**.
    - Multiplying again and again makes a notion of **importance** emerge.

    The same trick — follow the paths, add them up — is behind search
    ranking, epidemic models, recommendation systems, and community
    detection. Learn to read one multiplication and you can read them all.

    The default network is five people, so every number in every
    multiplication is visible. Nothing is hidden inside the math: you can
    check each product with your finger on the picture.
    """)
    return


# -----------------------------------------------------------------------------
# Section 1 — The network and its matrix
# -----------------------------------------------------------------------------


@app.cell
def section1_header(mo):
    mo.md(r"""
    ---
    ## 1. The network and its matrix

    The picture and the matrix carry exactly the same information.
    **Row $i$ answers one question: who is $i$ connected to?** A 1 in
    column $j$ means "yes, $j$". The diagonal is all zeros (nobody is
    their own neighbour), and for an undirected network the matrix is
    symmetric: if Alice knows Bob, Bob knows Alice.

    Pick a node and find its row.
    """)
    return


@app.cell
def s1_widgets(g, mo):
    row_node = mo.ui.dropdown(
        options=list(g.vs["name"]),
        value=g.vs["name"][0],
        label="Pick a node",
    )
    row_node
    return (row_node,)


@app.cell
def s1_plot(A, ACCENT, EDGE_COLOR, draw_graph, g, layout_coords, matrix_with_numbers, mo, np, plt, row_node):
    _n = g.vcount()
    _names = list(g.vs["name"])
    _f = _names.index(row_node.value)

    _fig, (_axL, _axR) = plt.subplots(1, 2, figsize=(11, 5.2))

    # Left: graph with the chosen node and its edges highlighted.
    _ecols = []
    _ewids = []
    for _e in g.es:
        if _f in (_e.source, _e.target):
            _ecols.append(ACCENT)
            _ewids.append(3.0)
        else:
            _ecols.append(EDGE_COLOR)
            _ewids.append(1.2)
    _vcols = ["#c8dde6"] * _n
    _vcols[_f] = ACCENT
    draw_graph(_axL, g, layout_coords, vertex_color=_vcols,
               edge_color=_ecols, edge_width=_ewids)
    _axL.set_title(f"{row_node.value} and its edges")

    # Right: the matrix with that row boxed.
    matrix_with_numbers(_axR, A, _names, highlight_row=_f,
                        title="Adjacency matrix A")
    plt.tight_layout()

    _row = A[_f]
    _nbrs = [_names[_j] for _j in range(_n) if _row[_j] > 0]
    _sym = bool(np.array_equal(A, A.T))
    _read = (
        f"**Row {row_node.value}** = "
        f"[{', '.join(str(int(_v)) for _v in _row)}]"
        if _n <= 16 else f"**Row {row_node.value}**"
    )
    _msg = mo.md(
        f"{_read} — connected to "
        f"{', '.join('**' + _b + '**' for _b in _nbrs) if _nbrs else 'nobody'}."
        f" The row sum is {int(_row.sum())}: that is the **degree**"
        f"{' (out-degree, since this network is directed)' if g.is_directed() else ''}."
        f" This matrix is {'symmetric — undirected' if _sym else 'NOT symmetric — directed: row = who I point to, column = who points to me'}."
        f" Trace (sum of the diagonal) = {int(np.trace(A))}."
    )
    mo.vstack([_fig, _msg])
    return


# -----------------------------------------------------------------------------
# Section 2 — A @ x: ask every node a question about its neighbours
# -----------------------------------------------------------------------------


@app.cell
def section2_header(mo):
    mo.md(r"""
    ---
    ## 2. `A @ x` — ask every node about its neighbours

    Multiply $A$ by a vector $x$ (one value per node):

    $$
    y = A\,x
    $$

    Entry $i$ of the result is **row $i$ of $A$ times the vector $x$**:
    the 1s in the row keep exactly the neighbours' values, the 0s erase
    everyone else. So $y_i$ is the total of $x$ over $i$'s neighbours —
    no loops, no code, one multiplication.

    Try it with the number of children each person has: $y$ answers
    "how many children do my friends have, in total?" for everybody at
    once. With $x$ = all ones, each neighbour contributes 1, so $y$ is
    the degree.
    """)
    return


@app.cell
def s2_widgets(g, mo):
    _opts = []
    if "children" in g.vs.attributes():
        _opts.append("Number of children (attribute)")
    _opts.append("All ones (counts neighbours = degree)")
    _opts.append("Example attribute (random 0-5)")
    x_kind = mo.ui.dropdown(
        options=_opts, value=_opts[0], label="Vector x",
    )
    focus_node = mo.ui.dropdown(
        options=list(g.vs["name"]),
        value=g.vs["name"][0],
        label="Spell out the multiplication for",
    )
    mo.hstack([x_kind, focus_node], gap=1.0, widths="equal")
    return focus_node, x_kind


@app.cell
def s2_plot(A, ACCENT, EDGE_COLOR, NUMBER_THRESHOLD, clean_axis, draw_block, draw_graph, focus_node, g, layout_coords, mo, np, plt, x_kind):
    _n = g.vcount()
    _names = list(g.vs["name"])
    _f = _names.index(focus_node.value)

    if x_kind.value == "Number of children (attribute)":
        _x = np.array(g.vs["children"], dtype=float)
        _xlabel = "x = number of children"
    elif x_kind.value == "All ones (counts neighbours = degree)":
        _x = np.ones(_n)
        _xlabel = "x = all ones"
    else:
        _rng = np.random.default_rng(7)
        _x = _rng.integers(0, 6, _n).astype(float)
        _xlabel = "x = example attribute (random 0-5)"

    _y = A @ _x

    # Figure 1: graph (x values on the nodes) + bar charts of x and y.
    _fig1 = plt.figure(figsize=(11.5, 5.2))
    _gs = _fig1.add_gridspec(2, 2, width_ratios=[1.5, 1.0])
    _axG = _fig1.add_subplot(_gs[:, 0])
    _axX = _fig1.add_subplot(_gs[0, 1])
    _axY = _fig1.add_subplot(_gs[1, 1])

    _ecols = []
    _ewids = []
    for _e in g.es:
        if _f in (_e.source, _e.target):
            _ecols.append(ACCENT)
            _ewids.append(3.0)
        else:
            _ecols.append(EDGE_COLOR)
            _ewids.append(1.2)
    _vcols = ["#c8dde6"] * _n
    _vcols[_f] = ACCENT
    if _n <= NUMBER_THRESHOLD:
        _labels = [f"{_names[_i]}\nx={_x[_i]:g}" for _i in range(_n)]
    else:
        _labels = None
    draw_graph(_axG, g, layout_coords, vertex_color=_vcols,
               edge_color=_ecols, edge_width=_ewids, labels=_labels,
               label_size=11)
    _axG.set_title(f"Each node carries its x value  ({_xlabel})")

    _show_ticks = _n <= 16
    _axX.bar(range(_n), _x, color="#888888", edgecolor="white")
    _axX.set_title("Input  x")
    _axY.bar(range(_n), _y, color=ACCENT, edgecolor="white")
    _axY.set_title("Output  y = A @ x")
    for _ax in (_axX, _axY):
        if _show_ticks:
            _ax.set_xticks(range(_n))
            _ax.set_xticklabels(_names, rotation=90, fontsize=9)
        else:
            _ax.set_xticks([])
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    plt.tight_layout()

    # Figure 2: the multiplication itself — A (row highlighted) @ x = y.
    if _n <= NUMBER_THRESHOLD:
        _fs = 12 if _n <= 6 else (10 if _n <= 10 else 7.5)
        _maxlen = max(len(_nm) for _nm in _names)
        _left_margin = 0.35 + 0.21 * _maxlen * _fs / 9.0
        _top_margin = 0.6 + 0.21 * _maxlen * _fs / 9.0
        _gap = 1.7

        _figW = 11.5
        _x_total = _left_margin + _n + _gap + 1 + _gap + 1 + 1.8
        _y_total = _n + 2.6 + _top_margin
        _figH = min(9.0, max(3.4, _figW * _y_total / _x_total))
        _fig2, _axM = plt.subplots(figsize=(_figW, _figH))
        clean_axis(_axM)
        _axM.set_aspect("equal")

        _xe = draw_block(_axM, A, 0, _names, _fs, row_names=True,
                         col_names=True, hl_row=_f)
        _axM.text(_xe + _gap / 2, _n / 2, "@", fontsize=15,
                  ha="center", va="center", color="#333333")
        _xv = _xe + _gap
        draw_block(_axM, _x.reshape(-1, 1), _xv, _names, _fs, hl_col=0)
        _axM.text(_xv + 1 + _gap / 2, _n / 2, "=", fontsize=15,
                  ha="center", va="center", color="#333333")
        _xy = _xv + 1 + _gap
        draw_block(_axM, _y.reshape(-1, 1), _xy, _names, _fs,
                   hl_cell=(_f, 0))

        # The products, written under the column they come from.
        for _m in range(_n):
            _zero = A[_f, _m] == 0
            _axM.text(
                _m + 0.5, _n + 0.85,
                f"{A[_f, _m]:g}*{_x[_m]:g}",
                ha="center", va="center", fontsize=_fs * 0.92,
                color="#bbbbbb" if _zero else "#c0223b",
            )
            if _m < _n - 1:
                _axM.text(_m + 1.0, _n + 0.85, "+", ha="center",
                          va="center", fontsize=_fs * 0.92,
                          color="#bbbbbb")
        _axM.text(_xy + 0.5, _n + 0.85, f"= {_y[_f]:g}",
                  ha="center", va="center", fontsize=_fs,
                  fontweight="bold", color="#c0223b")
        _axM.text(
            0, _n + 1.9,
            f"y({focus_node.value})  =  row {focus_node.value} of A  "
            f"times  x   (red row, blue vector)",
            ha="left", va="center", fontsize=_fs, color="#333333",
        )
        _axM.set_xlim(-_left_margin, _xy + 1.9)
        _axM.set_ylim(_n + 2.6, -_top_margin)
        _body = mo.vstack([_fig1, _fig2])
    else:
        _body = mo.vstack([_fig1, mo.md(
            f"_Switch to a network with at most {NUMBER_THRESHOLD} nodes "
            "to see the multiplication spelled out — the math is "
            "identical._"
        )])

    if x_kind.value == "All ones (counts neighbours = degree)":
        _note = (
            "With $x$ = all ones every neighbour contributes exactly 1, "
            "so `A @ x` = number of neighbours = **degree**. One "
            "multiplication computed the degree of every node at once."
        )
    else:
        _nbrs = [_names[_j] for _j in range(_n) if A[_f, _j] > 0]
        _note = (
            f"Check it on the picture: {focus_node.value}'s neighbours "
            f"are {', '.join(_nbrs) if _nbrs else 'nobody'}; add their x "
            f"values and you get {_y[_f]:g}. The matrix did this for "
            "every node simultaneously — that is all matrix "
            "multiplication is."
        )
    mo.vstack([_body, mo.md(_note)])
    return


# -----------------------------------------------------------------------------
# Section 3 — Average of friends and the friendship paradox
# -----------------------------------------------------------------------------


@app.cell
def section3_header(mo):
    mo.md(r"""
    ---
    ## 3. Average of friends, and the friendship paradox

    Divide the total by the number of neighbours (the degree) and you
    get an **average over friends**: `(A @ x) / k`, elementwise.

    Now feed the machine its own degrees: take $x = k$. Then each entry
    is "the average degree of my friends". Comparing it with your own
    degree gives a famous surprise — **on average, your friends have
    more friends than you**. Popular people show up in many friend
    lists, so they drag every list's average up.
    """)
    return


@app.cell
def s3_plot(A, ACCENT, draw_graph, g, layout_coords, mo, np, plt):
    _n = g.vcount()
    _names = list(g.vs["name"])
    _deg = A.sum(axis=1)
    _safe = np.where(_deg > 0, _deg, 1.0)
    _avg_nbr = (A @ _deg) / _safe
    _avg_nbr[_deg == 0] = np.nan

    _valid = ~np.isnan(_avg_nbr)
    _mean_self = float(_deg[_valid].mean()) if _valid.any() else 0.0
    _mean_nbr = float(np.nanmean(_avg_nbr)) if _valid.any() else 0.0

    # The network, with each node labelled by its degree.
    _figG, _axG = plt.subplots(figsize=(6.0, 5.2))
    if _deg.max() > 0:
        _sizes = list(20 + 22 * (_deg / _deg.max()))
    else:
        _sizes = None
    if _n <= 16:
        _labels = [f"{_names[_i]}\nk={int(_deg[_i])}" for _i in range(_n)]
    else:
        _labels = [""] * _n
    draw_graph(_axG, g, layout_coords, vertex_color=ACCENT,
               vertex_size=_sizes, labels=_labels, label_size=11)
    _axG.set_title("Node size = degree")
    plt.tight_layout()

    if _n <= 12:
        # Numbers-first: a table you can check by hand.
        _rows = ["| Node | Own degree | Friends (their degrees) | Friends' average |",
                 "|---|---|---|---|"]
        for _i in range(_n):
            _nbrs = [_j for _j in range(_n) if A[_i, _j] > 0]
            _flist = ", ".join(
                f"{_names[_j]} ({int(_deg[_j])})" for _j in _nbrs
            ) if _nbrs else "—"
            _avg = f"{_avg_nbr[_i]:.2f}" if _nbrs else "—"
            _winner = " **<**" if _nbrs and _avg_nbr[_i] > _deg[_i] else ""
            _rows.append(
                f"| {_names[_i]} | {int(_deg[_i])}{_winner} | {_flist} | {_avg} |"
            )
        _body = mo.hstack(
            [_figG, mo.md("\n".join(_rows))],
            widths=[1.0, 1.4], gap=1.0, align="center",
        )
    else:
        # Bigger networks: the two distributions next to the graph.
        _kmax = int(max(_deg.max(), np.nanmax(_avg_nbr)))
        _bins = np.arange(0, _kmax + 2) - 0.5
        _figH, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))
        _ax1.hist(_deg, bins=_bins, color="#888888", edgecolor="white")
        _ax1.axvline(_mean_self, color="#c0223b", linestyle="--",
                     linewidth=1.2, label=f"mean = {_mean_self:.2f}")
        _ax1.set_title("Your degree  k")
        _ax1.set_xlabel("k")
        _ax1.legend(frameon=False, fontsize=9)
        _ax2.hist(_avg_nbr[_valid], bins=_bins, color=ACCENT,
                  edgecolor="white")
        _ax2.axvline(_mean_nbr, color="#c0223b", linestyle="--",
                     linewidth=1.2, label=f"mean = {_mean_nbr:.2f}")
        _ax2.set_title("Friends' average degree")
        _ax2.set_xlabel("avg. neighbour degree")
        _ax2.legend(frameon=False, fontsize=9)
        for _ax in (_ax1, _ax2):
            _ax.spines["top"].set_visible(False)
            _ax.spines["right"].set_visible(False)
        plt.tight_layout()
        _body = mo.hstack(
            [_figG, _figH], widths=[0.9, 1.4], gap=1.0, align="center",
        )

    _diff = _mean_nbr - _mean_self
    if _diff > 1e-9:
        _verdict = (
            "your friends have more friends than you — the paradox holds"
        )
    elif _diff < -1e-9:
        _verdict = "friends average fewer — unusual, degree-equal networks can do this"
    else:
        _verdict = "exactly equal — every node has the same degree"
    _msg = mo.md(
        f"Mean degree = **{_mean_self:.2f}**; mean of friends' averages = "
        f"**{_mean_nbr:.2f}** — {_verdict}. No psychology involved: it is "
        "pure structure, and it fell out of one multiplication and one "
        "division."
    )
    mo.vstack([_body, _msg])
    return


# -----------------------------------------------------------------------------
# Section 4 — A @ A: a row times a column
# -----------------------------------------------------------------------------


@app.cell
def section4_header(mo):
    mo.md(r"""
    ---
    ## 4. `A @ A` — a row times a column

    What happens when the thing you multiply $A$ by is $A$ itself? Cell
    $(i, j)$ of `A @ A` is **row $i$ of the first matrix times column
    $j$ of the second**: pair up the entries, multiply each pair, add
    everything up.

    Each pair asks: *does $i$ know $m$* (row), *and does $m$ know $j$*
    (column)? Only when both are 1 does the pair contribute — and that
    pair **is** a stepping stone: a walk $i \to m \to j$ of length 2.
    Cell $(i, j)$ of `A @ A` therefore counts the walks of length 2 —
    the friends that $i$ and $j$ have in common.
    """)
    return


@app.cell
def s4_widgets(g, mo):
    _names = list(g.vs["name"])
    aa_i = mo.ui.dropdown(
        options=_names, value=_names[0], label="Row i (first matrix)",
    )
    aa_j = mo.ui.dropdown(
        options=_names, value=_names[min(2, len(_names) - 1)],
        label="Column j (second matrix)",
    )
    mo.hstack([aa_i, aa_j], gap=1.0, widths="equal")
    return aa_i, aa_j


@app.cell
def s4_plot(A, NUMBER_THRESHOLD, aa_i, aa_j, clean_axis, draw_block, g, mo, np, plt):
    _n = g.vcount()
    _names = list(g.vs["name"])
    _i = _names.index(aa_i.value)
    _j = _names.index(aa_j.value)
    _A2 = A @ A
    _val = _A2[_i, _j]

    if _n <= NUMBER_THRESHOLD:
        _fs = 11 if _n <= 6 else (9 if _n <= 10 else 6.5)
        _maxlen = max(len(_nm) for _nm in _names)
        _left_margin = 0.35 + 0.21 * _maxlen * _fs / 9.0
        _top_margin = 0.6 + 0.21 * _maxlen * _fs / 9.0
        _gap = 1.7

        _figW = 12.0
        _x_total = _left_margin + 3 * _n + 2 * _gap + 0.8
        _y_total = _n + 2.6 + _top_margin
        _figH = min(9.5, max(3.2, _figW * _y_total / _x_total))
        _fig, _ax = plt.subplots(figsize=(_figW, _figH))
        clean_axis(_ax)
        _ax.set_aspect("equal")

        _xe = draw_block(_ax, A, 0, _names, _fs, row_names=True,
                         col_names=True, hl_row=_i)
        _ax.text(_xe + _gap / 2, _n / 2, "@", fontsize=15,
                 ha="center", va="center", color="#333333")
        _x2 = _xe + _gap
        _xe2 = draw_block(_ax, A, _x2, _names, _fs, col_names=True,
                          hl_col=_j)
        _ax.text(_xe2 + _gap / 2, _n / 2, "=", fontsize=15,
                 ha="center", va="center", color="#333333")
        _x3 = _xe2 + _gap
        draw_block(_ax, _A2, _x3, _names, _fs, col_names=True,
                   hl_cell=(_i, _j))

        # The products, under the columns of the FIRST matrix: entry m
        # of the red row times entry m of the blue column.
        for _m in range(_n):
            _prod = A[_i, _m] * A[_m, _j]
            _ax.text(
                _m + 0.5, _n + 0.85,
                f"{A[_i, _m]:g}*{A[_m, _j]:g}",
                ha="center", va="center", fontsize=_fs * 0.92,
                color="#222222" if _prod > 0 else "#bbbbbb",
            )
            if _m < _n - 1:
                _ax.text(_m + 1.0, _n + 0.85, "+", ha="center",
                         va="center", fontsize=_fs * 0.92,
                         color="#bbbbbb")
        _ax.text(_xe + 0.6, _n + 0.85, f"= {_val:g}",
                 ha="left", va="center", fontsize=_fs,
                 fontweight="bold", color="#0e8c6d")
        _ax.text(
            0, _n + 1.9,
            f"(A @ A)({aa_i.value}, {aa_j.value})  =  "
            f"red row times blue column  =  {_val:g}",
            ha="left", va="center", fontsize=_fs, color="#333333",
        )
        _ax.set_xlim(-_left_margin, _x3 + _n + 0.8)
        _ax.set_ylim(_n + 2.6, -_top_margin)
        _body = _fig
    else:
        _body = mo.md(
            f"(A @ A)({aa_i.value}, {aa_j.value}) = **{_val:g}**. "
            f"_Switch to a network with at most {NUMBER_THRESHOLD} nodes "
            "to see the two matrices spelled out._"
        )

    _common = [
        _names[_m] for _m in range(_n) if A[_i, _m] * A[_m, _j] > 0
    ]
    if _i == _j:
        _interp = (
            f"You picked the diagonal: every neighbour of "
            f"{aa_i.value} is a walk out and straight back, so the cell "
            f"equals the **degree** ({_val:g})."
        )
    elif _common:
        _interp = (
            f"The nonzero pairs are the stepping stones: "
            f"{', '.join('**' + _c + '**' for _c in _common)}. "
            f"{aa_i.value} and {aa_j.value} share {_val:g} "
            f"common neighbour{'s' if _val != 1 else ''}."
        )
    else:
        _interp = (
            f"No pair is 1 * 1, so {aa_i.value} and {aa_j.value} share "
            "no neighbours: no walk of length 2 connects them."
        )
    mo.vstack([_body, mo.md(_interp)])
    return


# -----------------------------------------------------------------------------
# Section 5 — Powers of A: walks, reach, triangles
# -----------------------------------------------------------------------------


@app.cell
def section5_header(mo):
    mo.md(r"""
    ---
    ## 5. Keep multiplying: `A @ A @ A` and beyond

    Multiplying once more repeats the trick: cell $(i, j)$ of $A^k$
    (that is, `A @ A @ ... @ A`, $k$ times) counts the **walks of
    length $k$** from $i$ to $j$ — walks may revisit nodes. One number
    in a matrix, and you can draw every walk it counts on the network.

    Three things fall out of this for free: the walks themselves, the
    set of nodes you can reach in at most $k$ steps, and — on the
    diagonal of $A^3$ — the triangles.
    """)
    return


@app.cell
def s5_widgets(g, mo):
    _names = list(g.vs["name"])
    walk_i = mo.ui.dropdown(
        options=_names, value=_names[0], label="From i",
    )
    walk_j = mo.ui.dropdown(
        options=_names, value=_names[min(2, len(_names) - 1)], label="To j",
    )
    power_k = mo.ui.slider(
        start=1, stop=5, step=1, value=2,
        label="Walk length k", show_value=True, full_width=True,
    )
    mo.hstack([walk_i, walk_j, power_k], gap=1.0, widths="equal")
    return power_k, walk_i, walk_j


@app.cell
def s5_compute(A, np, power_k):
    _k = int(power_k.value)
    _M = np.eye(A.shape[0])
    for _ in range(_k - 1):
        _M = _M @ A
    Akm1 = _M          # A^(k-1), used to spell out the multiplication
    Ak = _M @ A        # A^k
    return Ak, Akm1


@app.cell
def s5_walks_plot(
    A, ACCENT, Ak, Akm1, NUMBER_THRESHOLD, PALETTE, clean_axis, draw_graph,
    enumerate_walks, g, layout_coords, matrix_with_numbers, mo, np,
    plt, power_k, walk_i, walk_j,
):
    import matplotlib.patches as _mpatches
    from matplotlib.path import Path as _MplPath

    _n = g.vcount()
    _names = list(g.vs["name"])
    _i = _names.index(walk_i.value)
    _j = _names.index(walk_j.value)
    _k = int(power_k.value)
    _count = int(round(Ak[_i, _j]))

    # The matrix already knows the count; only enumerate when nonzero.
    _walks = enumerate_walks(A, _i, _j, _k, cap=200) if _count > 0 else []
    _MAX_DRAWN = 20
    _drawn = _walks[:_MAX_DRAWN]

    # Walk colour = colour of its last intermediate node (the m in the
    # multiplication), so drawn walks and terms match by colour.
    def _walk_color(w):
        if len(w) < 3:
            return ACCENT
        return PALETTE[w[-2] % len(PALETTE)]

    _fig, (_axG, _axE) = plt.subplots(1, 2, figsize=(12, 5.6))

    # Left: the graph with walks drawn as bowed coloured curves.
    draw_graph(_axG, g, layout_coords, vertex_color="#c8dde6")
    _xs = [c[0] for c in layout_coords]
    _ys = [c[1] for c in layout_coords]
    _span = max(max(_xs) - min(_xs), max(_ys) - min(_ys), 1e-9)
    for _t, _w in enumerate(_drawn):
        _col = _walk_color(_w)
        _bow = ((_t % 5) - 2) * 0.055 * _span
        for _a, _b in zip(_w[:-1], _w[1:]):
            _p = np.array(layout_coords[_a])
            _q = np.array(layout_coords[_b])
            _mid = (_p + _q) / 2
            _d = _q - _p
            _nrm = float(np.hypot(*_d)) or 1.0
            _perp = np.array([-_d[1], _d[0]]) / _nrm
            _ctrl = _mid + _perp * _bow
            _path = _MplPath(
                [tuple(_p), tuple(_ctrl), tuple(_q)],
                [_MplPath.MOVETO, _MplPath.CURVE3, _MplPath.CURVE3],
            )
            _axG.add_patch(_mpatches.PathPatch(
                _path, fill=False, edgecolor=_col,
                linewidth=2.2, alpha=0.85, zorder=4,
            ))
    _xi, _yi = layout_coords[_i]
    _xj, _yj = layout_coords[_j]
    _axG.scatter([_xi], [_yi], s=520, facecolors="none",
                 edgecolors=ACCENT, linewidths=2.5, zorder=8)
    _axG.scatter([_xj], [_yj], s=380, facecolors="none",
                 edgecolors="#c0223b", linewidths=2.5, zorder=8)
    _extra = f"  (drawing {len(_drawn)} of {_count})" if _count > len(_drawn) else ""
    _axG.set_title(
        f"{_count} walk{'s' if _count != 1 else ''} of length {_k}: "
        f"{walk_i.value} (blue ring) to {walk_j.value} (red ring){_extra}"
    )

    # Right: the multiplication, one line per stepping stone m.
    clean_axis(_axE)
    if _k == 1:
        _v = int(round(A[_i, _j]))
        _axE.text(
            0.05, 0.6,
            f"A({walk_i.value}, {walk_j.value}) = {_v}",
            family="monospace", fontsize=13, fontweight="bold",
            transform=_axE.transAxes,
        )
        _axE.text(
            0.05, 0.45,
            "Length 1 is just the matrix entry:\n"
            + ("there IS a direct edge." if _v else "no direct edge."),
            fontsize=11, va="top", transform=_axE.transAxes,
        )
    elif _n <= NUMBER_THRESHOLD:
        _show_all = _n <= 8
        _w = max(len(_nm) for _nm in _names)
        _lhs = "A" if _k == 2 else f"A^{_k - 1}"
        _lines = [(
            f"(A^{_k})({walk_i.value},{walk_j.value}) = "
            f"row of {_lhs} times column of A:",
            "#333333", True,
        )]
        _skipped = 0
        for _m in range(_n):
            _a = Akm1[_i, _m]
            _b = A[_m, _j]
            _prod = _a * _b
            if _prod == 0 and not _show_all:
                _skipped += 1
                continue
            _col = PALETTE[_m % len(PALETTE)] if _prod > 0 else "#bbbbbb"
            _lines.append((
                f"  via {_names[_m]:<{_w}}  {int(_a)} * {int(_b)} = {int(_prod)}",
                _col, _prod > 0,
            ))
        if _skipped:
            _lines.append((f"  (+ {_skipped} terms that are 0)",
                           "#bbbbbb", False))
        _lines.append((
            f"  total = {_count} walk{'s' if _count != 1 else ''}",
            "#c0223b", True,
        ))
        _dy = 1.0 / (len(_lines) + 1)
        for _t, (_txt, _col, _bold) in enumerate(_lines):
            _axE.text(
                0.02, 1.0 - (_t + 1) * _dy, _txt,
                family="monospace", fontsize=10, color=_col,
                fontweight="bold" if _bold else "normal",
                transform=_axE.transAxes,
            )
        _axE.set_title("One line per stepping stone m")
    else:
        _axE.text(0.5, 0.5,
                  f"(A^{_k})({walk_i.value}, {walk_j.value}) = {_count}\n\n"
                  "Switch to a network with at most\n"
                  f"{NUMBER_THRESHOLD} nodes to see the terms.",
                  ha="center", va="center", fontsize=11)
    plt.tight_layout()

    # The full A^k matrix, with the chosen cell boxed.
    _figM, _axM = plt.subplots(figsize=(6.8, 6.2))
    matrix_with_numbers(
        _axM, Ak, _names, highlight_cell=(_i, _j), cmap="magma",
        title=f"A^{_k} — every cell counts walks of length {_k}",
    )
    plt.tight_layout()

    _walk_texts = [
        " -> ".join(_names[_v] for _v in _w) for _w in _drawn[:8]
    ]
    _list_md = (
        ("Walks drawn: " + ";  ".join(_walk_texts)
         + (" ; ..." if _count > 8 else "") + ".")
        if _count > 0 else
        f"No walk of length {_k} connects {walk_i.value} to {walk_j.value}."
    )
    mo.vstack([_fig, _figM, mo.md(_list_md)])
    return


@app.cell
def s5_reach_header(mo):
    mo.md(r"""
    ### Reachable in at most $k$ steps

    Add the powers up: a cell of `A + A@A + A@A@A + ...` is positive
    exactly when SOME walk of length at most $k$ connects the pair.
    The node and the walk length $k$ above drive this picture too —
    this is how "six degrees of separation" and epidemic horizons are
    computed.
    """)
    return


@app.cell
def s5_reach_plot(A, draw_graph, g, layout_coords, mo, np, plt, power_k, walk_i):
    _n = g.vcount()
    _names = list(g.vs["name"])
    _s = _names.index(walk_i.value)
    _kmax = int(power_k.value)

    # First step at which each node becomes reachable, via matrix
    # powers: (A^k)[s, v] > 0 and no smaller power reached it yet.
    _first = np.full(_n, -1, dtype=int)
    _first[_s] = 0
    _Acur = np.eye(_n)
    for _step in range(1, _kmax + 1):
        _Acur = _Acur @ A
        _newly = (_Acur[_s] > 0) & (_first < 0)
        _first[_newly] = _step
    _reached = _first > 0

    # Sequential colours: later steps = lighter.
    _cmap = plt.get_cmap("viridis")
    _step_colors = [
        _cmap(0.15 + 0.7 * (_step - 1) / max(1, _kmax - 1))
        for _step in range(1, _kmax + 1)
    ]
    _START_COLOR = "#c0223b"
    _UNREACHED = "#e0e0e0"
    _vcols = []
    for _v in range(_n):
        if _v == _s:
            _vcols.append(_START_COLOR)
        elif _first[_v] > 0:
            _vcols.append(_step_colors[_first[_v] - 1])
        else:
            _vcols.append(_UNREACHED)

    _fig, _ax = plt.subplots(figsize=(7.5, 6.0))
    draw_graph(_ax, g, layout_coords, vertex_color=_vcols)
    _handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=_START_COLOR, markersize=10,
                           label=f"start: {walk_i.value}")]
    for _step in range(1, _kmax + 1):
        if np.any(_first == _step):
            _handles.append(plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=_step_colors[_step - 1],
                markersize=10, label=f"reached at step {_step}",
            ))
    if np.any(_first < 0):
        _handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=_UNREACHED,
                                   markersize=10,
                                   label="not reachable"))
    _ax.legend(handles=_handles, loc="upper right", frameon=True,
               fontsize=9)
    _ax.set_title(
        f"From {walk_i.value}: {int(_reached.sum())} of {_n - 1} "
        f"other nodes within {_kmax} step{'s' if _kmax != 1 else ''}"
    )
    plt.tight_layout()

    _per_step = [int((_first == _step).sum()) for _step in range(1, _kmax + 1)]
    _msg = mo.md(
        "Newly reached per step: "
        + ", ".join(f"step {_t + 1}: {_c}" for _t, _c in enumerate(_per_step))
        + ". In matrix terms: count the positive entries in row "
        f"**{walk_i.value}** of `A + A@A + ... ` up to $A^k$ "
        "(diagonal ignored)."
    )
    mo.vstack([_fig, _msg])
    return


@app.cell
def s5_tri_header(mo):
    mo.md(r"""
    ### Triangles: the diagonal of `A @ A @ A`

    Set the target equal to the start: cell $(i, i)$ of $A^3$ counts
    round trips of length 3 — leave home, visit two friends, come back.
    Each such trip traces a **triangle**. On an undirected network every
    triangle through $i$ is walked twice (clockwise and
    counter-clockwise), so triangles through $i$ = $(A^3)_{ii} / 2$, and
    the total is $\mathrm{tr}(A^3)/6$ (each triangle has 3 corners).
    Triangles are the atoms of **clustering** — "are my friends friends
    with each other?" — one of the most-used quantities in network
    science.
    """)
    return


@app.cell
def s5_tri_plot(A, ACCENT, EDGE_COLOR, NUMBER_THRESHOLD, draw_graph, g, is_directed, layout_coords, mo, np, plt):
    _n = g.vcount()
    _names = list(g.vs["name"])
    _A3 = A @ A @ A
    _diag = np.diag(_A3)
    if is_directed:
        _tri_node = _diag.astype(float)
        _total = float(np.trace(_A3) / 3.0)
        _formula = "directed: per node $(A^3)_{ii}$, total $\\mathrm{tr}(A^3)/3$"
    else:
        _tri_node = _diag / 2.0
        _total = float(np.trace(_A3) / 6.0)
        _formula = "undirected: per node $(A^3)_{ii}/2$, total $\\mathrm{tr}(A^3)/6$"

    _fig, (_axL, _axR) = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.3, 1.0]},
    )

    # Left: graph; triangle edges highlighted, counts written on nodes.
    if not is_directed:
        _A2 = A @ A
        _ecols = []
        _ewids = []
        for _e in g.es:
            _u, _v = _e.source, _e.target
            if _A2[_u, _v] > 0:  # the edge closes at least one triangle
                _ecols.append("#e07b00")
                _ewids.append(3.0)
            else:
                _ecols.append(EDGE_COLOR)
                _ewids.append(1.2)
    else:
        _ecols = None
        _ewids = 1.2
    if _tri_node.max() > 0:
        _sizes = list(16 + 26 * (_tri_node / _tri_node.max()))
    else:
        _sizes = None
    if _n <= NUMBER_THRESHOLD:
        _labels = [
            f"{_names[_v]}\n{_tri_node[_v]:g}" for _v in range(_n)
        ]
    else:
        _labels = [""] * _n
    draw_graph(_axL, g, layout_coords, vertex_color=ACCENT,
               vertex_size=_sizes, labels=_labels, label_size=10,
               edge_color=_ecols, edge_width=_ewids)
    _axL.set_title(
        "Triangles through each node"
        + ("  (orange edge = part of a triangle)" if not is_directed else "")
    )

    # Right: sorted bar chart.
    _order = np.argsort(-_tri_node)
    _axR.bar(range(_n), [_tri_node[_o] for _o in _order],
             color=ACCENT, edgecolor="white")
    if _n <= 30:
        _axR.set_xticks(range(_n))
        _axR.set_xticklabels([_names[_o] for _o in _order],
                             rotation=90, fontsize=9)
    else:
        _axR.set_xticks([])
    _axR.set_title(f"Triangles per node  ({_formula})")
    _axR.spines["top"].set_visible(False)
    _axR.spines["right"].set_visible(False)
    plt.tight_layout()

    _top = int(np.argmax(_tri_node))
    _msg = mo.md(
        f"**Total triangles: {_total:g}.** Most triangular node: "
        f"**{_names[_top]}** ({_tri_node[_top]:g}). Divide each node's "
        "count by the number of pairs of its neighbours and you get the "
        "**local clustering coefficient** — same diagonal, one division "
        "away."
    )
    mo.vstack([_fig, _msg])
    return


# -----------------------------------------------------------------------------
# Section 6 — Multiply again and again
# -----------------------------------------------------------------------------


@app.cell
def section6_header(mo):
    mo.md(r"""
    ---
    ## 6. Multiply again and again — a ranking emerges

    One multiplication asked about neighbours; two asked about walks of
    length 2. What if you never stop? Iterate `x = A @ x`, rescaling
    each round so the numbers stay finite. Nodes with well-connected
    neighbours pull ahead — having many walks of every length flowing
    through you is what "being important" turns out to mean. After a
    few rounds the bars stop moving: the network has settled on a
    verdict.
    """)
    return


@app.cell
def s6_widgets(mo):
    iter_k = mo.ui.slider(
        start=0, stop=30, step=1, value=1,
        label="Iterations", show_value=True, full_width=True,
    )
    iter_k
    return (iter_k,)


@app.cell
def s6_plot(A, ACCENT, g, iter_k, mo, np, plt):
    _n = g.vcount()
    _names = list(g.vs["name"])
    _k = int(iter_k.value)

    _x = np.ones(_n) / np.sqrt(_n)
    for _ in range(_k):
        _xn = A @ _x
        _nrm = np.linalg.norm(_xn)
        if _nrm > 0:
            _x = _xn / _nrm
    _x = np.abs(_x)

    try:
        _ref = np.array(g.eigenvector_centrality(directed=g.is_directed()))
        _ref = np.abs(_ref) / max(1e-12, np.linalg.norm(_ref))
    except Exception:
        _ref = np.zeros(_n)

    _order = np.argsort(-_ref) if _ref.sum() > 0 else np.argsort(-_x)
    _xs = np.arange(_n)
    _fig, _ax = plt.subplots(figsize=(11, 3.8))
    _ax.bar(_xs - 0.2, [_x[_o] for _o in _order], width=0.4,
            color=ACCENT, edgecolor="white",
            label=f"x after {_k} iteration{'s' if _k != 1 else ''}")
    _ax.bar(_xs + 0.2, [_ref[_o] for _o in _order], width=0.4,
            color="#bbbbbb", edgecolor="white",
            label="where it converges (igraph eigenvector centrality)")
    if _n <= 30:
        _ax.set_xticks(range(_n))
        _ax.set_xticklabels([_names[_o] for _o in _order],
                            rotation=90, fontsize=9)
    else:
        _ax.set_xticks([])
    _ax.legend(frameon=False, fontsize=9)
    _ax.set_title("Repeated multiplication settles on a ranking")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    plt.tight_layout()

    _gap = float(np.linalg.norm(_x - _ref))
    _msg = mo.md(
        f"Distance to the converged ranking after {_k} "
        f"iteration{'s' if _k != 1 else ''}: **{_gap:.4f}** — slide right "
        "and watch it go to zero.\n\n"
        "This limit is **eigenvector centrality**. Do the same iteration "
        "on a row-normalised $A$ with a little random jumping and you "
        "get **PageRank** — the multiplication that ranked the web. The "
        "last section maps where this same multiplication returns during "
        "the rest of the week."
    )
    mo.vstack([_fig, _msg])
    return


# -----------------------------------------------------------------------------
# Section 7 — Where you will meet this again
# -----------------------------------------------------------------------------


@app.cell
def section7_connections(mo):
    mo.md(r"""
    ---
    ## 7. Where you will meet this again

    - **Day 2 — graph models:** is my network surprising? Compare its
      walk and triangle counts against a random network's.
    - **Day 3a — community detection:** communities are where random
      walks stay trapped.
    - **Day 3b — link prediction:** the simplest score for a missing
      edge $(i, j)$ is cell $(i, j)$ of `A @ A` — shared friends.
    - **Day 3b — node embeddings:** nodes that share many walks end up
      with similar vectors.
    - **Day 4 — graphical models:** message passing is `A @ x`,
      repeated.
    - **Day 5 — social contagion:** outbreaks spread along the
      reachability rings; the leading eigenvalue decides whether they
      take off.

    One question unlocks all of it: *which walks are being counted, and
    how much does each one weigh?*
    """)
    return


@app.cell
def footer(mo):
    mo.md(r"""
    ---
    Javier Garcia-Bernardo · ODISSEI SoDa & Methodology and Statistics,
    Utrecht University · Network Science Summer School 2026 ·
    standalone marimo companion app for Day 1b.
    """)
    return


if __name__ == "__main__":
    app.run()
