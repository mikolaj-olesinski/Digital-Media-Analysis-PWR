"""Small picture of the TikTok text graph for the class slide: where Cricket Wireless mentions sit against other mobile networks.
Same graph as notebook 09: TF-IDF cosine, 10 nearest neighbours, Louvain with seed 42."""
import numpy as np, networkx as nx, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import eda_utils as eu, extra_models as em, graph_lab as gl

PHONE = ["AT&T", "AT&T Internet", "T-Mobile", "Metro by T-Mobile", "Mint Mobile", "Total Wireless"]


def cricket_numbers(root):
    """Graph, communities, the node sets used in the picture and the top words of each group."""
    raw = eu.load(root); r = raw[raw.platform == "tiktok"].reset_index(drop=True); text = r.title.fillna("") + " " + r.content.fillna("")
    A, _ = gl.knn_graph(em.text_similarity(text.values)); c, _ = gl.louvain(A); cw = np.where(r.brand == "Cricket Wireless")[0]; ph = np.where(r.brand.isin(PHONE))[0]; k = np.bincount(c[cw]).argmax(); grp = np.where(c == k)[0]
    t = text.where(text.str.strip() != "", "emptytext"); v = TfidfVectorizer(min_df=3, sublinear_tf=True, stop_words="english"); X = v.fit_transform(t); w = np.array(v.get_feature_names_out())
    words = {int(q): list(w[np.asarray(X[c == q].mean(0)).ravel().argsort()[::-1][:4]]) for q in np.unique(c)}; phone_groups = [int(q) for q, m in zip(*np.unique(c[ph], return_counts=True)) if m >= 3]
    return dict(A=A, comm=c, brand=r.brand.values, cw=cw, phone=ph, group=grp, cricket_group=int(k), phone_groups=phone_groups, words=words, cw_in_group=int((c[cw] == k).sum()), cw_phone_edges=int(A[np.ix_(cw, ph)].astype(bool).sum()), group_brands=r.brand.iloc[grp].value_counts().to_dict())


def plot_cricket(N, path, lang="en"):
    """The Cricket Wireless group and the groups holding most mobile-network posts, each labelled with its top words."""
    T = (lambda a, b: a if lang == "pl" else b); A, c = N["A"], N["comm"]; gs = [N["cricket_group"]] + N["phone_groups"]; keep = np.where(np.isin(c, gs))[0]
    G = nx.from_numpy_array(A[np.ix_(keep, keep)]); pos = nx.spring_layout(G, seed=7, k=.18, iterations=300); P = np.array([pos[i] for i in range(len(keep))]); cw, ph = np.isin(keep, N["cw"]), np.isin(keep, N["phone"])
    tint = {g: col for g, col in zip(gs, ["#f3d3cf", "#d3e3ee", "#dfe9d9", "#ece3f1"])}
    fig, a = plt.subplots(figsize=(6.1, 4.3)); nx.draw_networkx_edges(G, pos, ax=a, edge_color="#c9c9c4", width=.5, alpha=.6)
    a.scatter(*P[~cw & ~ph].T, s=18, c=[tint[q] for q in c[keep][~cw & ~ph]], edgecolors="#9a9a94", linewidths=.4, zorder=2); a.scatter(*P[ph].T, s=46, c="#2f6f93", zorder=3, label=T("inne sieci komórkowe (AT&T, T-Mobile…)", "other mobile networks (AT&T, T-Mobile…)")); a.scatter(*P[cw].T, s=46, c="#c0392b", zorder=3, label=T("wzmianki „Cricket Wireless”", "\"Cricket Wireless\" mentions"))
    for g in gs:
        m = c[keep] == g; x, y = P[m].mean(0); hot = g == N["cricket_group"]; a.annotate(", ".join(N["words"][g][:3]), (x, y), xytext=(0, -40 if hot else 0), textcoords="offset points", ha="center", va="center", fontsize=10, fontweight="bold" if hot else "normal", color="#c0392b" if hot else "#1f2a33", bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#c0392b" if hot else "#9a9a94", lw=1, alpha=.92), zorder=5)
    a.legend(loc="upper center", bbox_to_anchor=(.5, -.01), ncol=2, frameon=False, fontsize=9.5, handletextpad=.2, columnspacing=.8); a.set_title(T("TikTok: punkt = post, linia = podobny tekst, ramka = słowa grupy", "TikTok: dot = post, line = similar text, box = group's words"), fontsize=11); a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)
