import numpy as np
import plotly.graph_objects as go

def moving_avg(x, MA_window=30):
    x = np.asarray(x, dtype=float)
    if len(x) < MA_window:
        return np.array([]), np.array([])
    ma = np.convolve(x, np.ones(MA_window)/MA_window, mode="valid")
    idx = np.arange(MA_window-1, len(x))
    return idx, ma

def plot_series(y, title, y_label, MA_window=30, mode="lines", filename=None):
    y = np.asarray(y, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode=mode, name="per episode"))
    idx, ma = moving_avg(y, MA_window)
    if len(ma):
        fig.add_trace(go.Scatter(x=idx, y=ma, mode="lines", name=f"MA({MA_window})", line=dict(width=5, color='red')))
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified",
    )
    if filename:
        fig.write_image(filename)
    fig.show()


#Old plotting code
#   plt.figure()
#    plt.plot(rewards, label="reward per episode")
#    w = 10
#    cumS = np.cumsum(np.insert(rewards, 0, 0))
#    rw_ma = (cumS[w:] - cumS[:-w]) / float(w)
#    plt.plot(range(w - 1, len(rewards)), rw_ma, label=f"reward MA({w})")
#    plt.title("Reward per episode")
#    plt.xlabel("Episode")
#    plt.grid(True)
#    plt.legend()
#    plt.show()
#
#
#
#    plt.figure()
#    plt.plot(nodes, label="nodes explored")
#    w = 10
#    cumS = np.cumsum(np.insert(nodes, 0, 0))
#    nd_ma = (cumS[w:] - cumS[:-w]) / float(w)
#    plt.plot(range(w - 1, len(nodes)), nd_ma, label=f"nodes MA({w})")
#    plt.title("Nodes explored per episode")
#    plt.xlabel("Episode")
#    plt.grid(True)
#    plt.legend()
#    plt.show()