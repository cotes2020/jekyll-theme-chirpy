import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px


def kde(p, q, points):
    """Plot the kernel density of a pair of vectors.

    Args:
        p: The momentum vector.
        q: The position vector.
        points: Number of points along each edge of the grid.

    Returns:
        The momentum edges, position edges, and grid of kde values.
    """
    xmin, xmax = p.min(), p.max()
    ymin, ymax = q.min(), q.max()
    x = np.linspace(xmin, xmax, points)
    y = np.linspace(ymin, ymax, points)
    X, Y = np.meshgrid(x, y)
    kernel = stats.gaussian_kde(np.vstack([p, q]))
    Z = np.reshape(kernel(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)
    return x, y, Z


def plot_hamiltonian(p, q, H, contours=20, points=100):
    """Plot the phase space of the Hamiltonian.

    Args:
        p: The momentum vector.
        q: The position vector.
        H: The Hamiltonian vector.
        contours: Number of contours rings in the KDE plot.
        points: Number of points along each edge of the grid.

    Returns:
        plotly Figure of the Hamiltonian phase space.
    """
    fig = go.Figure()
    P = np.exp(-H)
    x, y, Z = kde(p, q, points=points)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            # Scale the kde values to the size of the Hamiltonian.
            z=(Z / Z.max()) * max(P),
            name=f"KDE of p & q",
            opacity=0.05,
            showscale=False,
            showlegend=True,
            contours=dict(
                z=dict(
                    show=True,
                    start=min(P),
                    end=max(P),
                    size=(max(P) - min(P)) / contours,
                    project_z=True,
                    usecolormap=True,
                )
            ),
        ),
    )
    fig.add_trace(
        go.Scatter3d(
            x=p,
            y=q,
            z=P,
            opacity=0.1,
            name=f"Samples for p & q",
            mode="markers",
            marker=dict(
                size=2,
                color=P,
            ),
        ),
    )
    fig.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        legend=dict(yanchor="bottom", y=-10.0, xanchor="left", x=0.01),
        scene=dict(
            xaxis=dict(title="p"),
            yaxis=dict(title="q"),
            zaxis=dict(title="H"),
        ),
    )
    return fig


# NOTE: p, q, & H represent the momentum, position, and Hamiltonian vectors, respectively.
p = np.array([])
q = np.array([])
H = np.array([])

# Plot Hamiltonian value
fig = px.line(
    pd.DataFrame({"Iteration": range(len(H)), "Hamiltonian Value": H}),
    x="Iteration",
    y="Hamiltonian Value",
)
# Write the figure to html, then copy+paste into the markdown file.
fig.write_html(
    "path_1.html",
    include_plotlyjs="cdn",
)
fig.show()

# Phase Space Plots
fig = plot_hamiltonian(p[:, 0], q[:, 0], H)
fig.write_html(
    "path_2.html",
    include_plotlyjs="cdn",
)
fig.show()
fig = plot_hamiltonian(p[:, 1], q[:, 1], H)
fig.write_html(
    "path_3.html",
    include_plotlyjs="cdn",
)
fig.show()
