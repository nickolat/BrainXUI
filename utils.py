import numpy as np


# Get left and right surface arrays with shap values
def get_surf_shap(atlas_idx, shap_values, surf_left, surf_right):
    mapping = dict(zip(atlas_idx, shap_values))

    # LEFT
    left_shap = [mapping[i] if i!=0 else 0 for i in surf_left]
    # RIGHT
    right_shap = [mapping[i] if i!=0 else 0 for i in surf_right]

    return np.array(left_shap), np.array(right_shap)


# Get left and right surface arrays with region labels
def get_surf_labels(atlas_idx, labels, surf_left, surf_right):
    mapping = dict(zip(atlas_idx, labels))

    # LEFT
    left_labels = [mapping[i] if i!=0 else 'not mapped' for i in surf_left]
    # RIGHT
    right_labels = [mapping[i] if i!=0 else 'not mapped' for i in surf_right]

    return left_labels, right_labels


# Update figure layout: customize the plot and add hover label text (region name and shap value)
def update_figure_layout(fig, labels, shap):
    # to show layout
    fig.figure.add_shape(
        type="line", line_color="white", visible=False, opacity=1, line_dash="dot",
        x0=0, x1=1, xref="paper", y0=950, y1=950, yref="y"
    )

    # First trace
    # to show hover label with region and shap value
    trace = fig.figure.data[0]
    trace['customdata'] = np.stack((labels, shap), axis=-1)
    trace.hovertemplate = 'region: %{customdata[0]}<br>shap: %{customdata[1]:.2e}<extra></extra>'

    max_shap = np.max(shap)
    min_shap = np.min(shap)
    max_abs_shap = max(abs(max_shap), abs(min_shap))

    # update layout settings
    trace1 = fig.figure.data[1]
    trace1.update(colorbar=dict(
        title=dict(text='shap values<br>&nbsp;', font=dict(size=15)),
        tickfont=dict(size=15),
        tickformat=".1e",
        tickmode="array",
        tickvals=[-max_abs_shap, 0, max_abs_shap]
    )
    )

    fig.figure.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            bgcolor='white',
            dragmode="turntable"
        ),
        xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False),
        yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False),
        hoverlabel=dict(
            bgcolor='white'
        ),
        modebar_remove=["autoScale2d", "zoomIn2d", "zoomOut2d", "orbitRotation"]
    )
