import numpy as np
import pandas as pd
from nilearn import datasets, plotting, surface
import panel as pn
import plotly
import math
from utils import get_surf_labels, get_surf_shap, update_figure_layout

# Load Desikan-Killiany atlas surface (fetched from abagen)
left_surf = np.load('data/left_desikan_surface.npy')
right_surf = np.load('data/right_desikan_surface.npy')

# Shap DATA
deepshap_pd = pd.read_csv('data/deepSHAP_data.csv')  # deepSHAP
bilanciata_vol_shap_pd = pd.read_csv('data/bilanciata_vol_data.csv')  # bilanciata_vol_data
bilanciata_avg_th_shap_pd = pd.read_csv('data/bilanciata_avg_th_data.csv')  # bilanciata_avg_th_data

# FSAVERAGE surf mesh
fsaverage = datasets.fetch_surf_fsaverage()
mesh_r = surface.load_surf_mesh(fsaverage.pial_right)
mesh_l = surface.load_surf_mesh(fsaverage.pial_left)

# DESIKAN indexes corresponding to xai features
desikan_idx_left = np.array(range(1, 35), dtype='int32')  # 1/34
desikan_idx_right = np.array(range(42, 76), dtype='int32')  # 42/75
desikan_idx_full = np.concatenate((desikan_idx_left, desikan_idx_right))

# patient IDs
ids = deepshap_pd.iloc[:, 0].to_numpy()   # 757 id

# Shap numpy arrays
deepshap = deepshap_pd.iloc[:, 1:].to_numpy()   # (757, 68)
bilanciata_vol_shap = bilanciata_vol_shap_pd.iloc[:, 1:].to_numpy()
bilanciata_avg_th_shap = bilanciata_avg_th_shap_pd.iloc[:, 1:].to_numpy()

# demographic data
demo_data = pd.read_csv('data/demo_data.csv')
# convert dataframe to dict {partecipant_id: [sex, age, site]}
demo_data_dict = demo_data.set_index(demo_data.columns[0]).T.to_dict('list')

# brain region labels
labels = deepshap_pd.columns.values[1:]
left_labels, right_labels = get_surf_labels(atlas_idx=desikan_idx_full, labels=labels,
                                            surf_left=left_surf, surf_right=right_surf)

# ----------------------
#        PANEL
# ----------------------

# start
pn.extension('plotly')

# dropdown menus
dropdown_shap = pn.widgets.Select(name='Select SHAP data', options=['deepSHAP', 'bilanciata_vol_data', 'bilanciata_avg_th_data'])
dropdown_id = pn.widgets.Select(name='Select Patient ID', options=ids.tolist())
dropdown_hemi = pn.widgets.Select(name='Select hemisphere', options=['Left', 'Right'])

# Plotly pane
pane = pn.pane.Plotly()
markdown_pane = pn.pane.Markdown()


# Update the plot according to the options selected in the dropdown menus
def update_graph(event):
    selected_shap = dropdown_shap.value
    selected_hemi = dropdown_hemi.value
    selected_id = dropdown_id.value
    idx = np.where(ids == selected_id)[0][0]

    if selected_shap == 'deepSHAP':
        shap_values = deepshap
    elif selected_shap == 'bilanciata_vol_data':
        shap_values = bilanciata_vol_shap
    else:
        shap_values = bilanciata_avg_th_shap

    left_shap, right_shap = get_surf_shap(atlas_idx=desikan_idx_full, shap_values=shap_values[idx],
                                          surf_left=left_surf, surf_right=right_surf)

    if selected_hemi == 'Right':
        fig_r = plotting.plot_surf_stat_map(mesh_r, right_shap, hemi='right',
                                            view='lateral',
                                            bg_map=fsaverage.sulc_right,
                                            engine='plotly',
                                            cmap='seismic'
                                            )
        update_figure_layout(fig_r, right_labels, right_shap)
        pane.object = fig_r.figure

    else:  # Left
        fig_l = plotting.plot_surf_stat_map(mesh_l, left_shap, hemi='left',
                                            view='lateral',
                                            bg_map=fsaverage.sulc_left,
                                            engine='plotly',
                                            cmap='seismic'
                                            )

        update_figure_layout(fig_l, left_labels, left_shap)
        pane.object = fig_l.figure


# Update patient demographic data
def update_demo_data(event):
    selected_id = dropdown_id.value
    data = demo_data_dict[selected_id]
    markdown_pane.object = f"""
    **Patient data:**
    * Sex: {data[0]}
    * Age: {math.trunc(data[1])}
    * Site: {data[2]:.0f}"""


# Link dropdown menus to the update functions
dropdown_shap.param.watch(update_graph, 'value')
dropdown_hemi.param.watch(update_graph, 'value')
dropdown_id.param.watch(update_demo_data, 'value')
dropdown_id.param.watch(update_graph, 'value')


if __name__ == '__main__':
    layout = pn.Column(pane)

    update_graph('start')
    update_demo_data('start')

    template = pn.template.FastGridTemplate(
        title="Brain XAI Interactive Viewer",
        sidebar=[pn.pane.Markdown("Visualize how much the patient's brain regions affect age prediction"),
                 dropdown_shap,
                 dropdown_id,
                 dropdown_hemi,
                 pn.layout.Divider(),
                 markdown_pane],
        main=[layout],
        row_height=500,
        theme_toggle=False,
        favicon='src/favicon.png'
    )

    # template.servable() # on Colab
    template.show(port=5000)
