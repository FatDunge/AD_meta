#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
import matplotlib.colors

rgbs = ["#D81E5B", "#3D3B8E", "#D1710B", "#FFD5FF"]
custom_cmap = matplotlib.colors.ListedColormap(rgbs, name='my_colormap')
plt.ioff()

def models_to_dataframe(models):
    index = models.keys()
    cols = ['es', 'll', 'ul']
    df = pd.DataFrame(index=index, columns=cols)
    for k,v in models.items():
        df.loc[k]['es'] = v.total_effect_size
        df.loc[k]['ll'] = v.total_lower_limit
        df.loc[k]['ul'] = v.total_upper_limit
    return df

def draw_top(main_models, sub_models_list, cmap=custom_cmap,
             legend_names=None, topn=20, offset=0.2,
             width_ratio=0.1, height_ratio=0.2,
             linewidth=1, point_size=5, fontsize=12,
             box_aspect=None, value_aspect='auto',
             id_csv_path='./data/mask/cortical_id_new.csv',
             show=True, out_path=None):
    # load id Dataframe
    id_df = pd.read_csv(id_csv_path, index_col=1)

    main_df = models_to_dataframe(main_models)
    sub_dfs  = [models_to_dataframe(models) for models in sub_models_list]

    sorted_main_df = main_df.sort_values('es')
    top_df = sorted_main_df[:topn]
    
    colors = cmap(np.arange(1+len(sub_models_list))).tolist()
    main_color = colors.pop(0)

    fig = plt.figure(figsize=(width_ratio*topn, height_ratio*topn))
    ax = fig.add_axes([0, 0, 1, 1])
    legends = []

    y_labels =[]
    y = topn
    for index, row in top_df.iterrows():
        ll = row['ll']
        ul = row['ul']
        es = row['es']

        plt.scatter(es, y, s=point_size, color=main_color)
        b, = plt.plot((ll, ul), (y, y), linewidth=linewidth, color=main_color)
        if y == topn:
            legends.append(b)
        y = y - 1

        y_labels.append(id_df.loc[int(index)]['name'])

    i = 1
    offsets = []
    for i in range(2,len(sub_models_list)+2):
        if i % 2:
            offsets.append(offset*int(i/2))
        else:
            offsets.append(-offset*int(i/2))
    top_sub_dfs = [top_df.align(df, join='left')[1] for df in sub_dfs]
    for sub_df, color, _offset in zip(top_sub_dfs, colors, offsets):
        y = topn
        for index, row in sub_df.iterrows():
            ll = row['ll']
            ul = row['ul']
            es = row['es']

            plt.scatter(es, y+_offset, s=point_size, color=color)
            b, = plt.plot((ll, ul), (y+_offset, y+_offset), linewidth=linewidth, color=color)
            if y == topn:
                legends.append(b)
            y = y - 1
        i += 1
    
    ax.set_yticks(np.arange(topn+1))
    y_labels.append('')
    y_labels.reverse()
    ax.set_yticklabels(y_labels, fontdict={'fontsize':fontsize})
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', length=0)
    if legend_names is not None and len(legend_names)==len(legends):
        plt.legend(legends, legend_names)
    ax.set_box_aspect(aspect=box_aspect)
    ax.set_aspect(aspect=value_aspect)

    if out_path is not None:
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()

# Draw cesa radar plot
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=False, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, rotation=45,**kwargs)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def radar_plot(dfs, col_name, p_thres=0.05, cmap=custom_cmap,
               legend_loc=(-0.2,-0.2), legend_names=None,
               out_path=None, show=True, save=False):
    n = len(dfs[0])
    theta = radar_factory(n, frame='circle')
    _, ax = plt.subplots(subplot_kw=dict(projection='radar'))

    colors = cmap(np.arange(n))

    labels = [0 for i in range(n)] 
    for df,color in zip(dfs, colors):
        values = df[col_name]
        i = 0
        # mark significant label
        for value in values:
            if value < p_thres:
                labels[i] = 1
            i+=1
        ax.plot(theta, -np.log10(values), color=color)
    spoke_labels = list(dfs[0].index)

    i = 0
    for label in labels:
        if not label:
            spoke_labels[i] = ""
        i += 1

    if legend_names is not None:
        ax.legend(legend_names, loc=legend_loc)

    # Rotate labels
    ax.set_xticks(theta)
    ax.set_xticklabels(spoke_labels)

    angles = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
    angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    angles = np.rad2deg(angles)

    for label, angle in zip(ax.get_xticklabels(), angles):
        x,y = label.get_position()
        lab = ax.text(x,y-0.2, label.get_text(), transform=label.get_transform(),
                    ha=label.get_ha(), va=label.get_va())
        if angle % 180 > 90:
            lab.set_rotation(angle+90)
        else:
            lab.set_rotation(angle-90)
        labels.append(lab)
    ax.set_xticklabels([])
    # inverse y axis for display
    ax.invert_yaxis()
    if save:
        plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_pet_results(result_dict, show=True, save=True,
                     cmap=custom_cmap,
                     alpha=0.7, fontsize=12,
                     out_path='./results/correlation/PET.png'):
    gmv_marker = MarkerStyle(marker='o')
    ct_marker = MarkerStyle(marker='^')

    _, ax = plt.subplots()
    legend = []
    labels = []

    colors = cmap(np.arange(len(result_dict)))

    for k, v in result_dict.items():
        name = []
        if 'AD_NC' in k:
            c = colors[0]
        elif 'AD_MCI' in k:
            c = colors[1]
        elif 'MCI_NC' in k:
            c = colors[2]
        if 'GMV' in k:
            marker = gmv_marker
        elif 'CT' in k:
            marker = ct_marker
        sign_x = []
        sign_y = []
        not_sign_x = []
        not_sign_y = []
        x = 1
        for result in v:
            if result.p < 0.05:
                sign_x.append(x)
                sign_y.append(result.r)
            else:
                not_sign_x.append(x)
                not_sign_y.append(result.r)
            x += 1
            name.append(result.name)
        ls = ax.scatter(sign_x, sign_y, color=c, alpha=alpha, marker=marker)
        #ax.scatter(not_sign_x, not_sign_y, color=c, facecolors='none', alpha=alpha, marker=marker)
        labels.append(k)
        legend.append(ls)
    ax.axhline(0, color='black', lw=1)
    #ax.set_ylim(-1, 1)
    """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(legend, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    """
    ax.legend(legend, labels, prop={'size': 8})
    
    ax.set_xticks(range(1, len(name)+1))
    ax.set_xticklabels(name)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    if save:
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()

#%%
def get_ages_and_gmv(young_centers, old_centers, label=2, roi=1):
    all_ages = []
    all_roi_values = []
    for center in young_centers:
        roi_values, *_ = center.get_csv_values(label=0,
                            prefix='roi_gmv/{}.csv',
                            flatten=True)
        roi_values = roi_values[:, roi-1]
        ages, _ = center.get_ages(label=0)
        all_roi_values.append(roi_values)
        all_ages.append(ages)
    for center in old_centers:
        roi_values, *_ = center.get_csv_values(label=label,
                            prefix='roi_gmv/{}.csv',
                            flatten=True)
        if roi_values is not None:
            roi_values = roi_values[:, roi-1]
            ages, _ = center.get_ages(label=label)
            all_roi_values.append(roi_values)
            all_ages.append(ages*100)
    all_roi_values = np.concatenate(all_roi_values)
    all_ages = np.concatenate(all_ages)
    return all_ages, all_roi_values

def plot_roi_aging(young_centers, old_centers, labels, roi):
    slabels = ['NC', 'MCI', 'AD']
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5), sharex=True, sharey=True)
    all_ages, all_roi_values = get_ages_and_gmv(young_centers, old_centers, label=labels[0], roi=roi)
    ax1.scatter(all_ages, all_roi_values, alpha=0.5)
    ax1.set_title(slabels[labels[0]]+'-'+str(roi))

    all_ages, all_roi_values = get_ages_and_gmv(young_centers, old_centers, label=labels[1], roi=roi)
    ax2.scatter(all_ages, all_roi_values, alpha=0.5)
    ax2.set_title(slabels[labels[1]]+'-'+str(roi))
# %%
def plot_mmse_cor(old_centers, roi=1):
    all_ages = []
    all_roi_values = []
    for center in old_centers:
        roi_values, *_ = center.get_csv_values(
                            prefix='roi_gmv/{}.csv',
                            flatten=True)
        if roi_values is not None:
            roi_values = roi_values[:, roi-1]
            ages, _ = center.get_MMSEs()
            all_roi_values.append(roi_values)
            all_ages.append(ages)
    all_roi_values = np.concatenate(all_roi_values)
    all_ages = np.concatenate(all_ages)

    r = pearsonr(all_ages, all_roi_values)[0]
    p = pearsonr(all_ages, all_roi_values)[1]
    plt.scatter(all_ages, all_roi_values, alpha=0.5)
    plt.title('r:{:.2f}, p:{:.2e}'.format(r, p))
    plt.show()

# %%
