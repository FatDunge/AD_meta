"""
TODO:add license
"""
import PythonMeta as PMA
from scipy import stats
import numpy as np
import mask

def meta(studies, settings,
         plot_forest=False, plot_funnel=False):
    """Meta analysis for study

    Args:
        studies: list of string, details in $PythonMeta$.
        settings: dict, specify datatype, models, algorithm, effect.
        plot_forest: bool, whether to show forest plot.
        plot_funnel: bool, whether to show funnel plot.

    Returns:
        A tuple of (mean, std, n)
    """
    data = PMA.Data()
    _meta = PMA.Meta()
    fig = PMA.Fig()

    data.datatype = settings["datatype"]
    _meta.datatype = data.datatype
    studies = data.getdata(studies)

    _meta.models = settings["models"]
    _meta.algorithm = settings["algorithm"]
    _meta.effect = settings["effect"]
    results = _meta.meta(studies)
    if plot_forest:
        fig.forest(results).show()
    if plot_funnel:
        fig.funnel(results).show()
    return results

def show_forest(results):
    fig = PMA.Fig()
    fig.forest(results).show()

def get_center_roi_msn_by_label(center, roi, label, tissue_type='GMV', use_tiv=False):
    """get mean, std, n from feature with specified label.

    Args:
        center: Center instance, details in #center.py#
        roi: string, roi's niimage filename.
        label: int, which kind of person to get
        tissue_type: string, indicate tissue_type to calculate mean, std, n
        use_tiv: bool, whether to divide tiv while calculating roi volumn

    Returns:
        A tuple of (mean, std, n)
    """
    persons = center.get_by_label(label)
    data = [person.get_region_volumn(int(roi), tissue_type, use_tiv=use_tiv) for person in persons]
    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    count = data.shape[0]
    return mean, std, count

def gen_roi_study(centers_list, roi, label_eg, label_cg, tissue_type='GMV', use_tiv=False):
    """get study in the format PythonMeta requires.

    Args:
        centers_list: list of Center, details in #center.py#
        roi: string, roi's ID in csv file.
        label_eg: int, experimental group label
        label_cg: int, control group label
        tissue_type: string, indicate tissue_type to calculate mean, std, n
        use_tiv: bool, whether to divide tiv while calculating roi volumn

    Returns:
        list PythonMeta requires, [study1, study2, ...]
    """
    study = []
    i = 1
    for center in centers_list:
        mean_eg, std_eg, count_eg = get_center_roi_msn_by_label(center, roi, label_eg,
                                                            tissue_type=tissue_type,
                                                            use_tiv=use_tiv)
        mean_cg, std_cg, count_cg = get_center_roi_msn_by_label(center, roi, label_cg,
                                                            tissue_type=tissue_type,
                                                            use_tiv=use_tiv)
        if count_eg == 0 or count_cg == 0:
            pass
        else:
            study.append('{}, {}, {}, {}, {}, {}, {}'.format(center.name,
                                                             mean_eg, std_eg, count_eg,
                                                             mean_cg, std_cg, count_cg))
        i = i + 1
    return study

def gen_roi_studies(centers_list, rois, label_eg, label_cg, tissue_type='GMV', use_tiv=False):
    """get studies for each roi.

    Args:
        centers_list: list of Center, details in #center.py#
        rois: list of string, roi's niimage filename.
        label_eg: int, experimental group label
        label_cg: int, control group label
        tissue_type: string, indicate tissue_type to calculate mean, std, n
        use_tiv: bool, whether to divide tiv while calculating roi volumn

    Returns:
        dict of studies, {roi: study, ...}
    """
    studies = {}
    for roi in rois:
        study = gen_roi_study(centers_list, roi, label_eg, label_cg,
                          tissue_type=tissue_type, use_tiv=use_tiv)
        studies[roi] = study
    return studies

def get_center_voxel_msn_by_label(center, index, label):
    persons = center.get_by_label(label)
    data = []
    for person in persons:
        data.append(np.asarray(person.nii.dataobj).flatten()[index])
    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    count = data.shape[0]
    return mean, std, count

def gen_voxel_single_center_study(i, j, mean_eg, std_eg, count_eg, mean_cg, std_cg, count_cg):
    return 'center{}, {}, {}, {}, {}, {}, {}'.format(i,
                                                     mean_eg[j], std_eg[j], count_eg,
                                                     mean_cg[j], std_cg[j], count_cg)

def gen_voxel_studies(centers_list, mask, label_eg, label_cg):
    means_eg = []
    stds_eg = []
    counts_eg = []
    means_cg = []
    stds_cg = []
    counts_cg = []

    mask_data = mask.get_mask_data().flatten()
    index = mask_data>0
    mask_index = np.where(mask_data>0)[0]
    for center in centers_list:
        mean_eg, std_eg, count_eg = get_center_voxel_msn_by_label(center, index, label_eg)
        mean_cg, std_cg, count_cg = get_center_voxel_msn_by_label(center, index, label_cg)
        
        if count_eg and count_cg:
            means_eg.append(mean_eg)
            stds_eg.append(std_eg)
            counts_eg.append(count_eg)
            means_cg.append(mean_cg)
            stds_cg.append(std_cg)
            counts_cg.append(count_cg)

    studies = {}
    #for each feature
    for j in range(mask_index.shape[0]):
        study = []
        # for each center
        for i in range(len(means_eg)):
            study.append('center{}, {}, {}, {}, {}, {}, {}'.format(i,
                                                           means_eg[i][j], stds_eg[i][j], counts_eg[i],
                                                           means_cg[i][j], stds_cg[i][j], counts_cg[i]))
        studies[mask_index[j]] = study
    return studies

if __name__ == '__main__':
    pass
