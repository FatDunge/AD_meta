#%%
#Prepare features csv
import datasets
import numpy as np
import os
import csv
from meta_analysis import utils
from meta_analysis.main import csv_meta_analysis

filenames = 'origin.csv'
centers = datasets.load_centers_all(filenames=filenames)
out_path = './data/meta_csv'
label_pairs = [(2,0), (2,1), (1,0)]
csv_prefixs = ['roi_gmv_removed/', 'roi_ct_removed/']

def sort_models(models, filenames, orderby='es', descend=True):
    if orderby == 'es':
        list1 = [model.total_effect_size for model in models]
    list1, models, filenames = (list(t) for t in zip(*sorted(zip(list1, models, filenames), reverse=descend)))
    return list1, models, filenames

def bon_cor(models, filenames, thres=0.05):
    after = {}
    n = len(models)
    for model, filename in zip(models, filenames):
        if model.p * n <= thres:
            after[filename] = model
    return after
#%%
# create csv for meta analysis
for label_pair in label_pairs:
    label_eg, label_cg = label_pair
    label_dir = os.path.join(out_path, '{}_{}'.format(label_eg, label_cg))
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
    for csv_prefix in csv_prefixs:
        out_dir = os.path.join(label_dir, csv_prefix)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        mean_egs = []
        std_egs = []
        count_egs = []
        mean_cgs = []
        std_cgs = []
        count_cgs = []
        
        center_names = []
        for center in centers:
            features_eg, _, ids = center.get_csv_values(label=label_eg,
                                                        prefix=csv_prefix+'{}.csv',
                                                        flatten=True)
            features_cg, *_ = center.get_csv_values(label=label_cg,
                                                    prefix=csv_prefix+'{}.csv',
                                                    flatten=True)
            if features_eg is not None and features_cg is not None:
                mean_eg, std_eg, n_eg = utils.cal_mean_std_n(features_eg)
                mean_cg, std_cg, n_cg = utils.cal_mean_std_n(features_cg)
                mean_egs.append(mean_eg)
                std_egs.append(std_eg)
                count_egs.append(n_eg)
                mean_cgs.append(mean_cg)
                std_cgs.append(std_cg)
                count_cgs.append(n_cg)
                center_names.append(center.name)

        mean_egs = np.stack(mean_egs)
        std_egs = np.stack(std_egs)
        count_egs = np.stack(count_egs)
        mean_cgs = np.stack(mean_cgs)
        std_cgs = np.stack(std_cgs)
        count_cgs = np.stack(count_cgs)

        mean_egs_T = mean_egs.T
        std_egs_T = std_egs.T
        mean_cgs_T = mean_cgs.T
        std_cgs_T = std_cgs.T

        i = 0
        for ems, ess, cms, css,_id in zip(mean_egs_T, std_egs_T,
                                        mean_cgs_T, std_cgs_T,ids):
            i += 1
            csv_path = os.path.join(out_dir, '{}.csv'.format(_id))
            with open(csv_path, 'w', newline='') as file:
                fieldnames = ['center_name', 'm1','s1','n1', 'm2','s2','n2']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

                for em, es, en, cm, cs, cn, center_name in zip(
                    ems, ess, count_egs, cms, css, count_cgs, center_names
                ):
                    writer.writerow({'center_name': center_name,
                                    'm1': em,
                                    's1': es,
                                    'n1': en,
                                    'm2': cm,
                                    's2': cs,
                                    'n2': cn,})

#%%
thress = [0.05, 0.01, 0.001]
for csv_prefix in csv_prefixs:
    for pair in label_pairs:
        csv_dir = './data/meta_csv/{}_{}/{}'.format(pair[0], pair[1], csv_prefix)
        out_dir = './results/meta/{}_{}/{}'.format(pair[0], pair[1], csv_prefix)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        csvs = os.listdir(csv_dir)
        models = []
        filenames = []
        for f in csvs:
            csv_path = os.path.join(csv_dir, f)
            model = csv_meta_analysis(csv_path, model_type='random')
            models.append(model)
            filenames.append(f[:-4])

        for thres in thress:
            cor_model = bon_cor(models, filenames)
            print('{}_{}:{}_{}:{}'.format(csv_prefix, thres,pair[0], pair[1], len(bon_cor(models, filenames, thres=thres))))
        
        list1, models, filenames = sort_models(models, filenames, descend=False)
        top = len(models)
        i = 0
        """
        for model, filename in zip(models[:top], filenames[:top]):
            out_path = os.path.join(out_dir, '{}_{}.png'.format(i, filename))
            model.plot_forest(title=filename, save_path=out_path, show=False)
            i += 1
        """
        
        csv_path = os.path.join(out_dir, 'TOP{}.csv'.format(top))
        with open(csv_path, 'w', newline='') as file:
            fieldnames = ['name', 'es', 'se', 'll', 'ul', 'z','p']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for model, filename in zip(models[:top], filenames[:top]):
                writer.writerow({'name': filename,
                 'es': round(model.total_effect_size,2),
                  'se':round(model.total_standard_error,2), 
                  'll':round(model.total_lower_limit,2), 
                  'ul':round(model.total_upper_limit,2),
                  'z':round(model.z, 2),
                  'p':'{:.2e}'.format(model.p)})
        
# %%
import nibabel as nib
import pandas as pd
import os
from meta_analysis import utils
from nilearn import plotting
from meta_analysis.main import csv_meta_analysis

pairs = [(2,0), (2,1),(1,0)]
ps = [0.05, 0.01, 0.001]
csv_path = './data/mask/BNA_subregions.csv'
df = pd.read_csv(csv_path, index_col=1)
nii_path  = './data/mask/rBN_Atlas_246_1mm.nii'
nii = nib.load(nii_path)
fea = 'roi_gmv_removed'
for pair in pairs:
    for p in ps:
        nii_array = np.asarray(nii.dataobj, dtype=np.float32)
        csv_dir = './data/meta_csv/{}_{}/{}'.format(pair[0], pair[1], fea)
        out_dir = './results/meta/{}_{}/{}'.format(pair[0], pair[1], fea)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        csvs = os.listdir(csv_dir)
        models = []
        filenames = []
        for f in csvs:
            csv_path = os.path.join(csv_dir, f)
            model = csv_meta_analysis(csv_path, model_type='random')
            models.append(model)
            filenames.append(f[:-4])

        cor_model = bon_cor(models, filenames, thres=p)
        ll = [i for i in range(1, 247)]
        for k, v in cor_model.items():
            _id = df.loc[k]['ID']
            nii_array[nii_array==_id] = v.total_effect_size
            ll.remove(_id)
        for i in ll:
            nii_array[nii_array==i] = 0
        path = os.path.join(out_dir, 'cor_{}.nii'.format(str(p).replace('.', '_')))
        r = utils.gen_nii(nii_array, nii, path)
        plotting.plot_stat_map(r,
                                cmap=cmap.get_cmap(),
                                cut_coords=(30, -10,-10),
                                output_file=path[:-3]+'png')

        html_view = plotting.view_img(r)
        html_path = path[:-3]+'html'
        html_view.save_as_html(html_path)
# %%
import nilearn as nil
from nilearn import surface
from nilearn import plotting
import nibabel as nib
import pandas as pd
import os
from meta_analysis import utils
from nilearn import plotting
from meta_analysis.main import csv_meta_analysis
import numpy as np
import cmap

pairs = [(2,0), (2,1),(1,0)]
annots = ['lh.BN_Atlas.annot', 'rh.BN_Atlas.annot']
surfs = ['lh.pial', 'rh.pial']
bgs = ['lh.sulc', 'rh.sulc']
csv_path = './data/mask/cortical_id.csv'
df = pd.read_csv(csv_path, index_col=0)
label_dir = r'./data/mask/BN_Atlas_freesurfer/fsaverage/label/{}'
surf_dir = r'./data/mask/BN_Atlas_freesurfer/fsaverage/surf/{}'

def bon_cor(models, filenames,alpha=0.001):
    after = {}
    n = len(models)
    for model, filename in zip(models, filenames):
        if model.p * n <= alpha:
            after[filename] = model
    return after
for pair in pairs:
    for annot, surf, bg in zip(annots, surfs, bgs):
        a = surface.load_surf_data(label_dir.format(annot))
        a = a.astype(np.float32)
        b = surf_dir.format(surf)
        g = surf_dir.format(bg)

        csv_dir = './data/meta_csv/{}_{}/roi_ct_removed'.format(pair[0], pair[1])
        out_dir = './results/meta/{}_{}/roi_ct_removed'.format(pair[0], pair[1])
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        csvs = os.listdir(csv_dir)
        models = []
        filenames = []
        for f in csvs:
            csv_path = os.path.join(csv_dir, f)
            model = csv_meta_analysis(csv_path, model_type='random')
            models.append(model)
            filenames.append(f[:-4])

        cor_model = bon_cor(models, filenames)
        ll = np.unique(a).tolist()

        for k, v in cor_model.items():
            _id = np.float32(df.loc[k]['ID'])
            a[a==_id] = v.total_effect_size
            if _id in ll:
                ll.remove(_id)
        for i in ll:
            a[a==i] = 0
        p = os.path.join(out_dir, '{}'.format(annot))

        plotting.plot_surf_stat_map(b, a,
                            cmap=cmap.get_cmap(),
                            view='lateral',
                            bg_map=g,
                            bg_on_data=True,
                            darkness=0.2,
                            output_file=p[:-4]+'_l.png')
        plotting.plot_surf_stat_map(b, a,
                        hemi='right',
                        cmap=cmap.get_cmap(),
                        view='lateral', 
                        bg_map=g,
                        bg_on_data=True,
                        darkness=0.2,
                        output_file=p[:-4]+'_r.png')

        html_view = plotting.view_surf(b, a)
        html_path = p+'.html'
        html_view.save_as_html(html_path)


# %%

from nilearn import surface
a = surface.load_surf_data(r'./data/mask/BN_Atlas_freesurfer/fsaverage/label/{}'.format('lh.BN_Atlas.annot'))
a.shape

# %%
