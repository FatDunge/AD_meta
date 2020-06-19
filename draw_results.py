#%%
from nilearn import plotting
import os
import matplotlib
import nibabel as nib

centers = ['EDSD']
for center in centers:
    file_path = './report/{}'.format(center)
    filenames = os.listdir(file_path)

    for filename in filenames:
        if '.nii' in filename and '_es_' in filename:
            nii_path = os.path.join(file_path, filename)
            nii = nib.load(nii_path)
            title = center + filename
            plotting.plot_stat_map(nii, title=filename,
                                    cmap=matplotlib.cm.get_cmap('hot'))

#%%
from nilearn import plotting
import os
import matplotlib
import nibabel as nib
from nilearn.regions import connected_regions
from nibabel import nifti1

centers = ['MCAD', 'EDSD']
for center in centers:
    file_path = './report/{}'.format(center)
    filenames = os.listdir(file_path)

    for filename in filenames:
        if '.nii' in filename and '_p_' in filename and '_with_' in filename:
            nii_path = os.path.join(file_path, filename)
            nii = nib.load(nii_path)
            title = center + '_' + filename
            regions_percentile_img, index = connected_regions(nii,
                                                              min_region_size=300)
            nii_path_post = nii_path[:-4] + '_cluster.nii'
            nifti1.save(regions_percentile_img, nii_path_post)
            plotting.plot_prob_atlas(regions_percentile_img, output_file=nii_path[:-4]+'_cluster.png',
                                     view_type='contours', display_mode='z',
                                     cut_coords=5, title=title)
            plotting.show()


# %%
from nilearn import plotting
import os
import matplotlib
import nibabel as nib
from nilearn.regions import connected_regions
from nibabel import nifti1
file_path = r'E:\kxp\data_baseline\mri'
out = r'E:\kxp\adni_result'

filenames = os.listdir(file_path)
for filename in filenames:
    if '.nii' in filename and 'mwp1' in filename:
            nii_path = os.path.join(file_path, filename)
            nii = nib.load(nii_path)
            plotting.plot_anat(nii, title=filename, cut_coords=(50,50,50),
                                output_file=os.path.join(out, filename)+'.png')
            plotting.show()


# %%
from nilearn import plotting
import os
import matplotlib
import nibabel as nib
from nilearn.regions import connected_regions
from nibabel import nifti1

from nilearn import image

centers = ['ALL']
for center in centers:
    file_path = './report/{}'.format(center)
    filenames = os.listdir(file_path)

    for filename in filenames:
        if '_with_' in filename:
            nii_path = os.path.join(file_path, filename)
            es = image.index_img(nii_path, 0)
            plotting.plot_stat_map(es, title=filename)

# %%
from nilearn import plotting
import nibabel as nib
import matplotlib
import cmap
import numpy as np
path = './results/meta'
tests = os.listdir(path)
for test in tests:
    voxel_path = os.path.join(path, test, 'voxel')
    fs = os.listdir(voxel_path)
    for f in fs:
        if 'es_bon' in f and '.nii' in f:
            fp = os.path.join(voxel_path, f)
            es =  nib.load(fp)

            es_array = np.asarray(es.dataobj)
            corrected_array = np.clip(es_array,a_min=None, a_max=0)
            affine = es.affine
            header = es.header
            corrected_niis = nib.Nifti1Image(corrected_array, affine, header)
            plotting.plot_stat_map(corrected_niis,
                                cmap=cmap.get_cmap(),
                                cut_coords=(30, -10,-10),
                                output_file=fp[:-3]+'png')
            html_view = plotting.view_img(es,
                                cmap=cmap.get_cmap())
            html_path = fp[:-3]+'html'
            html_view.save_as_html(html_path)
# %%
# draw gii
from nilearn import plotting
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.surface import load_surf_data
from nilearn.plotting import view_surf,plot_surf_stat_map
from nilearn import datasets
import cmap
path = './results/meta'
tests = os.listdir(path)
#fsaverage = datasets.fetch_surf_fsaverage()

for test in tests:
    surf_path = os.path.join(path, test, 'surf')
    fs = os.listdir(surf_path)
    for f in fs:
        if 'es_bon' in f and '.gii' in f:
            fp = os.path.join(surf_path, f)
            gii = nib.load(fp)
            
            data = load_surf_data(fp)
            ftl = r'E:\software\matlabPlugin\spm12\toolbox\cat12\templates_surfaces_32k\lh.central.freesurfer.gii'
            ftr = r'E:\software\matlabPlugin\spm12\toolbox\cat12\templates_surfaces_32k\rh.central.freesurfer.gii'
            fbl = r'E:\software\matlabPlugin\spm12\toolbox\cat12\templates_surfaces_32k\lh.sqrtsulc.freesurfer.gii'
            fbr = r'E:\software\matlabPlugin\spm12\toolbox\cat12\templates_surfaces_32k\rh.sqrtsulc.freesurfer.gii'
            
            fig, (ax1, ax2) = plt.subplots(1, 2,
                        subplot_kw={'projection': '3d'})

            plot_surf_stat_map(ftl, data[-1][:32492],
                            cmap=cmap.get_cmap(),
                            view='lateral',
                            bg_map=fbl,
                            bg_on_data=True,
                            darkness=0.2,
                            output_file=fp[:-4]+'_l.png')
            plot_surf_stat_map(ftr, data[-1][32492:],
                            hemi='right',
                            cmap=cmap.get_cmap(),
                            view='lateral',
                            bg_map=fbr,
                            bg_on_data=True,
                            darkness=0.2,
                            output_file=fp[:-4]+'_r.png')
            
            """
            html_view = view_surf(ft,
                        np.clip(data[-1], a_min=None, a_max=0),
                        cmap=cmap.get_cmap(),
                        )
            html_path = fp[:-3]+'html'
            html_view.save_as_html(html_path)
            """
            

# %%
# %%
from nilearn.surface import load_surf_data
data = load_surf_data(r'E:\software\matlabPlugin\spm12\toolbox\cat12\templates_surfaces_32k\lh.inflated.freesurfer.gii')

# %%
data[0].shape

# %%
