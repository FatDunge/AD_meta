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
            #plotting.plot_stat_map(nii, title=filename, vmax=0,vmin=-0.5,
            #                        cmap=matplotlib.cm.get_cmap('hot'))
            #plotting.plot_glass_brain(nii, title=title,
                                      #threshold=0.8, colorbar=True,
                                      #output_file=os.path.join(file_path, filename)[:-3],
                                      #annotate=True, plot_abs=False)
            html_view = plotting.view_img_on_surf(nii, threshold=0.2)
            html_path = nii_path[:-3]+'html'
            html_view.save_as_html(html_path)

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
