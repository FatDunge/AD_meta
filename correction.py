#%%
import os
import numpy as np
import nibabel as nib
from nibabel import nifti1
from scipy.stats import norm


def voxelwise_correction(array, p_array, voxel_count, thres=0.05):
    thresed_p = p_array < thres / voxel_count
    return np.multiply(array, thresed_p)

mask_path = './data/mask/rBN_Atlas_246_1mm.nii'
mask_nii = nib.load(mask_path)
mask = np.asarray(mask_nii.dataobj)
voxel_count = np.size(mask[mask!=0])
#%%
path = './results/meta'
tests = os.listdir(path)
ps = [0.05, 0.01, 0.001]
for test in tests:
    voxel_path = os.path.join(path, test, 'voxel')
    es_path = os.path.join(voxel_path, 'es.nii')
    p_path =  os.path.join(voxel_path, 'p.nii')
    
    es =  nib.load(es_path)
    es_array = np.asarray(es.dataobj)
    p =  nib.load(p_path)
    p_array = np.asarray(p.dataobj)

    for p in ps:
        corrected_array = voxelwise_correction(es_array, p_array, voxel_count, thres=p)
        affine = es.affine
        header = es.header
        corrected_niis = nib.Nifti1Image(corrected_array, affine, header)
        new_f = os.path.join(voxel_path,'es_bon_{}.nii'.format(str(p)[2:]))
        print(new_f)
        print(len(corrected_array[corrected_array!=0]))
        nifti1.save(corrected_niis, new_f)

# %%
from nilearn import plotting
import nibabel as nib
import matplotlib
import os
path = './results/meta'
tests = os.listdir(path)
for test in tests:
    voxel_path = os.path.join(path, test, 'voxel')
    fs = os.listdir(voxel_path)
    for f in fs:
        if 'es_bon' in f and '.nii' in f:
            f = os.path.join(voxel_path, f)
            
            es =  nib.load(f)
            plotting.plot_stat_map(es, title=test,
                                   cmap=matplotlib.cm.get_cmap('RdGy'),)
            html_view = plotting.view_img(es, 
                                          cmap=matplotlib.cm.get_cmap('RdGy'),
                                          )
            html_path = f[:-3]+'html'
            html_view.save_as_html(html_path)

# %%
# gii correction
from nilearn.surface import load_surf_data
from nibabel.gifti.gifti import GiftiDataArray
import numpy as np
path = './results/meta'
tests = os.listdir(path)
ps = [0.05, 0.01, 0.001]
l_r = ['L', 'R']
temp_dir = r'./data/mask/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/{}'
surfs = ['fsaverage.L.inflated.32k_fs_LR.surf.gii', 'fsaverage.R.inflated.32k_fs_LR.surf.gii']
for test in tests:
    voxel_path = os.path.join(path, test, 'surf')
    for lr,surf in zip(l_r, surfs):
        es_path = os.path.join(voxel_path, 'es_{}.gii'.format(lr))
        p_path =  os.path.join(voxel_path, 'p_{}.gii'.format(lr))
        
        es_array = load_surf_data(es_path)[-1]
        p_array = load_surf_data(p_path)[-1]

        

        voxel_count = np.size(p_array) * 2

        for p in ps:
            corrected_array = voxelwise_correction(es_array, p_array, voxel_count, thres=p)
            new_f = os.path.join(voxel_path,'es_bon_{}_{}.gii'.format(lr,str(p)[2:]))
            ct_gii = nib.load(temp_dir.format(surf))
            gdarray = GiftiDataArray.from_array(corrected_array, intent=0)
            ct_gii.add_gifti_data_array(gdarray)
            nib.save(ct_gii, new_f)
        


# %%
