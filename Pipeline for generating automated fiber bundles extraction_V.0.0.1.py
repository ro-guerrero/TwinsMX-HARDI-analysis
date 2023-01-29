#!/usr/bin/env python
# coding: utf-8

# <center>
#     <h1>Pipeline for generating automated fiber bundles extraction</h1>
# </center>
# <center><h3> Juan Rodrigo Guerrero Morales, María Guadalupe García Gomar</h3></center>
# <center><h4>22/10/22</h4></center>

# ## 1. Data aquisition
# This script is only going to #run if the following files are present in the current working directory:
# + One reverse phase encoded volume
# + One phase encoded volume
# + Other two phase encoded volumes with different b
# + One anatomical T1-wheighted image

# ## 2. Denoising using Local-PCA
# ### Concatenation of DWI images
# We need to use MRtrix's command `mrcat` to concatenate the three diffusion wheighted volumes. For this the following cell is gonna search for all the nifti-files in the current working directory convert them to mif using `mrconvert`, concatenate them into a single nii volume and from this volume extract it's bvecs and bvals.

# In[2]:


from os import listdir as ls
from os.path import splitext, join
from subprocess import run

cwd_files = ls()
# Conventing the nii_files to mif
run('mkdir mif_convert', shell=True)
for nii_file in cwd_files:
    if '.nii.gz' in nii_file and 'DTI' in nii_file:
        base_name = nii_file.split('.')[0]
        bvecs     = base_name + '.bvec'
        bvals     = base_name + '.bval'
        mif_out   = join('mif_convert', base_name + '.mif')
        run(['mrconvert', nii_file, '-fslgrad', 
             bvecs, bvals, mif_out])
# Concatenation
run('mkdir raw_concat', shell=True)

cat_image = join('raw_concat', 'cat_AP_DWI.mif')
cat_basename = cat_image.split('.')[0]
run('mrcat mif_convert/*.mif raw_concat/cat_AP_DWI.mif', shell=True)
# nii conversion and bvals/bvecs extraction
# mrconvert DICOM/ dwi.nii.gz -export_grad_fsl bvecs bvals
ex_grad = '-export_grad_fsl'
cat_nii =  cat_basename + '.nii.gz'
bvec = cat_basename + '.bvec'
bval = cat_basename + '.bval'
run(['mrconvert', cat_image, cat_nii, ex_grad, bvec, bval])
run('mv -t raw_concat' + cat_nii + ' ' + cat_image + ' ' + bvec + bval, shell=True)


# ## Applying local-PCA 
# For the denoising we're gonna use Dipy's `local_pca` function based in Manjon's Local_pca denoising[Manjon, 2013]

# In[3]:


from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import numpy as np
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate

# Load the data
data, affine = load_nifti(cat_nii)
bval_read, bvec_read = read_bvals_bvecs(bval, bvec)
# Generate a gradient table
gtab = gradient_table(bval_read, bvec_read)
# #run the denoising
np.seterr(invalid='ignore')
sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=3)
denoised_vol = localpca(data, sigma, tau_factor=2.3, patch_radius=2)


# In[4]:


# Save the denoised volume
run('mkdir pca_gibbs', shell=True)
pca_nii = join('pca_gibbs', 'local_pca.nii.gz')
save_nifti(pca_nii, denoised_vol, affine)


# ## 3. Suppresion of Gibbs oscillators
# The removal of Gibbs Rings is performed using `gibbs_removal` from Dipy.

# In[5]:


from dipy.denoise.gibbs import gibbs_removal
gibbs_corrected = gibbs_removal(denoised_vol)
# Saving the image
gibbs_nifti = join('pca_gibbs','gibbs.nii.gz')
save_nifti(gibbs_nifti, gibbs_corrected, affine)


# ## 4. Motion and distortion correction

# The nii image needs to be converted to mif using `mrconvert`.

# In[6]:


run('mkdir eddy', shell=True)
gibbs_file= gibbs_nifti.split('/')[-1]
eddy_file = join('eddy', 'eddy.mif')
gibbs_mif = gibbs_nifti.split('.')[0] + '.mif'
run(['mrconvert', gibbs_nifti, '-fslgrad', 
     bvec, bval, gibbs_mif])


# The b0 images of the denoised and unringed volume need to be extracted and averaged.

# In[7]:


b0s = join('pca_gibbs','b0s_gibbs_mean')
b0s_mif = b0s + '.mif'
run('dwiextract ' + gibbs_mif + ' - -bzero | mrmath - mean ' 
    + b0s_mif + ' -axis 3', shell=True)


# In[8]:


run('mkdir maps', shell=True)
for nii_file in cwd_files:
    if 'DWI4fMaps' in nii_file and '.nii.gz' in nii_file:
        base_name = nii_file.split('.')[0]
        bvecs_map = base_name + '.bvec'
        bvals_map = base_name + '.bval'
        mif_4maps_out = join('maps',base_name + '.mif')
        print(mif_4maps_out)
        run(['mrconvert', nii_file, '-fslgrad', 
             bvecs_map, bvals_map, mif_4maps_out])


# In[9]:


b0s_maps = 'b0s_4maps'
b0s_4m_mif = join('maps', b0s_maps + '.mif')
run('dwiextract ' + mif_4maps_out + ' - -bzero | mrmath - mean ' 
    + b0s_4m_mif + ' -axis 3', shell=True)


# In[10]:


# Concatenaiting the b0 images
b0s_pair = join('eddy','b0_pair.mif')
run(['mrcat', b0s_mif, b0s_4m_mif, '-axis', '3', b0s_pair])


# In[ ]:


# Excecute topup
topup_base_n = 'preproc'
topup_mif = join('eddy', topup_base_n + '.mif')
run('dwifslpreproc' +' '+ gibbs_mif +' '+ topup_mif + ' -pe_dir' + ' AP' + ' -rpe_pair' + ' -se_epi ' +' '+ b0s_pair +' '+ ' -eddy_options ' + ' " --flm=linear"', shell=True)


# ## 5. T1 linear registration

# In[ ]:


run('mkdir t1_reg', shell=True)
# Extraction and mean b0s_preproc
b0s_preproc = join('eddy','b0s_preproc.nii.gz')
run('dwiextract ' + topup_mif + ' - -bzero | mrmath - mean ' 
    + b0s_preproc + ' -axis 3', shell=True)


# In[ ]:


for nii_file in cwd_files:
    if 'nii.gz' in nii_file and 'T1' in nii_file:
        T1_anatomical = nii_file
        base_name = nii_file.split('.')[0]

moving_image = T1_anatomical
ants_out     = join('t1_reg', '_ANTS_linear')
fixed_image  = b0s_preproc
run(['antsRegistrationSyN.sh',
     '-d', '3',
     '-f', fixed_image,
     '-m', moving_image,
     '-t', 'a',
     '-o', ants_out])


# In[ ]:


warped_T1 = ants_out + 'Warped.nii.gz'
t12std = warped_T1.split('.')[0] + '_2std.nii.gz'
fivettmask = 't1_reg/5tt_mask.nii.gz' 
# 1. reorientar a std
run(['fslreorient2std', warped_T1, t12std])
fsl_dir = '/home/inb/lconcha/fmrilab_software/fsl_5.0.6/data/standard'
mni_name= 'MNI152_T1_3mm.nii.gz'
mni_3mm = join(fsl_dir, mni_name)
t1_2_mni = 't1_reg/T1_MNI_'
run(['antsRegistrationSyN.sh',
     '-d', '3',
     '-f', mni_3mm,
     '-m', t12std,
     '-t', 'a',
     '-o', t1_2_mni])
#run(['5ttgen', 'fsl', warped_T1, fivettmask, '-nocleanup', '-debug'])


# In[ ]:


mni_mask = join(fsl_dir, 'MNI152_T1_3mm_brain_mask_dil.nii.gz')
t1_MNI_warped = t1_2_mni + 'Warped.nii.gz'

from dipy.segment.mask import applymask
t1_MNI_data, t1_MNI_affine = load_nifti(t1_MNI_warped)
mask_data, mask_aff = load_nifti(mni_mask)
t1_masked = applymask(t1_MNI_data, mask_data)
t1_masked_dir = 't1_reg/t1_mni_masked.nii.gz'
save_nifti(t1_masked_dir, t1_masked, t1_MNI_affine)


# In[ ]:


mat = t1_2_mni + '0GenericAffine.mat'

# np.loadtxt(mat, dtype='int32', delimiter=',', converters={_:lambda s: int(s, 16) for _ in range(3)})
# from scipy.io import loadmat
# trans_mat = loadmat(mat)


# In[ ]:


# affine = trans_mat['AffineTransform_double_3_3']
# fixed  = trans_mat['fixed']


# In[ ]:


# affine_mtx = affine.reshape(3,4)
# l_array = np.array([0.0000,0.0000,0.0000,1.0000])


# In[ ]:


# np_affine_2 = np.append(affine_mtx, l_array)


# In[ ]:


# np.savetxt('np_affine_2.mat', np_affine_2)


# ## Flirt registration to MNI

# In[ ]:


T1_MNI_flirt = 't1_reg/T1_MNI_flirt.nii.gz'
run(['first_flirt', t12std, T1_MNI_flirt ])


# In[ ]:


flirt_mat = T1_MNI_flirt.split('.')[0] + '.mat'
run('./t1_reg/hex.sh' + ' ' + flirt_mat + ' > ' + 'flirt_mtrx.mat', shell=True)


# ## First segmentation

# In[ ]:


thal_dir = '/home/inb/lconcha/fmrilab_software/fsl_5.0.6/data/first/models_336_bin/'
L_accu = join(thal_dir, 'L_Accu_bin.bmv')
out_first_acc = 't1_reg/first.nii.gz'
out_first_all = 't1_reg/first_all'
run(['#run_first', '-i', t12std,'-v', '-t', 'flirt_mtrx.mat', '-n', '40','-o', out_first_acc, '-m', L_accu])
run(['#run_first_all', '-d', '-b', '-a', 'flirt_mtrx.mat', '-i', t12std, out_first_all])


# In[ ]:


from os import listdir as ls
subgm_05 = join(thal_dir, '05mm')
caudate  = '_Caud_bin.bmv'
L_cau    = 'L' + caudate
R_cau    = 'R' + caudate
sub05_ls = ls(subgm_05)
sub05_ls.append(L_cau)
sub05_ls.append(R_cau)

nii_models = []
for gm in sub05_ls:
    if 'Cereb' not in gm:
        gm_name = gm.split('.')[0]
        gm_dir = join('t1_reg', gm_name)
        gm_out = join(gm_dir, gm_name)
        gm_nii = gm_out + '.nii.gz'
        nii_models.append(gm_nii)
        if '_Caud_' not in gm:
            model  = join(subgm_05, gm)
        else: 
            model = join(thal_dir, gm)
        run(['mkdir', gm_dir])
        run(['run_first', '-i', t12std,'-v', '-t', 'flirt_mtrx.mat', '-n', '40','-o', gm_out, '-m', model])        


# In[ ]:


models_data = []
models_affine = []
for nii in nii_models:
    data, affine = load_nifti(nii)
    condition = data > 0
    non_zero  = condition*1.0
    threshold = np.mean(non_zero)
    where = np.where(data > threshold, 1, 0)
    models_data.append(where)
    models_data.append(non_zero)
    models_affine.append(affine)
    
    


# In[ ]:


nii_models


# In[ ]:


# puta = models_data[0]


# In[ ]:


# save_nifti('puta_boolean_mtx.nii.gz', puta, models_affine[0])


# In[ ]:


# scgm_data_bin = np.sum(models_data,0)


# In[ ]:


# save_nifti('t1_reg/bin_scgm_mask.nii.gz', scgm_data_bin, models_affine[0])


# ## 7. Betting and mask application

# In[ ]:


bet_vol = 't1_reg/bet_f06'
fval     = '0.6'
run(['bet2', b0s_preproc, bet_vol, '-f', fval, '-m'])
bet_mask = bet_vol + '.nii.gz'
bet_bin  = bet_vol + '_mask.nii.gz'


# I need to apply the mask to the T1 image.

# In[ ]:


bet_bin


# In[ ]:


from dipy.segment.mask import applymask

topup_nii = 't1_reg/' + topup_base_n + '.nii.gz'

# T1 dwi registered betting
t1_std_data, t1_std_aff = load_nifti(t12std)
mask_data, mask_affine = load_nifti(bet_bin)
t1_betted = applymask(t1_std_data, mask_data)
save_nifti('t1_reg/T1_2_dwi_betted.nii.gz', t1_betted, t1_std_aff)

#dwi betting

masked_topup_v = 't1_reg/' + topup_base_n + '_betted.nii.gz'

run(['mrconvert', topup_mif, topup_nii])
topup_data, topup_affine = load_nifti(topup_nii)

topup_betted = applymask(topup_data, mask_data)

save_nifti(masked_topup_v, topup_betted, topup_affine)


# In[ ]:


mif_masked_topup = 't1_reg/' + topup_base_n + '_betted.mif'
run(['mrconvert', masked_topup_v, '-fslgrad', bvec, bval, mif_masked_topup])


# ## FAST 3tt

# In[ ]:


scgm_mask = 't1_reg/bin_scgm_mask.nii.gz'
t1_bet_dir = 't1_reg/T1_2_dwi_betted.nii.gz'
decimal_mat = 'flirt_mtrx.mat'
t1_fast = 't1_reg/T1_FAST.nii.gz'
run(['fast', '-a', decimal_mat, t1_bet_dir])


# In[ ]:


run('mkdir fast', shell=True)
run('mv t1_reg/T1_2_dwi_betted* fast', shell=True)


# In[ ]:


ls('fast')
pve_list = ['pve_' + str(i) for i in range(3)]
pve = []
for file in ls('fast'):
    for i in pve_list:
        if i in file:
            pve.append('fast/' + file)


# In[ ]:


pve


# In[ ]:


wm_data, wm_affine = load_nifti(pve[0])
gm_data, gm_affine = load_nifti(pve[1])
csf_data, csf_affine = load_nifti(pve[2])


# In[ ]:


# Remove subcortical gm from fast_gm:
gm_data, gm_affine = load_nifti(gm_fast)
gm_bin = (gm_data > 0) * 1.0
save_nifti('5tt/bin_gm.nii.gz', gm_bin, gm_affine)


# In[ ]:


sgm_path = 't1_reg/bin_scgm_mask.nii.gz'
sgm_data, sgm_affine = load_nifti(sgm_path)
sgm_bin = (sgm_data > 0) * 1.0
scgm_bin_path = '5tt/subc_gm_bin.nii.gz'
save_nifti(scgm_bin_path, sgm_bin, sgm_affine)


# In[ ]:





# In[ ]:


non_scgm = gm_bin - sgm_bin
non_scgm_bin = (non_scgm > 0) * 1.0
cortical_gm_path = '5tt/cortical_gm.nii.gz'
save_nifti(cortical_gm_path, non_scgm_bin, gm_affine)
cortical_gm = non_scgm_bin


# In[ ]:


# WM binary
wm_bin = (wm_data > 0) * 1.0
wm_clean = wm_bin - sgm_bin
wm_clean_bin = (wm_clean > 0) * 1.0
wm_clean_path = '5tt/wm_clean.nii.gz'
save_nifti(wm_clean_path, wm_clean_bin, wm_affine)


# In[ ]:


# # GM binary
gm_bin = (gm_data > 0) * 1.0
gm_clean = gm_bin - sgm_bin
gm_clean_bin = (gm_clean > 0) * 1.0
save_nifti('5tt/gm_clean.nii.gz', gm_clean_bin, gm_affine)


# In[ ]:


# Pathlogical tissue
pato_bin = (non_scgm_bin != 99) * 0
pato_path = '5tt/pato_bin.nii.gz'
save_nifti(pato_path, pato_bin, gm_affine)


# In[ ]:


# Concatenating all together c_gm, s_gm, wm, csf, pato
bin_tissues = (cortical_gm, sgm_bin, wm_clean_bin, csf_data, pato_bin)
concat_5tt = np.concatenate(bin_tissues, 2)


# In[ ]:


concat_manual = np.array([cortical_gm, sgm_bin, wm_clean_bin, csf_data, pato_bin]).T
concat_manual_path ='5tt/5tt_manual_concat_mask.nii.gz'
save_nifti(concat_manual_path, concat_manual, gm_affine)


# ## ESTA ES LA 5TT BUENA

# In[ ]:


T_concat_manual = np.array([cortical_gm.T, sgm_bin.T, wm_clean_bin.T, csf_data.T, pato_bin.T]).T
T_path = '5tt/transpose_5tt.nii.gz'
save_nifti(T_path, T_concat_manual, gm_affine)


# In[ ]:


fivettmask = T_path


# ### GMWM interface

# In[ ]:


gmwm_mask = '5tt/gm_wmfivettmaskerface_mask.nii.gz'
run(['5tt2gmwmi', fivettmask, gmwm_mask])


# ## 8. Response function estimation
# 

# In[ ]:


bvecs_raw_concat = bvec
bvals_raw_concat = bval


# In[ ]:


bet_bin


# In[ ]:


run('mkdir response_fx', shell=True)
voxels = 'response_fx/voxels.mif'
wm_rf  = 'response_fx/wm.txt'
gm_rf  = 'response_fx/gm.txt'
csf_rf = 'response_fx/csf.txt'
run(['dwi2response', 'dhollander', masked_topup_v, wm_rf, gm_rf, csf_rf,
     '-voxels', voxels, '-fslgrad', bvecs_raw_concat, bvals_raw_concat, '-mask', bet_bin])


# ## 9. FOD estimation and normalization

# In[ ]:


run('mkdir fods', shell=True)
wm_fod  = 'fods/wm_fod.mif'
gm_fod  = 'fods/gm_fod.mif'
csf_fod = 'fods/csf_fod.mif'
run(['dwi2fod', '-mask', bet_bin ,'-fslgrad', bvecs_raw_concat, bvals_raw_concat ,'msmt_csd', masked_topup_v, wm_rf, wm_fod, 
    gm_rf, gm_fod, csf_rf, csf_fod])


# In[ ]:


wm_fod_norm  = 'fods/wm_fod_norm.mif'
gm_fod_norm  = 'fods/gm_fod_norm.mif'
csf_fod_norm = 'fods/csf_fod_norm.mif'
run(['mtnormalise', wm_fod, wm_fod_norm, gm_fod, gm_fod_norm, 
     csf_fod, csf_fod_norm, '-mask', bet_bin])


# In[ ]:


run('mkdir metrics', shell=True)
tensor = 'metrics/DTI.mif'
run(['dwi2tensor', '-fslgrad', bvecs_raw_concat, bvals_raw_concat, masked_topup_v, tensor])

fa_map = 'metrics/fa_map.mif'
run(['tensor2metric', tensor, '-fa', fa_map])


# ## 10. Tractography generation

# In[ ]:


# 10 million streamlines tractography
run('mkdir tractography', shell=True)
tracto_10mio_norm = 'tractography/tracto10mio_norm.tck'
run(['tckgen', '-act', fivettmask, '-backtrack', '-seed_gmwm', 
     gmwm_mask, '-select', '10000000', wm_fod_norm, tracto_10mio_norm])


# In[ ]:


# Sift
tracto_sift_1mio = 'tractography/sift_1mio.tck'
run(['tcksift', '-act', fivettmask, '-term_number', 
    '1000000',tracto_10mio_norm, wm_fod_norm, tracto_sift_1mio])


# ## 11. Whole brain streamline registration

# In[ ]:


atlas_bundles_url = 'https://ndownloader.figshare.com/files/13638644/Atlas_80_Bundles.zip'
run(['wget', atlas_bundles_url])


# In[ ]:


atlas_bundles_zip = ''
for f in ls():
    if 'Atlas_80' in f: atlas_bundles_zip = f


# In[ ]:


run(['unzip', atlas_bundles_zip])


# In[ ]:


atlas_bundles = atlas_bundles_zip.split('.')[0]


# In[ ]:


from dipy.io.streamline import load_tck, save_tck, load_tractogram
from dipy.io.utils import create_tractogram_header
from dipy.data import (fetch_target_tractogram_hcp,
                      fetch_bundle_atlas_hcp842,
                      get_bundle_atlas_hcp842, get_two_hcp842_bundles)
from dipy.io.streamline import load_trk, save_trk, save_tck, load_tck
from dipy.align.streamlinear import whole_brain_slr

from dipy.segment.bundles import RecoBundles    
from dipy.io.stateful_tractogram import StatefulTractogram, Space


# In[ ]:


def create_streamlines_from_tck(tck_path, T1_2DWI_path):
    tck_data = load_tck(tck_path, T1_2DWI_path, bbox_valid_check=False)
    tck_streamlines = tck_data.streamlines
    tck_header = create_tractogram_header(tck_path, *tck_data.space_attributes)
    return(tck_data, tck_streamlines, tck_header)


# In[ ]:


def get_atlas_tractography_paths():
#     fetch_bundle_atlas_hcp842()
    whole_trk_path, bundles_path = get_bundle_atlas_hcp842()
    bundles_path = bundles_path.split('*.')[0]
    individual_bundles_path = []
    for file in ls(bundles_path):
        if '.trk' in file:
            trk_path = join(bundles_path, file)
            individual_bundles_path.append(trk_path)
    return(whole_trk_path, individual_bundles_path)


# In[ ]:


def create_streamlines_from_trk(trk_path):
    trk_data = load_trk(trk_path, 'same', bbox_valid_check=False)
    trk_streamlines = trk_data.streamlines
    trk_header = create_tractogram_header(trk_path, *trk_data.space_attributes)
    return(trk_data, trk_streamlines, trk_header)


# In[ ]:


from dipy.align.streamlinear import whole_brain_slr
def subject_2_atlas_reg(whole_atlas_streamlines, whole_subject_streamlines):
    subj_reg, subj_transform, qb_centroids1, qb_centroids2 = whole_brain_slr(whole_atlas_streamlines, whole_subject_streamlines,
                                                                             x0='affine', verbose=True, progressive=True,
                                                                             rng=np.random.RandomState(1984))
    np.save('Reco_bundles/whole_trk_transform.npy', subj_transform)
    return(subj_reg, subj_transform)


# In[ ]:


from dipy.segment.bundles import RecoBundles
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.stateful_tractogram import Space

def bundles_extraction(model_bundle_path, subject_reg, subject_streamlines, subject_header):
    model_data, model_streamlines, model_header = create_streamlines_from_trk(model_bundle_path)
    subject_bundles = RecoBundles(subject_reg, verbose=True, rng=np.random.RandomState(2001))
    rb_bundle, bundle_labels = subject_bundles.recognize(model_bundle=model_streamlines, model_clust_thr=0.1,
                                                        reduction_thr=15, reduction_distance='mdf',
                                                        pruning_distance='mdf', slr=True)
    # Subject space
    subject_recognized_bundle = StatefulTractogram(subject_streamlines[bundle_labels], subject_header, 
                                                  Space.RASMM)
    bundle_name_trk = model_bundle_path.split('/')[-1]
    name_bundle_subject  = bundle_name_trk.split('.')[0] + '.tck'
    subject_path =  join('Reco_bundles/subject_bundles', name_bundle_subject)
    save_tck(subject_recognized_bundle, subject_path, bbox_valid_check=False)
    
    # MNI space
    MNI_bundles = StatefulTractogram(rb_bundle, model_header, Space.RASMM)
    name_MNI = bundle_name_trk.split('.')[0] + '_MNI.tck'
    MNI_path = join('Reco_bundles/MNI_subject_bundles', name_MNI)
    save_tck(MNI_bundles, MNI_path, bbox_valid_check=False)
    
    return()
    


# ## Load the subject streamlines

# In[ ]:


sift_data, sift_streamlines, sift_header = create_streamlines_from_tck(sifted_tck_path, t12std)


# ## Get atlas paths

# In[ ]:


whole_atlas_path, atlas_individual_bundles_paths = get_atlas_tractography_paths()


# Load atlas streamlines

# In[ ]:


atlas_data, atlas_streamlines, atlas_header = create_streamlines_from_trk(whole_trk_path)


# ## Whole subject to atlas streamlines registration

# In[ ]:


run('mkdir Reco_bundles', shell=True)
streamlines_reg, stramlines_reg_transform = subject_2_atlas_reg(atlas_streamlines, sift_streamlines)


# ## Bundles extractio

# In[ ]:


run('mkdir Reco_bundles/subject_bundles', shell=True)
run('mkdir Reco_bundles/MNI_subject_bundles', shell=True)
for model_bundle_path in atlas_individual_bundles_paths:
    bundles_extraction(model_bundle_path, streamlines_reg, 
                       sift_streamlines, sift_header)

