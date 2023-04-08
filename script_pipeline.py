#!/usr/bin/env python
# coding: utf-8

# <center>
#     <h1>Pipeline for generating automated fiber bundles extraction</h1>
# </center>
# <center><h3> Juan Rodrigo Guerrero Morales, María Guadalupe García Gomar</h3></center>
# <center><h4>14/01/23</h4></center>
# 
# ## 1. Tipo de datos.
# Para que el script funcione este debe de estar en el mismo directorio que las imagenes dicom extraìdas del resonador. Estas deben de ser al menos:
# * Uno o màs volùmenes pesados por difusiòn (pueden tener el mismo valor b o diferentes) asì como un archivo de texto con sus *bvals* y otro con sus *bvecs*.
# * Un volumen en *reverse phase encoding*
# * Un volumen anatòmico pesado en T1.
# # 2. Preprocesamiento.
# Lo primero que hay que hacer es convertir las imàgenes en formato dcm a nifti. Para esto uso dcm2nii.
# 
# Li X, Morgan PS, Ashburner J, Smith J, Rorden C. (2016) The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods. 264:47-56.
# ## 2. Denoising using Local-PCA

# In[ ]:


# Importamos las librerìas necesarias
# listdir va hacer lo mismo que el comando -ls de bash, 
# este nos va a mostrar todos los archivos del directorio actual
from os import listdir as ls
# Subprocess nos va a permitir correr comandos de la terminal 
# desde la sesion de python actual.
from subprocess import run
def dcm2niix(directory=ls()):
    # Creamos un forloop que itere por los archivos de nuestro directorio 
    # Los archivos dcm son directorios asi que no tienen extension '.'.
    for folder in directory:
        if '.' not in folder:
            run(['dcm2niix', '-o', '.',  folder])
# Convertimos los archivos 
dcm2niix()


# In[1]:


# join va a juntar directorios
from os.path import splitext, join
# subprocess es para correr comandos desde la terminal
from subprocess import run
# Importamos las librerias de dipy para cargar/salvar imagenes pesadas por difusion, leer los
# bvals y bvecs, asi como crear una tabla de gradientes apartir de ellos.
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
# Numpy para hacer operaciones con los volumenes
import numpy as np


# In[2]:


# Hago una lista con todos los archivos que se encuentran en mi directorio actual
# cwd = getcwd()
# cwd_files = ls()
# # Ahora debo encontrar las imagenes pesadas por difusion y sus bvals/vecs.
# dwi_raw_data   = []
# dwi_raw_affine = []
# bvecs = []
# bvals = []
# directories = []
# Defino una funcion para poder cargar todos los volumenes que sean de un tipo en especifico
# Por ejemplo, dwi, t1, rpe.
def load_volumes():
    from os import getcwd
    from os import listdir as ls
    # Hago una lista con todos los archivos que se encuentran en mi directorio actual
    cwd       = getcwd()
    cwd_files = ls()
    dict_of_data   = {}
    dict_of_affine = {}
    dict_of_paths  = {}
    dict_of_bvals  = {}
    dict_of_bvecs  = {}
    
    print('Loaded volumes:\n' + '-'*45)
    for file in cwd_files:
        if '.nii' in file: 
            base_name = file.split('.')[0]
            vol   = join(cwd, file)
            bvec     = join(cwd, base_name + '.bvec')
            bval     = join(cwd, base_name + '.bval')
            data, affine = load_nifti(vol)
            
            
            if len(data.shape) == 4:
                # Asegurandome de que sea un vol pesado por difusion
                if data.shape[-1] > 10:
                    # Esta seria una dwi en phase encoding
                    vol_type = 'dwi_volume'

                
                else:
                    vol_type = 'rpe'

            elif len(data.shape) == 3:
                vol_type = 'anatomical' 
            print(vol_type + ': ' + file)
            if vol_type not in dict_of_data.keys():
                dict_of_data[vol_type] = []
                dict_of_data[vol_type].append(data)
                

            
            else:
                new = dict_of_data[vol_type].copy()
                new.append(data)
                dict_of_data[vol_type] = new
                

            
            if vol_type not in dict_of_paths:
                dict_of_paths[vol_type] = []
                dict_of_paths[vol_type].append(vol)
            
            else:
                new_p = dict_of_paths[vol_type].copy()
                new_p.append(vol)
                dict_of_paths[vol_type] = new_p                
            
            if vol_type not in dict_of_affine.keys():
                dict_of_affine[vol_type] = []
                dict_of_affine[vol_type].append(affine)
                
            else:
                new = dict_of_affine[vol_type].copy()
                new.append(affine)
                dict_of_affine[vol_type] = new
                #             print(bvec, bval)


            ls_path = [join(cwd, i) for i in ls()]
            if bvec and bval in ls_path:
                bval_read, bvec_read = read_bvals_bvecs(bval, bvec)

                if vol_type not in dict_of_bvals.keys():
                    dict_of_bvals[vol_type] = []
                    dict_of_bvals[vol_type].append(bval_read)
                else:
                    new = dict_of_bvals[vol_type].copy()
                    new.append(bval_read)
                    dict_of_bvals[vol_type] = new

                if vol_type not in dict_of_bvecs.keys():
                    dict_of_bvecs[vol_type] = []
                    dict_of_bvecs[vol_type].append(bvec_read)
                else:
                    new = dict_of_bvecs[vol_type].copy()
                    new.append(bvec_read)
                    dict_of_bvecs[vol_type] = new
            
            
#     return(raw_data, raw_affine, bvecs, bvals, directories)
    return(dict_of_data, dict_of_affine, dict_of_bvecs, dict_of_bvals, dict_of_paths)     


# In[3]:


raw_data, raw_affine, bvecs, bvals, paths = load_volumes()


# In[4]:


# Ya que tenemos todas las imagenes DWI, vamos a usar numpy para concatenarlas en el eje '3' 
# (tiempo)

# Guarde los datos en el diccionario raw_data, asi que necesito acceder a la key `dwi_volumes` 

# Vamos a hacer una copia de mis datos dwi
dwi_raw_data = raw_data['dwi_volume'].copy()

# El eje de concatenacion se indica en el segundo argumento de np.concatenate
dwi_raw_concat = np.concatenate(dwi_raw_data, 3)
# Los bvecs y bvals son concatenados en el eje 'x' = 0
bvecs_dwi = bvecs['dwi_volume'].copy()
bvals_dwi = bvals['dwi_volume'].copy()
bvecs_concat   = np.concatenate(bvecs_dwi, 0)
bvals_concat   = np.concatenate(bvals_dwi, 0)
# Para guardar una imagen con dipy, requiero su afin.
# Acabo de crear una nueva imagen dwi concatenada, asi que debo crear un nuevo afin
# concatenado (no estoy seguro de si use este para exportar la imagen concatenada)
dwi_affine = raw_affine['dwi_volume'].copy()
dwi_affine_concat = np.concatenate(dwi_affine, 0)


# In[ ]:


# Para realizar el denoising se debe importar `localpca` y `pca_noise_estimate`
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate

# Con `gradient_table` se obtiene la tabla de gradientes apartir de los bvecs y bvals 
# concatenados
gtab = gradient_table(bvals_concat, bvecs_concat)

# np.seterr va a determinar como se van a manejar los errores al hacer operaciones 
# con floats, si le pongo ignore, solo va a ignorar cuando estos sucedan.
np.seterr(invalid='ignore')

# Estimo el valor de sigma usando pca_noise_estimate a partir de mi esquema de difusion y
# mi volumen concatenado
# sigma = pca_noise_estimate(dwi_raw_concat, gtab, correct_bias=True, smooth=3)

# Realizo el denoising con mi volumen y el sigma que acabo de obtener# Estos son 
# local_pca_dwi = localpca(dwi_raw_concat, sigma, tau_factor=2.3, patch_radius=2, pca_method='svd')


# In[ ]:


# Importo `gibbs_removal` y la uso para realizar el unringing en el volumen sin ruido
from dipy.denoise.gibbs import gibbs_removal
# dwi_gibbs_vol = gibbs_removal(local_pca_dwi)

# Creo un directorio en donde guardar los outputs del preproceso
# run('mkdir preprocessing', shell=True)

# Creo tres directorios, el primero es para guardar la imagen concatenada sin preprocesar
# el segundo es para guardar la imagen concatenada sin local_pca
# el tercero es para guardar el volumen sin anillos de gibbs
folders = ['concat', 'local_pca', 'gibbs']

for folder in folders:
    new_dir = join('preprocessing', folder)
    run('mkdir ' + new_dir, shell=True)

# Set output paths 
# Declaro en donde quiero guardar el volumen crudo concatenado, sus bvals/vecs y la 
# tabla de gradientes.
dwi_c_path = 'preprocessing/concat/dwi_concat'
dwi_c_bval = 'preprocessing/concat/dwi_concat_bval'
dwi_c_bvec   = 'preprocessing/concat/dwi_concat_bvec'
gtab_path = 'preprocessing/concat/dwi_concat_gtab'

# Hago lo mismo para el volumen con denoising de local-pca(svd)
dwi_local_path = 'preprocessing/local_pca/dwi_localpca'
localpca_affine = 'preprocessing/local_pca/affine_localpca'

# Y para el volumen sin anillos de Gibbs
dwi_gibbs_path = 'preprocessing/gibbs/dwi_gibbs'
gibbs_affine   = 'preprocessing/gibbs/gibbs_affine'

# Declaro los nombres de los archivos en nii.gz
nifti_concat_path = dwi_c_path + '.nii.gz'
nifti_pca = dwi_local_path + '.nii.gz'
nifti_gibbs = dwi_gibbs_path + '.nii.gz'
# Guardo los archivos
save_nifti(nifti_concat_path , dwi_raw_concat, dwi_affine[0])
save_nifti(nifti_pca, local_pca_dwi, dwi_affine[0])
save_nifti(nifti_gibbs, dwi_gibbs_vol, dwi_affine[0])
# Ahora creo los bvals y bvecs en formato fsl. Para esto solo tengo que 
# hacer la traspuesta de los bvecs concatenados, que cada entrada este separada por un espacio
# y que sea un float
np.savetxt(dwi_c_bvec, bvecs_concat.T, delimiter=' ', fmt='% f')
# la opcion 'newline' va a decir que en vez de hacer una nueva linea, ponga un espacio
np.savetxt(dwi_c_bval, bvals_concat, delimiter=' ', fmt='% f', newline=' ')


# # 5. Motion and distortion correction
# ## 5.1 Obtener el volumen con las medias de las b0
# ### 5.1.1 Phase enconding 

# In[9]:


# # Lo primero que se debe de hacer es crear una volumen 3D que sea el promedio de las
# imagenes b0 del volumen sin anillos de gibbs. 
# La unica manera de saber donde se enceuntran los volumenes b0 es viendo que bvals 
# son 0. 
# Si hago la operacion bvals_concat == 0, el output va a ser una matriz donde todas las entradas
# que sean igual a zero sean 1 (True) y los demas 0 (False).
condition_bvals = bvals_concat == 0
# Cada np.array puede ser indexado usando el formato x[i:j:k] donde i = start,
# j = step, k = end.
# el elipsis '...' va a darme las entradas que sean igual a la condicion ==0 del volumen 
# sin anillos de Gibbs
dwi_gibbs_vol, dwi_guibbs_aff = load_nifti(nifti_gibbs)
bzeros = dwi_gibbs_vol[..., condition_bvals] # '...'ellipsis
# Ahora solo obtengo la media 
bzero_mean = np.mean(bzeros, axis=3)
# Creo un directorio para el motion and distortion correction y declaro el path de mi
# volumen con la media de las imagenes bzero.
# run('mkdir preprocessing/eddy', shell=True)
b0_mean_pth = 'preprocessing/eddy/bzeros_mean'
# Guardo el volumen
save_nifti(b0_mean_pth + '.nii.gz', bzero_mean, dwi_affine)
# Saving as a binary
np.save(b0_mean_pth, bzero_mean)
# Save the affine as a binary
np.save('preprocessing/raw_affine', dwi_affine)


# # De aqui para arriba lo volvi a correr

# ### 5.1.2 Reverse phase encoding

# In[10]:


# Para obtener sus bzero necesito cargar la imagen y su esquema de difusion
ep_data, ep_affine =  raw_data['rpe'][0], raw_affine['rpe'][0]
bvals_pe, bvecs_pe  = bvals['rpe'][0], bvecs['rpe'][0]
# el elipsis '...' va a darme las entradas que sean igual a la condicion ==0 del volumen 
condition_pe_bvals = bvals_pe == 0
bzeros_ep = ep_data[...,condition_pe_bvals] # '...'ellipsis
# Como solo es una imagen b0 no necesito hacer el mean.
# Concateno ambos volumenes de bzeros y los guardo.
# Nombre del volumen
bzero_pair_fname = 'preprocessing/eddy/bzeros_pair'
# Por alguna razon debo hacer la concatenacion manual 
# ep_data[:,:,:,0] me va a dar el valor de cada entrada que se encuentre
# en el lugar 0 del array
# Supongo que lo hice por que el volumen en reverse phase encoding 
# inicia con la imagen en b0 (index 0). Pero entonces
# por que no usar el array que obtuve con la condicional `condition_pe_bvals`?


# In[ ]:


# Hago el mean de las bzeros ( en este caso es solo una )
bzeros_ep_mean = np.mean(bzeros_ep, axis=3)
# Concateno las rpe_B0 y las dwi_B0
bzero_pair = np.array([bzero_mean.T, bzeros_ep_mean.T]).T
# Salvando 
save_nifti(bzero_pair_fname + '.nii.gz', bzero_pair, dwi_affine[0])
np.save(bzero_pair_fname, bzero_pair)


# ## 5.2 Correr 'Eddy'

# In[11]:


# Declaro el nombre del volumen ya preprocesado con eddy
topup_out = 'preprocessing/eddy/preprocessed_vol'
topup_out_nii = topup_out + '.nii.gz'
# Obtengo el nombre de los volumenes b0 en phase encoding and reverse phase
bzero_pair_nii = bzero_pair_fname + '.nii.gz'
# Para el denoising voy a usar el volumen sin ruido y corregido para anillos de Gibbs
gibbs_nii = dwi_gibbs_path + '.nii.gz'


# In[12]:


# Corro eddy
run('dwifslpreproc' +' '+ gibbs_nii +' '+ topup_out_nii + ' -pe_dir' + ' AP' + ' -rpe_pair' + ' -se_epi ' +' '+ bzero_pair_nii +' '+ ' -eddy_options ' + ' " --flm=linear " ' + '-grad ' + gtab_path, shell=True)


# In[13]:



# Cargo el volumen preprocesado 
data_preproc, affine_preproc, voxel_size_preproc = load_nifti(topup_out_nii, return_voxsize=True)
# Guardo los datos del volumen y su afin en binarios
np.save(topup_out, data_preproc)
np.save(topup_out + '_affine', affine_preproc)


# # 6 ACT usando FSL
# Voy a tratar de crear las mascaras de 5tt usando FAST en espacio T1. Despues debo usar run_fist para obtener la mascara subcortical.
# ## 6.1 Encontrar la T1 y reorientarla

# In[18]:


# El primer paso es encontrar el volumen T1 en el directorio actual
# Creo otro directorio para guardar las segmentaciones
run('mkdir ACT', shell=True)
T1_anatomical = paths['anatomical'][0]
# Ya que tengo la T1, debo de reorientarla a espacio estandar
T1_oriented = 'ACT/T1_oriented.nii.gz'
run(['fslreorient2std', T1_anatomical, T1_oriented])


# ## 6.2 Hacer media de las imagenes b0 de los volumenes preprocesados

# In[19]:


# Los datos del volumen preprocesados se encuentran cargados en la
# variable `data_preproc`, `condition_bvals` va a hacer que solo se
# extraigan los volumenes asociados a un bval == 0.
# Sigo sin entender bien el Elipsis(...) pero es una manera de poder
# indexar usando el condicionante de los bvals.
bzeros_preproc = data_preproc[..., condition_bvals]
# Obtengo la media. El axis 3 me indica el tiempo, ya que las direcc-
# iones se encuentran codificadas en el tiempo puedo obtener la media 
# de todas y que resulte un solo volumen en 3D.
bzeros_preproc_mean = np.mean(bzeros_preproc, axis=3)
b0_preproc_mean_fname   = 'ACT/preproc_vol_mean_b0s.nii.gz'
# Guardo el volumen de bzeros preprocesados.
save_nifti(b0_preproc_mean_fname, bzeros_preproc_mean, dwi_affine)


# ## 6.3 Obtener whole-brain mask usando Bet2

# In[3]:


run(['bet'])


# In[ ]:


# Bet por si solo agarra un poco de tejido del cuello anterior, 
# para arreglarlo hay que indicar que use como centro de gravedad 
# el splenio. Las cordenadas son 87, 106, 138 en voxeles.
bet_dir = 'ACT/bet_f06'
T1_oriented = 'ACT/T1_oriented.nii'
bet_masked_fname = bet_dir + '.nii.gz'
bet_mask_bin  = bet_dir + '_mask.nii.gz'
# run(['bet2', T1_oriented, bet_dir, '-f', '0.6', '-c', '87', '106', '138', '-m'])
run(['bet', T1_oriented, bet_dir, '-f', '0.6', '-m'])


# In[11]:


bet_dir = 'ACT/bet1_f06_neck'
T1_oriented = 'ACT/T1_oriented.nii'
bet_masked_fname = bet_dir + '.nii.gz'
bet_mask_bin  = bet_dir + '_mask.nii.gz'
run(['bet', T1_oriented, bet_dir, '-f', '0.6', '-m', '-B'])


# In[13]:


# Salio muy recortada de frontal y occipital, voy a reducir f a 0.4
bet_dir = 'ACT/bet1_f04_neck'
T1_oriented = 'ACT/T1_oriented.nii'
bet_masked_fname = bet_dir + '.nii.gz'
bet_mask_bin  = bet_dir + '_mask.nii.gz'
run(['bet', T1_oriented, bet_dir, '-f', '0.4', '-m', '-B'])


# In[16]:


run(['fslview', bet_masked_fname])


# In[15]:


# Sigue saliendo muy recortada de frontal y occipital, voy a reducir f a 0.35
bet_dir = 'ACT/bet1_f035_neck'
T1_oriented = 'ACT/T1_oriented.nii'
bet_masked_fname = bet_dir + '.nii.gz'
bet_mask_bin  = bet_dir + '_mask.nii.gz'
run(['bet', T1_oriented, bet_dir, '-f', '0.35', '-m', '-B'])


# In[17]:


# Sigue saliendo muy recortada de frontal y occipital, voy a reducir f a 0.3
bet_dir = 'ACT/bet1_f03_neck'
T1_oriented = 'ACT/T1_oriented.nii'
bet_masked_fname = bet_dir + '.nii.gz'
bet_mask_bin  = bet_dir + '_mask.nii.gz'
run(['bet', T1_oriented, bet_dir, '-f', '0.3', '-m', '-B'])


# In[19]:


# Sigue saliendo muy recortada de frontal y occipital, voy a reducir f a 0.25
bet_dir = 'ACT/bet1_f025_neck'
T1_oriented = 'ACT/T1_oriented.nii'
bet_masked_fname = bet_dir + '.nii.gz'
bet_mask_bin  = bet_dir + '_mask.nii.gz'
run(['bet', T1_oriented, bet_dir, '-f', '0.25', '-m', '-B'])
# Este se ve bien


# In[1]:


from subprocess import run


# In[2]:


run(['fslview', 'ACT/T1_oriented.nii', 'ACT/bet1_f025_neck.nii.gz'])


# In[20]:


run(['fslview', bet_masked_fname])


# ## 6.4 Obteniendo segmentacion inicial de tres tipos de tejidos con FAST

# In[21]:


# Solo corro FAST desde la terminal
run(['fast', bet_masked_fname])
# Hago una lista con los nombres de la segmentacion de tejidos
pve_list = [bet_dir + '_pve_' + str(i) + '.nii.gz' for i in [1,2,3]]


# In[31]:


pve_list


# ## 6.4 Hacer registro de T1 a espacio MNI 

# In[22]:


# first_flirt va a hacer el registro de la imagen T1 sujeto a T1 atlas, esto lo hago 
# para obtener la matriz de transformacion que voy a usar despues para first
T1_mni_space = 'ACT/T1_mni_space_2.nii.gz'
T1_mni_mat = 'ACT/T1_mni_space_2.mat'
run(['first_flirt', T1_oriented, T1_mni_space])
# Ahora debo de convertir la matriz a decimal, ya que first solo lee en decimal 
# Para eso solo debo de cargar la matriz con numpy y decirle que la guarde en decimal


# In[3]:


run(['fslview', 
     '/home/inb/lconcha/fmrilab_software/fsl_5.0.6/data/standard/MNI152_T1_1mm.nii.gz',
    'ACT/T1_mni_space_2.nii.gz'])


# In[24]:


# Cada entrada de la matriz debe ser float y se separada por espacios
mat_hex_data = np.loadtxt(T1_mni_mat)
decimal_mat = 'ACT/T1_mni_dec_2.mat'
np.savetxt(decimal_mat, mat_hex_data, delimiter=' ', fmt='% f')


# ## 6.5 Obtener mascaras de sustancia gris subcortical con First

# In[25]:


models_dir = '/home/inb/lconcha/fmrilab_software/fsl_5.0.6/data/first/models_336_bin/'
## from models_336_bin, we only need the caudate
cau_fname = '_Caud_bin.bmv'
caudate = ['L' + cau_fname, 'R' + cau_fname]
L_caudate_path, R_caudate_path = [join(models_dir, i) for i in caudate]

## All the others are located in the 05mm dir
s_gm_models_dir = join(models_dir, '05mm')
ls_05mm = ls(s_gm_models_dir)

# We dont need the cerebellum
ls_05mm.remove('R_Cereb_05mm.bmv')
ls_05mm.remove('L_Cereb_05mm.bmv')

# Obtaining the subcortical gray matter models paths
sc_gm_models = [join(s_gm_models_dir, i) for i in ls_05mm]
# Appending the caudate paths
sc_gm_models.append(L_caudate_path)
sc_gm_models.append(R_caudate_path)

# Creo el directorio para colocar los modelos de first
run('mkdir ACT/first_models', shell=True)

# Tengo los paths de los modelos en la variable sc_gm_models (subcortical_graymater_models)
# Asi que hago un for para obtener la segmentacion por cada modelo de esa lista

models_paths   = []
# En la lista voy a guardar los paths de los modelos que se generen
for model in sc_gm_models:
    model_name = model.split('/')[-1]
    model_nii  = model_name.split('.')[0] + '.nii.gz'
    model_out = join('ACT/first_models', model_nii)
    run(['run_first', '-i', T1_oriented, '-v', '-t', decimal_mat, 
         '-n', '40', '-o', model_out, '-m', model])
    models_paths.append(model_out)
    
# Sumo las mascaras de las estructuras subcorticales en una sola mascara subcortical
models_data = []
models_affine = []
for path in models_paths:
    data, affine = load_nifti(path)
    models_data.append(data)
    models_affine.append(data)

# Cargo la segmentacion de gray mater usando fast para obtener su afin y usarla
# para guardar la de sustancia gris subcortical
gm_data, gm_affine = load_nifti(pve_list[0])
# Voy a guardaro en float32 ya que ese formato es en el que se encuentran las imagenes
# crudas del resonador (y aparte cuando lo guardo sin hacer eso me aparece un warning
# de nibabel diciendo que puede que cause problemas de incopatibilidad)
models_sum = np.float32(np.sum(models_data, 0))
# Binarizo la sumatoria y la guardo 
bin_models_sum = (models_sum > 0) * 1.0
bin_scgm_fname = 'ACT/bin_scgm_mask.nii.gz'
save_nifti(bin_scgm_fname, bin_models_sum, gm_affine)


# ## 6.6 Retirar la scgm de sustancia gris y sustancia blanca de FAST

# In[43]:


run(['fslview', 'ACT/bet1_f025_neck_pve_1.nii.gz'])


# In[44]:


# Cargo las mascaras de fast
csf_data, csf_affine = load_nifti('ACT/bet1_f025_neck_pve_0.nii.gz')
wm_data, wm_affine   = load_nifti('ACT/bet1_f025_neck_pve_2.nii.gz')
gm_data, gm_affine   = load_nifti('ACT/bet1_f025_neck_pve_1.nii.gz')
fast_data, fast_affine = [csf_data, wm_data, gm_data], [csf_affine, wm_affine, gm_affine]
# Defino la funcion para restar esrtructuras

def extract_mask(fast_mask, external_mask):
    # A la mascara actual le resto la mascara de scgm
    extraction = fast_mask - external_mask
    extraction[extraction < 0] = 0 
    extraction_round = np.round(extraction, decimals=2)
    return(extraction_round)
tissues = ['csf', 'wm', 'gm']
no_scgm_tissues = []
final_tissues = []
for i in range(3):
    tissue_data, tissue_affine = fast_data[i], fast_affine[i]
    new_tissue = extract_mask(tissue_data, bin_models_sum)
    name = tissues[i] + '_no_scgm.nii.gz'
    final_tissues.append(name)
    no_scgm_tissues.append(new_tissue)
    save_nifti(name, new_tissue, tissue_affine)


# In[45]:


pato = np.zeros(no_scgm_tissues[0].shape)
# Concateno todo junto 
gm_T = no_scgm_tissues[2].T
sc_T = bin_models_sum.T
wm_T = no_scgm_tissues[1].T
# Debo de cargar los datos de la mascara de csf
csf_T = no_scgm_tissues[0].T
fivett = np.round(np.array([gm_T, sc_T, wm_T, csf_T, pato.T]).T, 2)

# Guardo la mascara 5tt
save_nifti('fivett_t1_space.nii.gz', fivett, wm_affine)


# In[4]:


run(['5ttcheck', 'fivett_t1_space.nii.gz'])


# # INICIA CORRECION

# In[30]:


# Utilizar ACT/preproc_vol_mean_b0s.nii.gz como referencia para la transformada
run(['antsApplyTransforms', '-d', '3', '-e', '0', '-i', 'gmwmi_T1.nii.gz', '-r', 
    'ACT/preproc_vol_mean_b0s.nii.gz', '-t', 'MNI_0GenericAffine.mat',
   '-o', 'gmwmi_dwi_b0sref.nii.gz', '-n', 'NearestNeighbor'])
data_gmwmi_dwi, affine_gmwmi_dwi = load_nifti('gmwmi_dwi_b0sref.nii.gz')


# In[32]:


# Veo que si tenga las dimensiones correctas
run(['fslinfo', 'gmwmi_dwi_b0sref.nii.gz'])


# In[34]:


# Voy a hacer lo mismo pero para la 5tt
# Transformando las imagenes de espacio T1 a DWI
# Hago la transformada
final_tissues = ['csf_no_scgm.nii.gz',
 'wm_no_scgm.nii.gz',
 'gm_no_scgm.nii.gz',
 'ACT/bin_scgm_mask.nii.gz']
tissues_dwi = []
tissues_dwi_data, tissues_dwi_af = [], []
for tissue in final_tissues:
    new_name = tissue.split('.')[0] + '_dwi_b0sref.nii.gz'
    tissues_dwi.append(new_name)
    run(['antsApplyTransforms', '-d', '3', '-e', '0', '-i', tissue, '-r', 
        'ACT/preproc_vol_mean_b0s.nii.gz', '-t', 'MNI_0GenericAffine.mat',
       '-o', new_name, '-n', 'NearestNeighbor'])
    data, affine = load_nifti(new_name)
    tissues_dwi_data.append(data)
    tissues_dwi_af.append(affine)
# hago la 5tt en dwi space
pato = np.zeros(data_gmwmi_dwi.shape)
# Concateno todo junto 
gm_T = tissues_dwi_data[2].T
sc_T = tissues_dwi_data[3].T
wm_T = tissues_dwi_data[1].T
# Debo de cargar los datos de la mascara de csf
csf_T = tissues_dwi_data[0].T
fivett = np.round(np.array([gm_T, sc_T, wm_T, csf_T, pato.T]).T, 2)


# In[36]:


# Guardo la mascara 5tt
save_nifti('fivett_dwi_b0sref.nii.gz', fivett, tissues_dwi_af[0])


# In[39]:


run(['5ttcheck', 'fivett_dwi_b0sref.nii.gz'])
# Se ven bien 


# In[ ]:


# Obtener FODS


# In[76]:


# Transformar la mascara betT1 a dwi
run(['antsApplyTransforms', '-d', '3', '-e', '0', '-i', 'ACT/bet1_f025_neck_mask.nii.gz',
     '-r', 'MNI_Warped.nii.gz', '-t', 'MNI_0GenericAffine.mat',
   '-o', 'bet1_f025_dwi.nii.gz', '-n', 'NearestNeighbor'])


# In[16]:


# Para estimar la funcion de respuesta uso dwi2response de mrtrix3
run('mkdir response_fx', shell=True)
voxels = 'response_fx/voxels.mif'
wm_rf  = 'response_fx/wm.txt'
gm_rf  = 'response_fx/gm.txt'
csf_rf = 'response_fx/csf.txt'
run(['dwi2response', 'dhollander', 'preprocessing/eddy/preprocessed_vol.nii.gz',
     wm_rf, gm_rf, csf_rf,'-voxels', voxels, '-fslgrad', 
     'preprocessing/concat/dwi_concat_bvec', 
     'preprocessing/concat/dwi_concat_bval', 
     '-mask', 'bet1_f025_dwi.nii.gz'])


# In[17]:


# Obtengo los FODs a apartir de la funcoin de respuesta
run('mkdir fods', shell=True)
wm_fod  = 'fods/wm_fod.mif'
gm_fod  = 'fods/gm_fod.mif'
csf_fod = 'fods/csf_fod.mif'

run(['dwi2fod', '-mask', 'bet1_f025_dwi.nii.gz' ,'-fslgrad', 
 'preprocessing/concat/dwi_concat_bvec',
 'preprocessing/concat/dwi_concat_bval',
 'msmt_csd', 'preprocessing/eddy/preprocessed_vol.nii.gz',
 wm_rf, wm_fod, gm_rf, gm_fod, csf_rf, csf_fod])


# In[18]:


# NORMALIZO
wm_fod_norm  = 'fods/wm_fod_norm.mif'
gm_fod_norm  = 'fods/gm_fod_norm.mif'
csf_fod_norm = 'fods/csf_fod_norm.mif'
run(['mtnormalise', wm_fod, wm_fod_norm, gm_fod, gm_fod_norm, 
     csf_fod, csf_fod_norm, '-mask', 'bet1_f025_dwi.nii.gz'])


# In[29]:


run(['fslinfo', 'preprocessing/eddy/preprocessed_vol.nii.gz'])


# In[8]:


get_ipython().run_line_magic('pwd', '')


# In[28]:


run(['fslinfo', 'fivett_dwi.nii.gz'])
# La mascara de gmwmi tiene dimansiones distintas (1x1x1)


# # TRACTO CORRECTA

# In[3]:


# 10 million streamlines tractography pero con las 5tt, gmwmi corregidas y los vectores
fivettmask = 'fivett_dwi_b0sref.nii.gz'
gmwm_mask_fname = 'gmwmi_dwi_b0sref.nii.gz'
wm_fod_norm = 'fods/wm_fod_norm.mif'
run('mkdir tractography_b0sref', shell=True)
tracto_10mio_norm = 'tractography_b0sref/_b0sreftracto10mio.tck'
run(['tckgen', '-act', fivettmask,'-debug', '-backtrack', '-seed_gmwm', 
     gmwm_mask_fname, '-select', '10000000', wm_fod_norm, tracto_10mio_norm])


# In[4]:




# SIFT
tracto_sift_1mio = 'tractography_b0sref/sift_1mio.tck'
run(['tcksift', '-act', fivettmask, '-term_number', 
    '1000000',tracto_10mio_norm, wm_fod_norm, tracto_sift_1mio])


# Importo todo el desmadre
from dipy.io.streamline import load_tck, save_tck, load_tractogram
from dipy.io.utils import create_tractogram_header
from dipy.data import (fetch_target_tractogram_hcp,
                      fetch_bundle_atlas_hcp842,
                      get_bundle_atlas_hcp842, get_two_hcp842_bundles)
from dipy.io.streamline import load_trk, save_trk, save_tck, load_tck
from dipy.align.streamlinear import whole_brain_slr

from dipy.segment.bundles import RecoBundles    
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.segment.bundles import RecoBundles
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.stateful_tractogram import Space
import numpy as np
from dipy.align.streamlinear import whole_brain_slr



# Defino mis funciones
def create_streamlines_from_tck(tck_path, T1_2DWI_path):
    tck_data = load_tck(tck_path, T1_2DWI_path, bbox_valid_check=False)
    tck_streamlines = tck_data.streamlines
    tck_header = create_tractogram_header(tck_path, *tck_data.space_attributes)
    return(tck_data, tck_streamlines, tck_header)

def create_streamlines_from_trk(trk_path):
    trk_data = load_trk(trk_path, 'same', bbox_valid_check=False)
    trk_streamlines = trk_data.streamlines
    trk_header = create_tractogram_header(trk_path, *trk_data.space_attributes)
    return(trk_data, trk_streamlines, trk_header)

def get_atlas_tractography_paths():
    fetch_bundle_atlas_hcp842(())
    whole_trk_path, bundles_path = get_bundle_atlas_hcp842()
    bundles_path = bundles_path.split('*.')[0]
    individual_bundles_path = []
    for file in ls(bundles_path):
        if '.trk' in file:
            trk_path = join(bundles_path, file)
            individual_bundles_path.append(trk_path)
    return(whole_trk_path, individual_bundles_path)


def subject_2_atlas_reg(whole_atlas_streamlines, whole_subject_streamlines):
    subj_reg, subj_transform, qb_centroids1, qb_centroids2 = whole_brain_slr(whole_atlas_streamlines, whole_subject_streamlines,
                                                                             x0='affine', verbose=True, progressive=True,
                                                                             rng=np.random.RandomState(1984))
    np.save('Reco_bundles/whole_trk_transform.npy', subj_transform)
    return(subj_reg, subj_transform)

def bundles_extraction(model_bundle_path, subject_reg, subject_streamlines, subject_header, alpha=0.1,
theta=15):
    model_data, model_streamlines, model_header = create_streamlines_from_trk(model_bundle_path)
    subject_bundles = RecoBundles(subject_reg, verbose=True, rng=np.random.RandomState(2001))
    # Change model_clust_thr from 0.1 to 0.01 and reduction_thr from 15 to 30
    rb_bundle, bundle_labels = subject_bundles.recognize(model_bundle=model_streamlines, model_clust_thr=alpha,
                                                        reduction_thr=theta, reduction_distance='mdf',
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

from os import listdir as ls
from os.path import join 


# In[5]:


run(['fslview', 'MNI_Warped.nii.gz'])


# In[6]:


# ## Load the subject streamlines
warped_T1_std = 'MNI_Warped.nii.gz'
sift_data, sift_streamlines, sift_header = create_streamlines_from_tck(tracto_sift_1mio, warped_T1_std)
# ## Get atlas paths
whole_atlas_path, atlas_individual_bundles_paths = get_atlas_tractography_paths()
# Load atlas streamlines
atlas_data, atlas_streamlines, atlas_header = create_streamlines_from_trk(whole_atlas_path)
run('mkdir Reco_bundles', shell=True)
# ## Whole subject to atlas streamlines registration
streamlines_reg, stramlines_reg_transform = subject_2_atlas_reg(atlas_streamlines, sift_streamlines)
run('mkdir Reco_bundles/subject_bundles', shell=True)
run('mkdir Reco_bundles/MNI_subject_bundles', shell=True)

# ## Bundles extraction
for model_bundle_path in atlas_individual_bundles_paths:
    bundles_extraction(model_bundle_path, streamlines_reg, 
                       sift_streamlines, sift_header)

