# Pretrained Models
Pretrained models at checkpoint 50 for ODGnet and checkpoint 43 for fssdf
is here https://drive.google.com/drive/folders/1f-DPbcvOMFSfN61VTeiJiA_br67tZA6w?usp=sharing


# Code modifications:
1. Vamshi kumar Reddy - Pointnet.py model add data_augment function
line 186-208, all CUDE environment changes probably 35-55 lines,  overall around 50 lines.

Setup the pipeline to ingest incomplete point clouds and output completed ones from ODGNet

2. Srinik - Uptrans.py - Added batch normalization - changed around 50 lines
Set up pipeline to preprocess data from ODGNet and ingest into FS-SDF. 

3. Amaan Kayum - Data Generation and Movement scripts. test_extra sh files. train_text_extract.py
changes around 30-40 lines
Helped in integration of pipelines and modifying architectures


# ODGNet
## Requiremnets
Create any venv and do below
```
pip install -r requirements.txt
```

## Install extensions 
Setup Libs
Install pointnet2_ops_lib and Chamfer Distance in the extension Folder:
python3 setup.py install (in both folders)

## Data Set Download
use "sh script.sh" to download into the "/root/ODGNet/data.zip" folder. assuming pwd as "/root/ODGNet"
After download, unzip the files into a new folder data/ShapeNet55-34/shapenet_pc such that shapenet_pc folder will contain the preprocessed shapenet dataset

Create data/ShapeNet55-34/shapenet_subset as a sibling to shapenet_pc

```
python test_text_extract.py
python train_text_extract.py
```

The above python files will copy the selected subcatogories into the shapenet_subset which is used for training and testing

## Training/Testing
Please check the bash files, e.g., "sh train_55.sh" for the shapenet dataset.

"sh test.sh" for testing with the model localtion.
model placed at the location : experiments/UpTrans/ShapeNet55_models/shape55_upTransb1/ckpt-epoch-050.pth

# FSSDF Code
Create a conda environment as detailed in the readme of IFNet folder in fssdf code

FSSDF uses IFNet data preprocessing to generate the data. Now we move to the IfNet folder
and download and create the data for preprocessing.

## Install
```
conda env create -f if-net_env.yml
conda activate if-net
```

Install the needed libraries with:
```
cd data_processing/libmesh/
python setup.py build_ext --inplace
cd ../libvoxelize/
python setup.py build_ext --inplace
cd ../..
```

## Data Preparation

The processing will take atleast 1-2Hrs and the the data will explode to approx 90GB 
Download the [ShapeNet](https://www.shapenet.org/) data preprocessed by [Xu et. al. NeurIPS'19] from [here](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr)
into the `shapenet` folder.

Now extract the files into `shapenet\data` with: (for cabinet and lamp)

```
ls shapenet/02933112.tar.gz |xargs -n1 -i tar -xf {} -C shapenet/data/
ls shapenet/03636649.tar.gz |xargs -n1 -i tar -xf {} -C shapenet/data/
```
The above downloads will be 3-4GB

Next, the inputs and training point samples for IF-Nets are created. 

First, the data is converted to the .off-format and scaled using
```
python data_processing/convert_to_scaled_off.py
```

The input data for Voxel Super-Resolution of voxels is created with
```
python data_processing/voxelize.py -res 32
```
using `-res 32` for 32<sup>3</sup> and `-res 128` for 128<sup>3</sup> resolution.

The input data for Point Cloud Completion is created with
```
python data_processing/voxelized_pointcloud_sampling.py -res 128 -num_points 300
```
using `-num_points 300` for point clouds with 300 points and `-num_points 3000` for 3000 points.



To generate ground truth Signed distance values instead of occupancies run:
```
python data_processing/boundary_sampling_sdf.py -sigma 0.1
python data_processing/boundary_sampling_sdf.py -sigma 0.01
```
## Training
 To start a training please run  the following command:
 ````
 python train.py -res 128 -pc_samples 3000 -epochs 100 -inner_steps 5 -batch_size 8

````
  You can add the following  options `-p_enc` and `_p_dec` to initialize the encoder and/or the decoder with a pretrained model. Also, you can freeze the encoder during the training by adding the option  `-freeze` to your command. 

## Generation

The command:

````
python generate.py -res 128 -pc_samples 3000 -batch_size 8 -inner_steps 5 -exp <exp_name> -checkpoint <checkpoint>  
````
Where `exp_name` is the path to the folder containing the trained model checkpoints.

## Evaluation
Please run

```
python data_processing/evaluate.py -reconst -generation_path experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v32_mShapeNet32Vox/evaluation_10_@256/generation/
```
to evaluate each reconstruction, where `-generation_path` is the path to the reconstructed objects generated in the previous step.
> The above evaluation script can be run on multiple machines in parallel in order to increase generation speed significantly.

Then run
```
python data_processing/evaluate.py -voxels -res 32
```
 to evaluate the quality of the input. For voxel girds use '-voxels' with '-res' to specify the input resolution and for point clouds use '-pc' with '-points' to specify the number of points.

The quantitative evaluation of all reconstructions and inputs are gathered and put into `experiment/YOUR_EXPERIMENT/evaluation_CHECKPOINT_@256` using

```
python data_processing/evaluate_gather.py -voxel_input -res 32 -generation_path experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v32_mShapeNet32Vox/evaluation_10_@256/generation/
```
where you should use `-voxel_input` for Voxel Super-Resolution experiments, with `-res` specifying the input resolution or `-pc_input` for Point Cloud Completion, with `-points` specifying the number of points used.
