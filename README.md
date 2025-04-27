# SREGS: Sparse View Gaussian Splatting with Regularized Geometry and Region Exploration

## Environmental Setups
We provide install method based on Conda package and environment management:
```bash
conda env create --file environment.yml
conda activate SREGS
```

## Data Preparation
For different datasets, divide the training set and use Colmap to extract point clouds based on the training views. Mip-NeRF 360 uniformly divides 24 views as input, while LLFF uses 3 views.Note that extracting point clouds using this method requires a GPU-supported version of Colmap. The following steps follow the FSGS：

``` 
cd SREGS
mkdir dataset 
cd dataset

# download LLFF dataset
gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g

# run colmap to obtain initial point clouds with limited viewpoints
python tools/colmap_llff.py

# download MipNeRF-360 dataset
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -d mipnerf360 360_v2.zip

# run colmap on MipNeRF-360 dataset
python tools/colmap_360.py
```
If you encounter difficulties during data preprocessing, you can download dense point cloud data that has been preprocessed using Colmap. You may download them [through this link](https://drive.google.com/drive/folders/1VymLQAqzXtrd2CnWAFSJ0RTTnp25mLgA?usp=share_link). 

To use DepthAnythingV2, please create a directory and download the weights.
``` 
cd SREGS
mkdir checkp
https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true
Place the downloaded weights in the `checkp` directory.
```

## Training
LLFF datasets.
``` 
python train_llff.py  -s dataset/nerf_llff_data/trex -m output/trex --eval --n_views 3
```

MipNerf360 datasets
``` 
python train_360.py  -s dataset/mipnerf360/bicycle -m output/bicycle --eval --n_views 24
```
## Rendering

```
python render.py --source_path dataset/nerf_llff_data/trex  --model_path  output/trex  --render_depth
```
You can render a downloadable video file by adding the 'video' parameter.
```
python render.py --source_path dataset/nerf_llff_data/trex  --model_path  output/trex  --video  --fps 30
```

## Evaluation
You can just run the following script to evaluate the model.  

```
python metrics.py --source_path dataset/nerf_llff_data/trex  --model_path  output/trex
```

## About noise scale
For the llff dataset, the noise scale range can be adjusted according to the following parameters. Of course, this is only the optimal parameter for our test：

```
horns
noise_scale = get_dynamic_noise_scale(iteration, min_value=0.0002, max_value=0.0015, max_iter=8500)

trex
noise_scale = get_dynamic_noise_scale(iteration, min_value=0.001, max_value=0.02, max_iter=8500)

fern
noise_scale = get_dynamic_noise_scale(iteration, min_value=0.0001, max_value=0.0002, max_iter=8500)

room-s
noise_scale = get_dynamic_noise_scale(iteration, min_value=0.0008, max_value=0.001, max_iter=8500)

flower
noise_scale = get_dynamic_noise_scale(iteration, min_value=0.0008, max_value=0.001, max_iter=8500)

leaves
noise_scale = get_dynamic_noise_scale(iteration, min_value=0.0008, max_value=0.0012, max_iter=8500)

orchids
noise_scale = get_dynamic_noise_scale(iteration, min_value=0.8, max_value=1, max_iter=8500)

```




## Acknowledgement

Our method benefits from these excellent works.

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [SparseNeRF](https://github.com/Wanggcong/SparseNeRF)
- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)
- [3dgs-mcmc](https://github.com/ubc-vision/3dgs-mcmc)
- [Depth-AnythingV2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file)
- [DNGaussian](https://github.com/Fictionarry/DNGaussian)
