# Diffusion Dictionary Learning

This project is based of **Label-Efficient Semantic Segmentation with Diffusion Models** repository code and data - [here](https://github.com/yandex-research/ddpm-segmentation). We apply SAE to img2img pixel-space Diffusion Model and find features/latents that overlap with GT segmentation masks of images for particular classes. We conduct experiments to find best timestep and train SAE on block=6 outputs. Data collection and processing is implemented in `collect_features.py` script, and SAE training, visualization and metric evaulation is done in `train-sae.ipynb` notebook. Feel free to explore. The notebook with our best-trained SAE is `train-sae-BEST-10scale+big3batch+moreEp+t150.ipynb`. You can find visualizations for every class there. 

In short, we find that feature maps of img2img DM can be decomposed, using the Sparse Autoencoder, into semantic features that align with GT semantic masks in terms of IoU. See our presentation for the results: [Link](https://docs.google.com/presentation/d/1gUjpomvHIki4aPuMgWJjJUIv6ydq96NVEWT2e4ch9a0/edit#slide=id.p)

Sections below are left from original codebase to better understand code structure and data.

&nbsp;
## Datasets

The evaluation is performed on 6 collected datasets with a few annotated images in the training set:
Bedroom-18, FFHQ-34, Cat-15, Horse-21, CelebA-19 and ADE-Bedroom-30. The number corresponds to the number of semantic classes.

[datasets.tar.gz](https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/datasets.tar.gz) (~47Mb)


&nbsp;
## DDPM

### Pretrained DDPMs

The models trained on LSUN are adopted from [guided-diffusion](https://github.com/openai/guided-diffusion).
FFHQ-256 is trained by ourselves using the same model parameters as for the LSUN models.

*LSUN-Bedroom:* [lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)\
*FFHQ-256:* [ffhq.pt](https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/models/ddpm_checkpoints/ffhq.pt) (Updated 3/8/2022)\
*LSUN-Cat:* [lsun_cat.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_cat.pt)\
*LSUN-Horse:* [lsun_horse.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_horse.pt)

### Run 

1. Download the datasets:\
 &nbsp;&nbsp;```bash datasets/download_datasets.sh```
2. Download the DDPM checkpoint:\
 &nbsp;&nbsp; ```bash checkpoints/ddpm/download_checkpoint.sh <checkpoint_name>```
3. Check paths in ```experiments/<dataset_name>/ddpm.json``` 
4. Run: ```bash scripts/ddpm/train_interpreter.sh <dataset_name>```
   
**Available checkpoint names:** lsun_bedroom, ffhq, lsun_cat, lsun_horse\
**Available dataset names:** bedroom_28, ffhq_34, cat_15, horse_21, celeba_19, ade_bedroom_30

**Note:** ```train_interpreter.sh``` is RAM consuming since it keeps all training pixel representations in memory. For ex, it requires ~210Gb for 50 training images of 256x256. (See [issue](https://github.com/nv-tlabs/datasetGAN_release/issues/34))

**Pretrained pixel classifiers** and test predictions are [here](https://www.dropbox.com/s/kap229jvmhfwh7i/pixel_classifiers.tar?dl=0).

### How to improve the performance

* Tune for a particular task what diffusion steps and UNet blocks to use.


&nbsp;
## DatasetDDPM


### Synthetic datasets

To download DDPM-produced synthetic datasets (50000 samples, ~7Gb) (updated 3/8/2022):\
```bash synthetic-datasets/ddpm/download_synthetic_dataset.sh <dataset_name>```

### Run | Option #1

1. Download the synthetic dataset:\
&nbsp;&nbsp; ```bash synthetic-datasets/ddpm/download_synthetic_dataset.sh <dataset_name>```
2. Check paths in ```experiments/<dataset_name>/datasetDDPM.json``` 
3. Run: ```bash scripts/datasetDDPM/train_deeplab.sh <dataset_name>``` 

### Run | Option #2

1. Download the datasets:\
 &nbsp;&nbsp; ```bash datasets/download_datasets.sh```
2. Download the DDPM checkpoint:\
 &nbsp;&nbsp; ```bash checkpoints/ddpm/download_checkpoint.sh <checkpoint_name>```
3. Check paths in ```experiments/<dataset_name>/datasetDDPM.json```
4. Train an interpreter on a few DDPM-produced annotated samples:\
   &nbsp;&nbsp; ```bash scripts/datasetDDPM/train_interpreter.sh <dataset_name>```
5. Generate a synthetic dataset:\
   &nbsp;&nbsp; ```bash scripts/datasetDDPM/generate_dataset.sh <dataset_name>```\
   &nbsp;&nbsp;&nbsp; Please specify the hyperparameters in this script for the available resources.\
   &nbsp;&nbsp;&nbsp; On 8xA100 80Gb, it takes about 12 hours to generate 10000 samples.   

5. Run: ```bash scripts/datasetDDPM/train_deeplab.sh <dataset_name>```\
   &nbsp;&nbsp; One needs to specify the path to the generated data. See comments in the script.

**Available checkpoint names:** lsun_bedroom, ffhq, lsun_cat, lsun_horse\
**Available dataset names:** bedroom_28, ffhq_34, cat_15, horse_21
