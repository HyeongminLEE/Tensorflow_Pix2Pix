# Pix2Pix in Pytorch

Study Friendly Implementation of Pix2Pix in Tensorflow

More Information: [Original Paper](https://arxiv.org/pdf/1611.07004v1.pdf)

Identical Pytorch Implemenation will be uploaded on [taeoh-kim's Github](https://github.com/taeoh-kim/Pytorch_Pix2Pix)

- GAN: [[Pytorch]][[Tensorflow]]
- DCGAN: [[Pytorch]][[Tensorflow]]
- InfoGAN: [[Pytorch]][Tensorflow]
- Pix2Pix: [[Pytorch]][Tensorflow]
- DiscoGAN: [[Pytorch]][Tensorflow]

## 1. Environments

- Windows 10
- Python 3.5.3 (Anaconda)
- Tensorflow 1.0.0
- Numpy 1.13.1

## 2. Code Description

- `train.py`: Main Code
- `test.py`: Test Code after Training
- `model.py`: Generator and Discriminator
- `dbread.py`: My Own Code for Reading Database

## 3. Networks and Parameters

### 3.1 Hyper-Parameters

- Image Size = 256x256 (Resized)
- Batch Size = 1 or 4
- Learning Rate = 0.0002
- Adam_beta1 = 0.5
- Lambda_A = 100 (Weight of L1-Loss)

Detail Recommandations for Each Dataset are on the Last Page of [Original Paper](https://arxiv.org/pdf/1611.07004v1.pdf)


### 3.2 Generator Networks (network.py)

<p align="center"><img width="100%" src="images/generator.PNG" /></p>

### 3.3 Discriminator Networks (network.py)

<p align="center"><img width="100%" src="images/discriminator.PNG" /></p>

## 4. Database

### 4.1 DB Download

This Table is from [this github link](https://github.com/affinelayer/pix2pix-tensorflow)
| dataset | example |
| --- | --- |
| `python tools/download-dataset.py facades` <br> 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/). (31MB) <br> Pre-trained: [BtoA](https://mega.nz/#!2xpyQBoK!GVtkZN7lqY4aaZltMFdZsPNVE6bUsWyiVUN6RwJtIxQ)  | <img src="docs/facades.jpg" width="256px"/> |
| `python tools/download-dataset.py cityscapes` <br> 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/). (113M) <br> Pre-trained: [AtoB](https://mega.nz/#!rxByxK6S!W9ZBUqgdGTFDWVlOE_ljVt1G3bU89bdu_nS9Bi1ujiA) [BtoA](https://mega.nz/#!b1olDbhL!mxsYC5AF_WH64CXoukN0KB-nw15kLQ0Etii-F-HDTps) | <img src="docs/cityscapes.jpg" width="256px"/> |
| `python tools/download-dataset.py maps` <br> 1096 training images scraped from Google Maps (246M) <br> Pre-trained: [AtoB](https://mega.nz/#!i8pkkBJT!3NKLar9sUr-Vh_vNVQF-xwK9-D9iCqaCmj1T27xRf4w) [BtoA](https://mega.nz/#!r8xwCBCD!lNBrY_2QO6pyUJziGj7ikPheUL_yXA8xGXFlM3GPL3c) | <img src="docs/maps.jpg" width="256px"/> |
| `python tools/download-dataset.py edges2shoes` <br> 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. (2.2GB) <br> Pre-trained: [AtoB](https://mega.nz/#!OoYT3QiQ!8y3zLESvhOyeA6UsjEbcJphi3_uEt534waSL5_f_D4Y) | <img src="docs/edges2shoes.jpg" width="256px"/>  |
| `python tools/download-dataset.py edges2handbags` <br> 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. (8.6GB) <br> Pre-trained: [AtoB](https://mega.nz/#!KlpBHKrZ!iJ3x6xzgk0wnJkPiAf0UxPzhYSmpC3kKH1DY5n_dd0M) | <img src="docs/edges2handbags.jpg" width="256px"/> |

### 4.2 DB Setting
- You just have to get a text file that contains all directories of your image data(filelist.txt).
- You can make filelist with following command.

```bash
cd <database_dir>
dir /b /s > filelist.txt
```
- Then you must open 'filelist.txt' and delete line 'your_db_dir/filelist.txt'.


## 5. Train

```bash
python train.py --train filelist.txt
```

- `--train`: Directory of the Text File of Train Filelist 
- `--out_dir`: Directory to Save your Train Result
- `--epochs`: Num of Epochs You Want
- `--batch_size`: Batch Size You Want
- `--direction`: 'AtoB' or 'BtoA'

After finish training, saved models are in the `./output/checkpoint` directory and the Train Results are saved in `./output/result`.(default)

## 6. Test

```bash
python test.py --train filelist.txt
```

- `--test`: Directory of the Text File of Test Filelist
- `--out_dir`: Directory to Save your Train Result
- `--ckpt_dir`: Directory of Trained Model
- `--visnum`: Number of Visualization in an Image File
- `--direction`: 'AtoB' or 'BtoA'


Test results will be saved in `./output_test`(default)


## 7. Results

### [Input | Generated | Ground Truth]

### Edges to Shoes (8 Epochs)
15 Epochs (which is in the Paper) will give better results

<p align="center"><img width="100%" src="results/e2s1.png" /></p>
<p align="center"><img width="100%" src="results/e2s2.png" /></p>


### Maps to Aerials (200 Epochs)

<p align="center"><img width="100%" src="results/m2a1.png" /></p>
<p align="center"><img width="100%" src="results/m2a2.png" /></p>
<p align="center"><img width="100%" src="results/m2a3.png" /></p>
<p align="center"><img width="100%" src="results/m2a4.png" /></p>
<p align="center"><img width="100%" src="results/m2a5.png" /></p>
<p align="center"><img width="100%" src="results/m2a6.png" /></p>
<p align="center"><img width="100%" src="results/m2a7.png" /></p>
<p align="center"><img width="100%" src="results/m2a8.png" /></p>

### Architectural labels to Photo (200 Epochs)

<p align="center"><img width="100%" src="results/a2p1.png" /></p>
<p align="center"><img width="100%" src="results/a2p2.png" /></p>
<p align="center"><img width="100%" src="results/a2p3.png" /></p>
<p align="center"><img width="100%" src="results/a2p4.png" /></p>
<p align="center"><img width="100%" src="results/a2p5.png" /></p>
<p align="center"><img width="100%" src="results/a2p6.png" /></p>
<p align="center"><img width="100%" src="results/a2p7.png" /></p>
<p align="center"><img width="100%" src="results/a2p8.png" /></p>



























