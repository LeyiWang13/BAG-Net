# BAG-Net implementation

<p align="left">
  <a href="https://ijcai.org/proceedings/2026/0193.pdf">
    <img src="https://img.shields.io/badge/PDF-BAG--Net-2563EB?style=flat-square"
         alt="Paper PDF" height="32">
  </a>
</p>

This repository provides the official PyTorch implementation of BAG-Net (Bidirectional Receptive-Field Graph Network), a deep learning framework focuses on pruning correspondences from an initial set of correspondences with a low inlier ratio.

If you find this project useful, please cite:

```bibtex
@inproceedings{wang2026bagnet,
  title={BAG-Net: Bidirectional Receptive-Field Graph Network for Two-View Correspondence Pruning},
  author={Wang, Leyi and Chen, Hao and Yang, Changcai},
  booktitle={Proceedings of IJCAI-ECAI},
  year={2026}
}
```

## Requirements

Please use Python 3.10, opencv-python (4.7.0.72), and PyTorch (>= 2.1.1+cu118). Other dependencies should be easily installed through pip or conda.

## Example scripts

### Run the demo

For a quick start, clone the repo and download the pretrained model.

```bash
git clone https://github.com/LeyiWang13/BAG-Net
```

### Generate training and testing data

First download the YFCC100M dataset.

```bash
bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8
tar -xvf raw_data_yfcc.tar.gz
```

Download SUN3D testing (1.1G) and training (31G) datasets if needed.

```bash
bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2
tar -xvf raw_sun3d_test.tar.gz

bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
tar -xvf raw_sun3d_train.tar.gz
```

### Train model on YFCC100M or SUN3D

After generating the dataset for YFCC100M, run the training script.

```bash
cd ./core
python main.py
```

You can change the default settings for network structure and training process in `./core/config.py`.

### Train with your own local features or data

The provided models are trained using SIFT. You had better retrain the model if you want to use BAG-Net with your own local features, such as RootSIFT, SuperPoint, etc.

## Acknowledgement

This code is borrowed from [OANet](https://github.com/zjhthu/OANet) and [CLNet](https://github.com/sailor-z/CLNet). If using the parts of code related to data generation, testing, and evaluation, please cite these papers.

```bibtex
@inproceedings{zhang2019learning,
  title={Learning two-view correspondences and geometry using order-aware network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5845--5854},
  year={2019}
}

@inproceedings{zhao2021progressive,
  title={Progressive correspondence pruning by consensus learning},
  author={Zhao, Chen and Ge, Yixiao and Zhu, Feng and Zhao, Rui and Li, Hongsheng and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6464--6473},
  year={2021}
}
```
