# OSIRIS: Optimized Sparse Inference and Reconstruction for Interdependent Spaces

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()

Official repository for the paper:

**A Unified Generative and Discriminative Model for Interpreting Multimodal and Incomplete Datasets**  
*Submitted to NeurIPS 2025 (under review)*

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Output](#output)
- [Citation](#citation)
- [License](#license)

---

## Overview

**OSIRIS** is a unified Bayesian framework that bridges generative and discriminative paradigms to analyze multimodal and incomplete datasets. It is designed to:

- Efficiently integrate multimodal data with a task-specific formulation.
- Learn a generative latent space for semi-supervised learning and missing value imputation, without introducing task bias.
- Mitigate overfitting, especially in low-sample-size scenarios, via a Bayesian formulation.
- Promote interpretability and compactness using sparsity-inducing priors on latent variables and features.

---

## Model Architecture

```
                               +----------------------------+
          ------------------>  | Disciminative Latent Space | ---> Task-specific Prediction
          |                    +----------------------------+
          |
+-----------------------+
| Multimodal Inputs     |
| (e.g., images, genes) |
+-----------------------+
          |                  
          |                    +--------------------------+
          ------------------>  | Generative Latent Space  |   ---> Missing Value Imputation
                               +--------------------------+
```

The model combines sparse discriminative and generative components into a unified inference pipeline.

---

## Getting Started

### Requirements

- Python 3.8+
- See `requirements.txt` for dependencies.

### Usage

To train the model on a specific dataset, run:

```bash
python main_{dataset}.py --fold {test fold} --seed {random seed} --model {model to run}
```

where, if you wish to reproduce the paper results, the model arguments can take the following values:
- fold: from 0 to 9 (10 folds).
- seed: from 0 to 9 (10 seeds).
- model: 'MVFA', 'MVPLS', 'MKL', 'CPM', 'CSMVIB', 'TMC', 'DeepIMV', 'OSIRIS'. These models include baselines and OSIRIS. Note that, the baseline models have either been cross-validated or use the default parameters provided by their original authors.

This will:

- Train the indicated model using the specified parameters and dataset.
- Internally create train/validation/test splits used in the original experiments.
- Save soft predictions and true labels of the test set to the `./Results/{dataset}/` directory. The name of the saved files will be: `{model}_{seed}_{fold}.npy`

To store the results for each dataset, create a dedicated directory following the structure: `./Results/{dataset}/`

Refer to the script headers for argument documentation.

Once all models have been executed, any evaluation metric can be computed using the soft predictions and the corresponding true labels. In the case of OSIRIS, the Evidence Lower Bound (ELBO) associated with each initialization has also been stored, enabling the selection of the model instance that achieves the highest ELBO. Thus, the final results will be: [soft predictions, true labels, ELBO].

---

## Datasets

The following publicly available multimodal datasets were used:

| Dataset    | Classes | Samples | Modalities (dim)        | Description                        |
|------------|---------|---------|--------------------------|------------------------------------|
| Arrhythmia [[1]](#1) | 2       | 452     | (15; 264)                | ECG features                       |
| Digit [[2]](#2)      | 10      | 2000    | (216; 76; 64; 6; 240; 47)| Image features                     |
| Sat [[3]](#3)        | 6       | 6435    | (4; 4)                   | Image features                     |
| LFWA [[4]](#4)       | 7       | 1277    | (2400; 73)               | Flattened images + discrete feats  |
| Fashion [[5]](#5)    | 4       | 2725    | (512; 512)               | CLIP embeddings                    |
| Tadpole [[6]](#6)    | 3       | 1296    | (5; 4; 5; 3; 7; 7)       | General medical data               |
| X-Ray [[7]](#7)      | 10      | 800     | (273; 112)               | Image features                     |
| Scene15 [[8]](#8)    | 15      | 4485    | (8; 90; 10)              | Image features                     |
| ADNI [[6]](#6)       | 2       | 1082    | (114; 9; 13; 152; 4)     | Brain regions                      |

> Note: Datasets are **not included** due to licensing.  
> All except ADNI and TADPOLE are publicly accessible.

Upon downloading each dataset, make sure to convert every view into a NumPy array before passing it to the model. Once the data has been downloaded and converted to NumPy format, save each view in the folder named after the specific dataset inside the Data/ directory: Data/{dataset}. Please make sure to check the code to see how each view has been named, so that your filenames match the expected format.

The specific train and test splits for the 10-fold cross-validation performed on each dataset are stored in the folder `Data/{dataset}/10fold_{dataset}`. The model automatically accesses these files to apply the correct partitioning during training and evaluation.

---

## Output

After training, the following outputs will be available:

- **Test metric**: Balanced accuracy printed to console.
- **Saved predictions**: Stored in the `./Results/{dataset}/{model}_{seed}_{fold}.npy` directory.

---

## Missing data imputation experiment

To reproduce the missing data imputation experiment, we follow a procedure analogous to the performance experiments. Specifically, to train the model with a given missing data rate and a specified imputation algorithm, we execute the following command:
```bash
python main_{dataset}.py --fold {test fold} --seed {random seed} --perc {percentage of missing values} --imputer {imputation algorithm}
```
where, if you wish to reproduce the paper results, the model arguments can take the following values:
- fold: from 0 to 9 (10 folds).
- seed: from 0 to 9 (10 seeds).
- perc: 5,10,15,20,25, and 30.
- imputer: 'median', 'mean', 'MICE', 'KNN', 'VAE', 'RF', and 'None'. None corresponds to using OSIRIS's internal imputation mechanism.

This will:

- Train OSIRIS model with a specified percentage of missing values and a specific imputation algorithm.
- Save soft predictions and true labels of the test set to the `/Results_imputation/{dataset}/` directory. The name of the saved files will be: `{seed}_{fold}_{perc}_{imputer}.npy`

To store the results for each dataset, create a dedicated directory following the structure: `./Results_imputation/{dataset}/`

# Output

After training, the following outputs will be available:

- **Test metric**: Balanced accuracy printed to console.
- **Saved predictions**: Stored in the `./Results/Imputation/{dataset}/OSIRIS_{seed}_{fold}_{perc}_{imputer}.npy` directory. Each file will follow the format: [Predicted labels, True labels, ELBO]


---

## Citation

Please cite the paper using the placeholder below. This will be updated upon publication:

```bibtex
@inproceedings{osiris2025,
  title={A Unified Generative and Discriminative Model for Interpreting Multimodal and Incomplete Datasets},
  author={Anonymous},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---

## License

This project is licensed under an open-source license (TBD upon publication).

## References

<a id="1">[1]</a> H. Guvenir, B. Acar, H. Muderrisoglu, and R. Quinlan. *Arrhythmia*. UCI Machine Learning Repository, 1997.

<a id="2">[2]</a> M. van Breukelen, R. P. W. Duin, D. M. J. Tax, and J. E. Den Hartog. *Handwritten digit recognition by combined classifiers*. Kybernetika, 34(4):381–386, 1998.

<a id="3">[3]</a> A. Ray et al. *SAT: Dynamic spatial aptitude training for multimodal language models*, 2025.

<a id="4">[4]</a> L. Wolf, T. Hassner, and Y. Taigman. *Effective unconstrained face recognition by combining multiple descriptors and learned background statistics*. IEEE TPAMI, 33(10):1978–1990, 2010.

<a id="5">[5]</a> N. Rostamzadeh et al. *Fashion-Gen: The generative fashion dataset and challenge*. arXiv:1806.08317, 2018.

<a id="6">[6]</a> ADNI (Alzheimer’s Disease Neuroimaging Initiative). [https://adni.loni.usc.edu](https://adni.loni.usc.edu)

<a id="7">[7]</a> J. Liu, J. Lian, and Y. Yu. *ChestX-Det10: Chest X-ray dataset on detection of thoracic abnormalities*, 2020.

<a id="8">[8]</a> S. Lazebnik, C. Schmid, and J. Ponce. *Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories*. CVPR, 2006.

