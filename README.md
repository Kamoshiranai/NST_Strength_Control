# Neural Style Transfer for Data Encoding Using Strength Control

This repository contains code for the paper [Neural Style Transfer for Data Encoding Using Strength Control](TODO: add link). We propose downsampling and masking techniques to achieve spatial control of style strength while preserving existing color encodings via luminance-only style transfer. This method can be used to encode an additional scalar field as style strength.

## Setup

### 1. Clone the repository

```bash
git clone github.com/Kamoshiranai/NST_Strength_Control
cd NST_Strength_Control
```

### 2. Create the conda environment

```bash
conda env create --file=environment.yml
conda activate nst_strength_control
```

## Usage

### Gatys NST (`GatysNST/`)

Optimization-based style transfer using a VGG19 backbone. Includes our proposed downsampling and masking strategies for spatial style strength control, optional luminance-only style transfer and an optional Mean-Std loss in addition to the original Gram loss.

```bash
cd GatysNST
jupyter notebook NeuralStyleTransfer.ipynb
```

### Deep Feature Synthesis (`DeepFeatureSynthesis/`)

Optimization-based style transfer method that uses a neural-neighbor loss and techniques from texture synthesis.Includes our proposed downsampling and masking strategies for spatial style strength control and optional luminance-only style transfer.

```bash
cd DeepFeatureSynthesis
jupyter notebook deepfeaturesynthesis.ipynb
```

### SANet (`SANet/`)

Feed-forward arbitrary style transfer. Includes our proposed downsampling and masking strategies for spatial style strength control and an optional luminance-only style transfer.

```bash
cd SANet
python Eval.py \
    --content ../images/content/<content_image> \
    --style ../images/styles/<style_image> \
    --scalar_field ../images/scalar_fields/<scalar_field_image>
```

### URST (`URST/`)

Ultra-high-resolution style transfer. Includes our proposed downsampling and masking strategies for spatial style strength control and an optional luminance-only style transfer.

```bash
cd URST/Li2018Learning
python test.py \
    --URST \
    --content ../images/content/<content_image> \
    --style ../images/styles/<style_image> \
    --scalar_field ../images/scalar_fields/<scalar_field_image>
```

### Create example scalar fields using `create_scalar_fields.py`

This creates gradient and binary scalar field images for the given content image.

```bash
python create_scalar_fields.py \
    --content ../images/content/<content_image>
```

> Check the `--help` output of each script for additional arguments controlling the downsampling strategy, color remapping, content and style size, etc.

## Repository Structure and Third-Party Code

The code in the following directories except `DeepFeatureSynthesis/` is based on existing open-source projects.

| Path | Based on | License |
|---|---|---|
| `GatysNST/` | [PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer) (Leon Gatys, 2021) | MIT |
| `SANet/` | [SANET](https://github.com/GlebSBrykin/SANET) (Глеб Брыкин, 2019) | MIT |
| `URST/` | [URST](https://github.com/czczup/URST) (Zhe Chen) | Apache 2.0 |
| `Li2018Learning/` (in `URST/`) | [LinearStyleTransfer](https://github.com/sunshineatnoon/LinearStyleTransfer) (SunshineAtNoon, 2018) | BSD 2-Clause |
| `DeepFeatureSynthesis/` | Our implementation of [Neural style transfer based on deep feature synthesis](https://link.springer.com/article/10.1007/s00371-022-02664-2) (Dajin Li, Wenran Gao) | — | 

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full attribution details and the `LICENSES/` directory for the complete license texts of each third-party project.

## Acknowledgments

Thanks to the authors of PytorchNeuralStyleTransfer, SANET, URST, and LinearStyleTransfer for releasing their code.

## Citation
#TODO

```
@inproceedings{NST_for_data_encoding_using_strength_control
booktitle = {Vision, Modeling, and Visualization},
title = {{Neural Style Transfer for Data Encoding Using Strength Control}},
author = {Merk Niklas and Sterzik Anna and Lawonn Kai},
year = {2026},
publisher = {The Eurographics Association},
}
```