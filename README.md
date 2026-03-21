# Learning Shared Representations from Unpaired Data (SUE)

**[Project Page](https://shaham-lab.github.io/SUE_page/) | [arXiv](https://arxiv.org/abs/2505.21524)**

This is the official PyTorch implementation of SUE from the paper "Learning Shared Representations from Unpaired Data".

<p align="center">
    <img src="https://github.com/shaham-lab/SUE/blob/main/SUE.png" width="600">
</p>


## Installation
To run the project, clone this repo and then create a conda environment via:

```bash
conda env create -f environment.yml
```
Subsequently, activate this environment:

```bash
conda activate sue
```

## Running  
To run an example of the project on the retrieval task, follow these steps:  

1. **Download** the model checkpoints and data encodings from [here](https://drive.google.com/drive/folders/1RO4dlpCpOOE5gZbxP_DotVQ-GzbQYsBs?usp=sharing).  
2. **Unzip** the downloaded files.  
3. **Locate**:  
   - The model checkpoint file: `checkpoints_flickr30.pth` (inside the `checkpoints` folder).  
   - The data encodings: found under `data/flickr30`.
  
4. **Run** the following command: 
```bash
python retrieval.py --test flickr30
```
5. If you want to train the model from scratch, use the following command:
```bash
python retrieval.py --train flickr30
```

## Citation
If you find our work useful, please cite it:

```bibtex
@inproceedings{yacobi2025sue,
    title={Learning Shared Representations from Unpaired Data},
    author={Yacobi, Amitai and Ben-Ari, Nir and Talmon, Ronen and Shaham, Uri},
    journal={Advances in Neural Information Processing Systems},
    year={2025}
}
