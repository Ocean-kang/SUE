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

### Using the Trainer Directly
If you want to bypass the command-line interface or apply SUE to your own data pipeline, you can interact with the `Trainer` class directly.

Before training, Ensuring your training data is properly formatted as weakly paired. You can achieve this using the create_weakly_parallel_data function.

```python
from data import load_dataset, create_weakly_parallel_data
from trainer import Trainer

# 1. Load your configuration and dataset
train_set, test_set = load_dataset("your_dataset_name", n_test=400)

# 2. Make your data weakly paired (Crucial step for SUE)
train_set = create_weakly_parallel_data(train_set, n_parallel=100)

# 3. Initialize the Trainer
trainer = Trainer(
    dataset_name="your_dataset_name",
    n_parallel=100, 
    n_components=30, # Adjust based on your config
    configs=your_config_dict
)

# 4. Fit the model using the weakly paired data
trainer.fit(
    train_set=train_set, 
)

# 5. Test the model
trainer.test(test_set=test_set)
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
