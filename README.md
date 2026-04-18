
# **CIS582 Project**
# **Alex Conklin**
# ** Applying Poisonspot to the HaGRID Dataset**
---

This repository contains modifications to the original PoisonSpot code https://github.com/um-dsp/PoisonSpot in order to facilitate experiments using the HaGRID 384p Sample Dataset https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p



## Overview of the folder structure
```
PoisonSpot/
├── configs/                   # **Updated** Added a new configuration that runs a narcissus attack using the HaGRID dataset with poison rate 10% and suspected poison rate 50%
├── src/                       # Source code
│   ├── attacks/               # Attack implementations
│   │   ├── HiddenTriggerBackdoor/
│   │   ├── Labelconsistent/
│   │   ├── mixed/
│   │   ├── Narcissus/         # **Updated** Added a new function to perform a narcissus attack on the HaGRID Dataset
│   │   └── Sleeperagent/
│   ├── data/                  # **Updated** Updated folder structure to store transformed HaGRID data while running experiments to speed up successive attempts.
│   ├── helpers/               
│   │   ├── data.py            # Data loading function
│   │   ├── provenance.py      # **Updated** Commented out a single line that was forcing images to 32x32 and breaking stacks with the larger HaGRID images
│   │   ├── scoring.py         # Score samples for poisoning 
│   │   └── train.py           # model training funciton 
│   ├── models/                # Model definitions (e.g., ResNet)
│   ├── results/               # **Updated** Added results from last HaGRID run
│   ├── saved_models/          # Model checkpoints for clean training, poisoned training, and retraining. 
│   ├── temp_folder/           # Temporary folder to save model during provenance capture
│   └── Training_Prov_Data/    # Stores captured provenance data (batch-level & sample-level) 
├── main.py                    # **Updated** Added additional branches to the code to support the new HaGRID experiment path
├── README.md                  # **Updated** Modified this to discuss whats new, as you can tell since you're reading this!
└── requirements.txt           # **Updated** Removed some libraries that weren't being used and made some minor updates to be compatible with Colab
```

## New Configuration Files
This section is small since there's only one- a narcissus attack utilizing the HaGRID dataset. Feel free to modify its parameters to run other experiments.

### Narcissus on HaGRID

| Filename                                   | Dataset   | Scenario       | pr_tgt (%) | pr_sus (%) | 
|--------------------------------------------|-----------|----------------|-----------:|-----------:|
| `configs/config_narcissus_hagrid_10_10_50.yaml`| CIFAR-10  | fine_tuning   |         10 |         50 |

## Hardware and Software Setup

### Hardware (tested on)
 
- **GPU:** Google Colab A100  

### Software (tested on)

- **Python:** 3.12
- **CUDA version:** 12.5

---
---
---

## Remainder is identical to the Original README



## Arguments

| Argument             | Description                                                    | Default Value                           |
|----------------------|----------------------------------------------------------------|-----------------------------------------|
| `batch_level`        | Capture batch-level provenance data                            | `True`                                  |
| `clean_training`     | Perform clean training by removing the suspected samples       | `False`                                 |
| `poisoned_training`  | Perform training using the suspected samples                   | `False`                                  |
| `sample_level`       | Capture sample-level provenance data                           | `True`                                  |
| `score_samples`      | Score suspected samples based on the sample-level provenance data | `True`                               |
| `retrain`            | Retrain the model by excluding predicted poisoned samples      | `True`                                  |
| `pr_sus`             | Percentage of poisoned data in the suspected set (%)           | `50`                                  |
| `ep_bl`              | Training epochs for batch-level provenance capture             | `1`                                   |
| `ep_bl_base`         | Epoch number to start batch-level provenance capture           | `200`                                  |
| `ep_sl`              | Training epochs for sample-level provenance capture            | `10`                                   |
| `ep_sl_base`         | Epoch number to start sample-level provenance capture          | `200`                                  |
| `pr_tgt`             | Percentage of poisoned data in the target set (%)              | `10`                                   |
| `bs_sl`              | Batch size for sample-level provenance capture                 | `128`                                  |
| `bs_bl`              | Batch size for batch-level provenance capture                  | `128`                                  |
| `bs`                 | Batch size for clean training, poisoned training, and retraining | `128`                               |
| `eps`                | Perturbation budget (`eps/255`)                                | `16`                                   |
| `vis`                | Pixel value for Label consistent attack                                                  | `255`                                 |
| `target_class`       | Target class for the attack                                    | `2`                                    |
| `source_class`       | Source class for the attack (mainly relevant for `sa`)                                    | `0`                                    |
| `dataset`            | Dataset to use for the experiment. (`CIFAR10`, `slt10`,`imagenet`, )                              | `CIFAR10`                            |
| `attack`             | Attack to use for the experiment, (`lc`,`narcissus`,`sa`,`ht`)                              | `lc`                                 |
| `model`              | Model to use for the experiment, (`ResNet18`,`CustomCNN`,`BasicResNet`,`CustomResNet18`,`ViT`)                                | `ResNet18`                           |
| `dataset_dir`        | Root directory for the datasets                                | `./data/`                            |
| `clean_model_path`   | Path to the trained clean model used for fine-tuning           | `./saved_models/resnet18_200_clean.pth` |
| `saved_models_path`  | Path to save the trained models                                | `./saved_models/`                    |
| `global_seed`        | Global seed for the experiment                                 | `545`                                  |
| `gpu_id`             | GPU device ID to use for the experiment                        | `0`                                    |
| `lr`                 | Learning rate for the experiment                               | `0.1`                                  |
| `results_path`       | Path to save the figures                                       | `./results/`                         |
| `prov_path`          | Path to save the provenance data                               | `./Training_Prov_Data/`              |
| `epochs`             | Number of epochs for clean training, poisoned training, and retraining | `200`                              |
| `scenario`           | Scenario to use for the experiment (`fine_tune`,`from_scratch`) | `from_scratch`                   |
| `get_result`         | Get results from previous runs                                 | `False`                                |
| `force`              | Force the run overwriting previous model checkpoints           | `False`                                |
| `custom_threshold`   | Custom threshold for scoring suspected samples                 | `0.5`                                  |
| `Threshold_type`     | Kmeans, Gaussian, or custom threshold choice                   | `Kmeans`                               |
| `k_1`                | First phase threshold                                          | `1`                                    |
| `k_2`                | Second phase threshold                                         | `0.0001`                               |
| `sample_from_test`   | Sample from the test set                                       | `False`                                |
| `cv_model`           | Model to use for cross-validation (`RandomForest`, `LogisticRegression`, `LinearSVM`, `KernelSVM` , `MLP`)                              | `RandomForest`                       |
| `groups`             | Number of groups to use for cross-validation                   | `5`                                    |
| `opt`                | Optimizer to use for the experiment (`sgd`, `adam`)            | `sgd`                                |
| `random`             | Random trigger for Sleeper-Agent attack (`True`,`False`)    | `False`                                |
| `training_mode`      | Training mode for the experiment (`true`, `false`)           | `true`                                 |


---


## Installation

1. **Obtain the code:**
   - **From GitHub:**  
     ```bash
     git clone https://github.com/Philenku/PoisonSpot.git
     ```
   - **From Zenodo:**  
     ```bash
     unzip PoisonSpot-v1.0.3.zip   # Download the zip file from zenodo and adjust filename as needed. 
     ```

2. **(Optional) Create & activate a virtual environment:**
   ```bash
   python -m venv venv
   ```
   ```bash
   source venv/bin/activate  # linux 
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Enter the PoisonSpot directory:**
   ```bash
   cd PoisonSpot # or go to the desired folder name
   ```



# Usage Instructions

1. **Prepare your config file**  
   Pick an example configuration file (YAML) from `configs/`, then edit the arguments such as`pr_tgt` and `pr_sus` to suit your experiment:
   For example, to run a full pipeline Label-Consistent attack on CIFAR-10 from scratch, you can use the following configuration file:

   ```bash
   configs/config_lc_cifar_10.yaml
   ```

2. **Run the full pipeline**
   You can run the full pipeline using the `main.py` script. The script will execute all steps of PoisonSpot based on the provided configuration file.

   ```bash
   python3 main.py -c configs/config_lc_cifar_10.yaml
   ```
3. **Alternative Usage method** 
   To run several configuration files in one go, use the shell script in the scripts/ directory. List the configs you want inside the script, make it executable, and then run it:

   ```bash
   chmod +x scripts/run.sh
   ```
   ```bash 
   ./scripts/run.sh
   ```

---

## Note
- The `pr_tgt` argument specifies the percentage of poisoned samples in the target set. So, to be consistent with our paper, in which we use the percentage of poisoned samples in the whole dataset, you can divide pr_tgt by the number of classes (10). For example, if you set `pr_tgt=10`, it means you are running an experiment with a 1% poisoned sample percentage in the training set.

- The `data` folder stores downloaded poisoned datasets for the label-consistent, sleeper-agent, and hiddentrigger backdoors which saves time by avoiding the need to regenerate the poisoned datasets every time you run an experiment.
- The `saved_models` directory also contains saved poisoned models for the configs listed, so you can set `poisoned_training = False`. If you want to discard the saved checkpoint, you need to revert it to `True` to train from scratch. 
- If you encounter a high number of important features during `batch_level` training, the provenance data may be too large for your CPU, so you can increase the epochs for the batch-level capture (`ep_bl`), raise the first-phase threshold `k`, or set `poisoned_training = True` to train the model correctly before capturing the provenance.




### **Results**
- The output results are saved in `results` folder 
   ```
   src/results/experiment_<attack>_<dataset>_<pr_tgt>_<pr_sus>/results.csv
   ```
You can also specify a custom experiment name via the `exp` field in your config.

The CSV contains the following columns:

| Column                      | Description                                                      |
|-----------------------------|------------------------------------------------------------------|
| `epochs`                    | Number of epochs  for the specific step                          |
| `Clean training ACC`        | Accuracy on clean test set after clean training                  |
| `Poisoned training ACC`     | Accuracy on poisoned test set after poisoned training            |
| `Poisoned training ASR`     | Attack success rate on poisoned test set                         |
| `Batch-level \|features\|`  | Number of important features from batch level provenance capture |
| `TPR KMeans`                | True positive rate using K-means threshold   after scoring samples |
| `FPR KMeans`                | False positive rate using K-means threshold                      |
| `TPR Gaussian`              | True positive rate using Gaussian threshold              |
| `FPR Gaussian`              | False positive rate using Gaussian hreshold             |
| `Retrain ACC`               | Accuracy on clean test set after retraining    |
| `Retrain ASR`               | Attack success rate after retraining           |

- The results in the paper are mostly based on the `KMeans` threshold, but you can also use the `Gaussian` threshold for comparison if it brings a better result.
- The provenance data collected during training is saved in the `Training_Prov_Data` folder.
- Visualizations of the poison score distribution from the experiments are saved in the experiment folder under `results/experiment_<attack>_<dataset>_<pr_tgt>_<pr_sus>/` as images.



## Citation

```bibtex
@inproceedings{hailemariam2025poisonspot,
  author    = {Philemon Hailemariam, Birhanu Eshete},
  title     = {PoisonSpot: Precise Spotting of Clean-Label Backdoors via Fine-Grained Training Provenance Tracking},
  booktitle = {Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security},
  year      = {2025}
}
```
---

