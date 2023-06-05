# Multiview Robustness

In this repository, we provide the code for the project "Multiview Robustness" as the final project for the EPFL's CS-503 Visual Intelligece course by Prof. Amir Zamir in Spring 2023.

## File Hierarchy
```bash
|-- multiview_robustness
    |-- run_baseline.py                 # training script for the multiview model
    |-- run_multiview.py                # training script for the baseline model
    |-- cfgs
    |   |-- baseline.yaml               # baseline model configuration file
    |   |-- multiview.yaml              # multiview model configuration file
    |-- results                         # empty folder for saving the results
    |-- scripts
    |   |-- attention_plots.py          # script to plot the attention plots
    |   |-- test_baseline_omnidata.py   # script to test the baseline model
    |   |-- test_multiview_omnidata.py  # script to plot the multiview model
    |-- src
    |   |-- checkpoint.py               # functions to save/load model checkpoints
    |   |-- utils.py                    # extra helper functions
    |   |-- dataloaders
    |   |   |-- rgb_depth_dataset.py    # default dataloader for the training scripts
    |   |-- losses
    |   |-- models
    |       |-- multiview_model.py      # multiview model implementation
```

## Installing Packages
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Dataset
To download the train dataset follow the instructions [here](https://github.com/EPFL-VILAB/omnidata#dataset) and run the following command:
```bash
omnitools.download point_info rgb depth_zbuffer  \    
    --components replica_gso \
    --subset fullplus \
    --dest ./omnidata_starter_dataset/ \
    --name YOUR_NAME --email YOUR_EMAIL --agree_all
```
and divide it into train, validation and test folders. Each of these folders should contain 3 folders named `point_info`, `rgb` and `depth_zbuffer`.


## Running the experiments
Before running the experiments, in the `yaml` files in `cfgs` folder specify the path to train, validation and test dataset. To train the baseline DPT model run the command:
```bash
python run_baseline.py --config  cfgs/baseline.yaml
```

And to train the multiview DPT model run the following:
```bash
python run_multiview.py --config  cfgs/multiview.yaml
```

The weights and sample outputs will be saved in the `output_dir` specified in the configuratoin. To get the description of the arguments in the configuration files run the following command:
```bash
python run_multiview.py --help
```

## Inference

Similar to the training scripts to test the baseline run the command:
```bash
python scripts/test_baseline_omnidata.py --config  cfgs/baseline.yaml
```

And to test the multiview model run:
```bash
python scripts/test_multiview_omnidata.py --config  cfgs/multiview.yaml
```

To get the attention ratio figures for the multiview model run this command:
```bash
python scripts/attention_plots.py --config  cfgs/multiview.yaml
```


## Group Members
Alphabetical Order:
- [Ali Garjani](mailto:ali.garjani@epfl.ch)
- [Amin Asadi Sarijalou](mailto:amin.asadisarijalou@epfl.ch)
- [Vishal Pani](mailto:vishal.pani@epfl.ch)