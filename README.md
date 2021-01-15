# pytorch-hdf5-example
Very Simple Data Loading Example using HDF5 (h5py) for Pytorch.

Tested on Ubuntu 18.04 with Python 3.8 and Pytorch 1.7

## 0. Prepare Environment
Clone this repository, and you may create a separate conda environment using the provided `environment.yml` file:

```bash
conda env create --file environment.yml
conda activate pytorch-ex
```

## 0. Prepare Dataset
In this tutorial, we use dataset from the paper [Ingraham et al. (2019)](#ingraham2019).
Please download the dataset first from [here](https://figshare.com/articles/dataset/Predicting_energy_cost_from_wearable_sensors_A_dataset_of_energetic_and_physiological_wearable_sensor_data_from_healthy_individuals_performing_multiple_physical_activities/7473191), and store at any location you want to.
You may download a subset (e.g., a single MAT file) for now.

## 1. Generate HDF5 Dataset File 
```bash
python 1_generate_h5.py
```
See `1_generate_h5.py` for details, e.g., path setup.

## 2. Build Index File 
```bash
python 2_build_index.py
```
See `2_build_index.py` for details, e.g., path setup.

## 3. Train a (stupid) Model 
```bash
python 3_train_model.py
```
See `3_train_model.py` for details, e.g., path setup, we use `argparse` for this file.



# References

<a name="ingraham2019"></a>[(Ingraham et al., 2019) Ingraham, Kimberly A., Daniel P. Ferris, and C. David Remy. "Evaluating physiological signal salience for estimating metabolic energy cost from wearable sensors." Journal of Applied Physiology 126.3 (2019): 717-729.](https://journals.physiology.org/doi/full/10.1152/japplphysiol.00714.2018)
