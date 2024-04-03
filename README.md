# <div align="center">**CoBEVT OPV2V Track**</div>
This repository contains the source code and data for our TempCoBEV. The whole pipeline is based on [OpenCOOD(ICRA2022)](https://github.com/DerrickXuNu/OpenCOOD)

## <div align="center">**Data Preparation**</div>
1. Download OPV2V origin data and structure it as required. See [OpenCOOD data tutorial](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html) for more detailed insructions.
2. After organize the data folders, download the `additional.zip` from [this url](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu?usp=sharing). This file contains BEV semantic segmentation labels that origin OPV2V data does not include.
3. The `additional` folder has the same structure of original OPV2V dataset. So unzip `additional.zip` and merge them with original opv2v data.
4. Remove scenario `opv2v/train/2021_09_09_13_20_58`, as this scenario has some bug for camera data.
## <div align="center">**Installation**</div>

```bash
# Clone repo
git clone https://github.com/cvims/TempCoBEV.git

cd TempCoBEV

# Setup conda environment
conda create -y --name tempcobev python=3.7

conda activate tempcobev
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch

# Install dependencies

pip install -r requirements.txt
mim install mmcv==2.0.1

python opencood/utils/setup.py build_ext --inplace
python setup.py develop
```


## <div align="center">**Setup**</div>
Please refer to the CoBEVT repository to set up the dataset structure and download the pretrained model weights of [CoBEVT](https://github.com/DerrickXuNu/CoBEVT/tree/main/opv2v).
After downloading the model weights adapt the file paths in python file
```
opencood/tools/create_model_embeddings_dataset.py
```
to generate the embeddings of the models, for example CoBEVT, update the file paths to point to the pretrained model and run the scripts.
This script creates pickle files that contain information about the embeddings etc. to speed up the training of the temporal module.




## <div align="center">**Training**</div>
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint **on a single gpu**, run the following command:
```python
python opencood/tools/training_temporal/train_temporal.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
Arguments explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/temporal_opcamera_embeddings/cobevt/basic_recurrent_iou_loss.yaml`.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.



## <div align="center">**Evaluation**</div>
To evaluate the temporal model, run the following command:
```python
python opencood/tools/evaluation_temporal/cobevt_embeddings/temporal.py --temporal_model_path ${CHECKPOINT_FOLDER}
```
Arguments explanation:
- `temporal_model_path`: the path of the checkpoints. In this path you must add a 'config_eval.yaml' with adjusted paths to the test folder of the data. 


To evaluate the temporal model on future frames, run the following command:
```python
python opencood/tools/evaluation_temporal/cobevt_embeddings/future_frames.py --temporal_model_path ${CHECKPOINT_FOLDER} --future_predictions ${INTEGER VALUE}
```
Arguments explanation:
- `temporal_model_path`: the path of the checkpoints. In this path you must add a 'config_eval.yaml' with adjusted paths to the test folder of the data. 
- `future_predictions`: the amount of future predictions. For all future predictions the model uses only the ego perspective and tries to recover as much information as possible from previous frames.
