# MIDI-FiF-RoAR
This is the repo for MIDI-FiF-RoAR. The official code release of the paper “**Fine-grained Position Helps Memorizing More, A Novel Music Compound Transformer Model with Feature Interaction Fusion**”.

## Citing MIDI-FiF-RoAR

```
@article{li2023midififroar,
  title={Fine-grained Position Helps Memorizing More, A Novel Music Compound Transformer Model with Feature Interaction Fusion},
  author={Li, Zuchao and Gong, Ruhang and Chen, Yineng and Su, Kehua},
  journal={AAAI},
  year={2023}
}
```


## Acknowledgement

Part of our codes are adapted from [ScienceQA](https://github.com/wazenmai/MIDI-BERT) and [Transformers](https://github.com/huggingface/transformers).

We thank Yi-Hui (Sophia) Chou, I-Chun (Bronwin) Chen for providing the codebase for baselines.


## Installation
* Python3
* Install generally used packages for MIDI-FiF-RoAR:
```python
git clone https://github.com/zcli-charlie/MIDI-FiF-RoAR.git
cd MIDI-FiF-RoAR
pip install -r requirements.txt
```


## A. Prepare Data

All data in CP/REMI token are stored in ```data/CP``` & ```data/remi```, respectively, including the train, valid, test split.

You can also preprocess as below.

### 1. Download Dataset and Preprocess
Save the following dataset in `Dataset/`
* [Pop1K7](https://github.com/YatingMusic/compound-word-transformer)
* [ASAP](https://github.com/fosfrancesco/asap-dataset)
  * Download ASAP dataset from the link
* [POP909](https://github.com/music-x-lab/POP909-Dataset)
  * preprocess to have 865 pieces in qualified 4/4 time signature
  * ```cd preprocess_pop909```
  * ```exploratory.py``` to get pieces qualified in 4/4 time signature and save them at ```qual_pieces.pkl```
  * ```preprocess.py``` to realign and preprocess
  * Special thanks to Shih-Lun (Sean) Wu
* [Pianist8](https://zenodo.org/record/5089279)
  * Step 1: Download Pianist8 dataset from the link
  * Step 2: Run `python3 pianist8.py` to split data by `Dataset/pianist8_(mode).pkl`
* [EMOPIA](https://annahung31.github.io/EMOPIA/)
  * Step 1: Download Emopia dataset from the link
  * Step 2: Run `python3 emopia.py` to split data by `Dataset/emopia_(mode).pkl`

### 2. Prepare Dictionary

```dict/make_dict.py``` customize the events & words you'd like to add.

In this paper, we only use *Bar*, *Position*, *Pitch*, *Duration*.  And we provide our dictionaries in CP & REMI representation.

```dict/CP.pkl```

```dict/remi.pkl```

### 3. Prepare CP & REMI
Note that the CP & REMI tokens here only contain Bar, Position, Pitch, and Duration.
Please look into the repos below if you prefer the original definition of CP & REMI tokens.

```./prepare_data/CP```

* Run ```python3 main.py ```.  Please specify the dataset and whether you wanna prepare an answer array for the task (i.e. melody extraction, velocity prediction, composer classification and emotion classification).
* For example, ```python3 main.py --dataset=pop909 --task=melody --dir=[DIR_TO_STORE_DATA]```
* For custom dataset, run `python3 main.py --input_dir={your_input_directory}`, and the data in CP tokens will be saved at `../../data/CP/{your input directory name}.npy`.  Or you can specify the filename by adding `--name={filename}`.

```./prepare_data/remi/```

* The same logic applies to preparing REMI data. 

Acknowledgement: [CP repo](https://github.com/YatingMusic/compound-word-transformer), [remi repo](https://github.com/YatingMusic/remi/tree/6d407258fa5828600a5474354862353ef4e4e8ae)

## B. Pre-train MIDI-FiF-RoAR

```./MidiFormer/CP``` and ```./MidiFormer/remi```

* pre-train a MidiBERT-Piano
```python
python3 main.py --name=default --pos_type absolute --batch_size 20 --use_clm --use_mlm --use_fif --cuda_devices 0 1
```
A folder named ```result/pretrain/default/``` will be created, with checkpoint & log inside. You can replace the value of ```--name``` and ```--pos_type``` with ```relk```  and ```relative_key``` or ```relkq``` and ```relative_key_query```. Besides, the arguments ```--use_clm``` ```--use_mlm``` and ```--use_fif``` are optional.

* customize your own pre-training dataset
Feel free to select given dataset and add your own dataset.  To do this, add ```--dataset```, and specify the respective path in ```load_data()``` function.
For example,
```python
# to pre-train a model with only 2 datasets
python3 main.py --name=default --dataset pop1k7 asap --pos_type absolute --batch_size 20 --use_clm --use_mlm --use_fif --cuda_devices 0 1	
```

Acknowledgement: [HuggingFace](https://github.com/huggingface/transformers), [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

*Except for the ```--use_fif```, the other settings could be applied to REMI representation.*

## C. Fine-tune on Downstream Tasks

```./MidiFormer/CP``` and ```./MidiFormer/remi```

* ```finetune.py```
```python
python3 finetune.py --task=melody --name=default --pos_type absolute --use_fif --ckpt result/pretrain/default/model_best.ckpt
```
A folder named ```result/finetune/{name}/``` will be created, with checkpoint & log inside, and test loss & accuracy will be printed. You can replace the value of ```--name``` and ```--pos_type``` with ```relk```  and ```relative_key``` or ```relkq``` and ```relative_key_query```. The value of ```--task``` could be ```composer```, ```emotion```, ```melody``` and ```velocity```.

*Except for the ```--use_fif```, the other settings could be applied to REMI representation.*

## D. Baseline Model (MidiBERT-Piano)

```./MidiBERT/CP``` and ```./MidiBERT/remi```

* pre-train a MidiBERT-Piano

```python
python3 main.py --name=default
```

A folder named ```result/pretrain/default/``` will be created, with checkpoint & log inside.

* Fine-tuning

```python
python3 finetune.py --task=melody --name=default --ckpt result/pretrain/default/model_best.ckpt
```

A folder named ```result/finetune/{name}/``` will be created, with checkpoint & log inside.Test loss & accuracy will be printed, and a figure of confusion matrix will be saved.

*The same logic applies to REMI representation.*

## E. Bi-LSTM

```./Bi-LSTM/CP``` & ```./Bi-LSTM/remi```

We seperate Bi-LSTM model to note-level tasks, which used a Bi-LSTM, and sequence-level tasks, which used a Bi-LSTM + Self-attention model.

For evaluation, in note-level task, please specify the checkpoint name.
In sequence-level task, please specify only the output name you set when you trained.

* Train a Bi-LSTM
	* note-level task
	```python
	python3 main.py --task=melody --name=0710
	```
	* sequence-level task
	```python
	python3 main.py --task=composer --output=0710
	```

* Evaluate
	* note-level task:
	```python
	python3 eval.py --task=melody --ckpt=result/melody-LSTM/0710/LSTM-melody-classification.pth
	```
	* sequence-level task
	```python
	python3 eval.py --task='composer' --ckpt=0710
	```

The same logic applies to REMI representation. 

Special thanks to Ching-Yu (Sunny) Chiu

## F. Skyline

Get the accuracy on pop909 using skyline algorithm
```python
python3 cal_acc.py
```

Since Pop909 contains *melody*, *bridge*, *accompaniment*, yet skyline cannot distinguish  between melody and bridge.

There are 2 ways to report its accuracy:

1. Consider *Bridge* as *Accompaniment*, attains 78.54% accuracy
2. Consider *Bridge* as *Melody*, attains 79.51%

Special thanks to Wen-Yi Hsiao for providing the code for skyline algorithm.
