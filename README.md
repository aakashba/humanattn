# humanattn2022
# Replication package for the paper "Modelling Human Attention from Eye Movements for Neural Source Code Summarization" under review at ICSE 2022.
Kindly download the data for replication at https://drive.google.com/drive/folders/1hH1bS8ulN84tT0LvXyAhBpGL5F0HolJH?usp=sharing

## Step 1: Prepare human data
From the google drive folder, extract humandata.zip and either place the contents at /nfs/projects/humanattn/data/eyesum or change this locaton in train.py inside the humantrain/ directory in this repo

## Step 2: Train on human attention
Using the following command(s) inside the humantrain directory of this repo, a model can be trained over human eye gazing data
```
python3 train.py --model-type=astgnn --epochs=70 --batch-size=200 
```
Alternatively, 
```
./runner.sh>output.txt
```
may be used to extract average correlations for training human attention

**The resulting models are provided in the humanmodels.zip folder**

## Step 3: Prepare machine attention data
From the google drive folder, extract q90dats.zip and either place the contents at /nfs/projects/humanattn/data/javastmt/q90 or change this location inside biodatmaker.py inside the makebiodat/ directory in this repo.

## Step 4: Make biodats
Using the following command inside the makebiodat directory , create the biodats
```
python3 biodatmaker.py
```
You may need to edit the model location and name inside the biodatmaker.py file
These biodats are also all provided in the dataset at biodats.zip 

## Step 5: Train machine attention
Using the following command inside the main repository, you may train the model that mimicks human attention
```
python3 train.py --model-type=attendgru-bio --with-graph --with-biodats=biodats_q90_astgnn.pkl --epochs=10
```
You may need to change the directories inside train.py to read in the q90data as well as the biodats.

**These trained models are also available in the dataset at biomodels.zip, the timestamp indicated which biodats were used:**
* bio_base - 1626830712
* astflat_pretrain- 1628781070
* astflat - 1628774592
* gnn-pretrain - 1628770439
* gnn - 1628089432

## Step 6: Predictions
```
python3 predict.py /path/to/model/ --with-graph --with-biodats=biodats_q90_astgnn.pkl
```

**gpu=X can be used with any of these scripts to make use of gpu instead of cpu depending on availability, we used a titan rtx 24gb to train these models with tensorflow 2.4**

Resulting predictions are available inside the predictions directory of this repository.
