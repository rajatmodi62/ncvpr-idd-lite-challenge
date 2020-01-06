# IDD-challenge
IDD lite dataset challenge

# To-Do list

- [x] Register the team for the IDD challenge. The name of the team should be PARIMAL.
- [x] Cleanup dataset, introduce RLE encoding
- [x] Compute Class balance
- [x] Pearson plots for feature correlation
- [x] Visualization of multiple segmentation maps
- [x] CleanUp Dataset
- [x] Split Train set into folds to perform bagging
- [x] Construct the Preprocessing Pipeline
- [x] Added Optimizers, models and losses factory
- [x] Added papers for RAdam etc.
- [x] Training pipeline has been set up. Use with the experiments.
- [x] Upload the config files
- [x] Decide the image resolutions for the experiments.
- [x] make validation code
- [x] ensemble the models
- [x] First iteration of masks generated.
- [ ] [@Rajat]Push the complete docker image to support the environment
- [ ] [@Deepansh] Fork this code in a separate branch
- [ ] [@Deepansh] Make a submission script and check the leaderboard ranking
- [ ] [@Deepansh]Add the resize handling in the ensembling.py script
- [ ] [@Deepansh]Train the rest of the config files in config/seg/dolater folder.
- [ ] [@Deepansh]Compute the mask & label threshold for post-processing for more accurate masks
- [ ] [@Deepansh]Combine the train & val folders and train collectively





# Environment
Do a hit and trial as of now. Will update with docker script shortly. 


# Setting Up The Dataset
Cleaned up IDDLite Dataset can be found [here](https://drive.google.com/open?id=10cuXNHqD7JZgbNO-ah2mpSpYNIIzZWe1).



```bash
├── dataset
│   ├── <unzip dataset here>
│   ├── test/
│   ├── train/
│   ├── val/
│   ├── masks/
│   ├── train.csv
│   ├── val.csv
│   ├── labels.csv
│   ├── train_folds.csv
├── README.md
├── config
├── utils
├── jupyter_experiments
└── .gitignore
```
# Creating Dataset folds
```
$ python split_fold.py --config config/base_config.yml

```
* Edit the base_config.yml to adjust the no of folds.
This is gonna create train_folds.csv for bagging (default:5). Use val folder as a test set for now (for Val score).

* Later on mix both train and val sets to increase the training accuracy.
# Training
```
$ bash train.sh

```
* Edit the configuration file based on the experiment you which to perform. Initialize the Architecture, Encoder, and the types of pre-trained to initialize the models with.

* Ensemble the models with ensemble.py
```
$ python ensemble.py

```

# Results
Results with version information:

| Trial Date | Model Information |Model Weights/Masks
| ------------- | ------------- |--------
| 7/12/19  | EfficientNet b7+adam, Efficientnetb0 + cosineannealing  |[Link](https://drive.google.com/open?id=1W7_HQCv5oZZZXKlhqwUwvSJlFniQixu7)
| Dummy  | Dummy  |Dummy

# PostProcessing
@Deepansh, I have used connected components in the mask postprocessing. The pixels which are less than some particular threshold in the mask are being set as 0 currently. Compute the proper threshold per class experimentally. Please refer to preprocessing function in the utils folder. Let me know if you have any doubts.
# Errors

* Mac creates .DS_Store in dir. Handle file parsing of .jpg images.
