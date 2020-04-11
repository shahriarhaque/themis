Themis

# Steps to reproduce Themis model

## Original Paper
[Phishing Email Detection Using Improved RCNN Model With Multilevel Vectors and Attention Mechanism](https://ieeexplore.ieee.org/document/8701426)

## Download training data:

1) [Nazario Phishing - MBox Format](https://monkey.org/~jose/phishing/phishing3.mbox)
2) [IWSPA 2018 Phishning & Benign - TXT Format](https://github.com/BarathiGanesh-HB/IWSPA-AP/blob/master/data/Dataset_Full_Header_Training.zip)

## Convert IWSPA TXT Format to JSON

1) Unzip IWSPA zip archive
2) Edit `data/iwspa-to-json.py` and modify `INPUT_DIRECTORY` variable to point to unzipped legit training folder.
3) Modify `OUTPUT_FILE` as needed.
4) Run `data/iwspa-to-json.py`
5) Repeat Steps 2, 3, and 4 for unzipped phishing folders.

## Pre-process IWSPA JSON files.

1) Edit `data/iwspa-preprocess.py` and modify `INPUT_DIRECTORY` and `INPUT_FILE` variables to point to the legit IWSPA JSON file from the previous section.
2) Modify `OUTPUT_DIRECTORY` and `OUTPUT_FILE` as needed.
3) Run `data/iwspa-preprocess.py`.
4) Repeat steps 1,2, and 3 for phishing JSON file.

## Train Themis Model
1) Edit `model/themis-build-model.py` and modify `INPUT_DIRECTORY`, `LEGIT_INPUT_FILE`, and `PHISH_INPUT_FILE` to point to the directory containing the JSON files from the previous sections, legit IWSPA JSON file and phishing IWSPA JSON files respectively.
2) Modify `MODEL_DIRECTORY` and `MODEL_FILE` as needed. This is the location where the trained model and Keras Tokenizers will be saved.
3) Run `model/themis-build-model.py`.
4) Toggle the following function calls in `run` as per the following instruction:
  - `train_and_save_tokenizers`: Train new tokenizers based on the training data and save them.
  - `load_model_and_tokenizer`: Load an existing tokenizer instead of training them.
  - `predict_classes(model, test_df, tokenizers)`: Run model on test data frame.
  - `predict_classes(model, validation_df, tokenizers)`: Run model on validation data frame.
  - `prediction_metrics(y_test, y_hat_test)`: Print confusion matrix
