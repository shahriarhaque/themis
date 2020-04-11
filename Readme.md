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
