ECBM4040 Project Notification:
Author: RECO (jm4743)

Prog 1: Data_Extraction.ipynb
    Run the File first to download and extract the dataset, and to acquire the basic data info such as the amount of the dataset.

Prog 2: TFrecords_Preparation.py 
    (just rememer to modify the parameter mentioned below to create the data you need)
    extra_included: Whether the image in the "extra" folder is included in the train/val dataset (I set 1 as default value)
    WT_flag: Whether the Wavelet Transform is applied on the data (I set 0 as default value)
    (When you want to create the trrecords as you with, you need to modify the program in this file, but to run the file ,you can just run the file train.py, so that the data and the training will be finished one time.)
    
Prog 3: train.py
    Run the file after run the Prog 1, to train the ConvNet, and run an evaluation for the entire training & testing dataset to get accuracy for a selected model.
    Check the main function for 
        1. model parameter adjusting (parameter "flatten" need to be calculated according to the parameter you set before).
        2. if the TFrecord has not been created, then remember to run encoder_main(). Otherwise, delete or add "#" at the begining of that line
        3. For the variable LogTrain_dir in main function, this item determines the folder name used to restore the checkpoint file. I set that by concatenating that the time information. Thus, in the log folder, you can find the corresponding folder and no need to change the name of already existed folder to retrain the model
	4.Take the item mentioned in training into consederation, the variable Coresponding_time is set to help you find the corresponding checkpoint folder. just seting the variable as "XXXXXXXXXXXXXXXXX" part of folder name in /log “trainXXXXXXXXXXXXXXXXX"
        2.Remember to change the corresponding model_param and tfrecords name

Prog 4: model.py
    Build the model in loop form. This will be easier for you to adjust the parameter and adopt the idea of ResNet. But remember to change the parameter setting in corresponding function

Appendix:Spectrum.ipynb
    This file is used to test the calculation of wavelet transform, but the wavelet part of the image processing has been added into the file TFrecords_Preparation.py
    
Appendix:Inference.ipynb
    The file required to display the result (For the convenience of comparison, we rename the data with wavelet transform by applying prefix "WT5", e.g. WT5-test.tfrecord)
    
About data shape:
    The paper set the max digit length to 5. 
    Even through the maximum of digit length in train set is 6, but there is only one example that has 6 digit, which is pretty  insignificant considering the total amount of images is 248,823. So according to the instruction from that paper, we set the digit length to 5. The length can be modified by adjusting the parameter in the following function unit: Decoder_TFrecord, Encoder_TFrecord, inference [for digit_cnt in range(6)], loss [sum up loss]
