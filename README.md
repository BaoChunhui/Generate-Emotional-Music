# Generate-Emotional-Music

There are dataset, code, and models for "Generating Music with Emotions"

The paired lyric-melody datasets are in the file “lyrics_datasets_v3”. All the datasets are stored in the .npy format. 

Syllable-level and word-level skip-gram models with different embedding dimensions are stored in the file “Skip-gram_lyric_encoders”.

The code for training the dataset annotator by using GoEmotions dataset and Edmonds Dance dataset is shown in the file “Annotator”. To run this code, torch==1.4.0, transformers==2.11.0, attrdict==2.0.1 are required. This code is like https://github.com/monologg/GoEmotions-pytorch, so you can refer to this GitHub repository for more details.

The code for training the classifier by using EMOPIA dataset is shown in the file “EMOPIA_cls”. To run this code, torch==1.8.0 is required. You can also refer to https://annahung31.github.io/EMOPIA/ for more details. 

The code for music emotion classifier, lyric and melody generator and emotional beam search algorithm are also attached. To run this code, torch==1.9.0 is required.

For training the music emotion classifier, please first run “lyrics_datasets_v3/splitdata.py”, then run “LSTM_cls.py” and “Transformer_cls.py”. 

For training the lyric and melody generator, please run “GRU_generator.py” and “Transformer_generator.py”. 

For using the EBS algorithm to generate music segments, please run “GRU_EBS.py” and “Transformer_EBS.py”. 



