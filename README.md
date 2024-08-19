# mustard_sarcasm
Leveraging Modality-Agnostic Self Distillation Transformers for Multimodal Sarcasm Detection


This repository contains:
1) data_cleaning_keyframe.ipynb:
   notebook used to clean up mismatched instances and extract keyframes with Katna from MUStARD++
3) preprocess_feature_extraction.py:
   used to clean, preprocess, and perform feature extraction on the three modalities (audio, text, visual) of the MUStARD++ dataset
4) hyperparameter_pretesting.ipynb:
   notebook used to conduct exhaustive hyperparameter search across all ablation studies
5) final_evaluation.ipynb:
   script used to train final model and perform 5-fold cross validation on final evaluation metrics
