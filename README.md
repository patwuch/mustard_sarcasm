# mustard_sarcasm
Leveraging Self Distillation Multimodal Transformers for Sarcasm Detection


This repository contains:
1) preprocess_feature_extraction.ipynb:
   used to clean, preprocess, and perform feature extraction on the three modalities (audio, text, visual) of the MUStARD++ dataset
2) hyperparameter_pretesting.ipynb:
   notebook used to conduct exhaustive hyperparameter search across all ablation studies
3) final_model.py:
   script used to train final model and perform 5-fold cross validation on final evaluation metrics
