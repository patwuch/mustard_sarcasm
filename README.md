# mustard_sarcasm
Leveraging Modality-Agnostic Self Distillation Transformers for Multimodal Sarcasm Detection


This repository contains:
1) keyframe_extraction.ipynb:
   extract keyframes with Katna from MUStARD++ and isolate videos whose keyframe cannot be extracted
2) preprocess_feature_extraction.py:
   used to clean, preprocess, and perform feature extraction on the three modalities (audio, text, visual) of the MUStARD++ dataset
   (the unextractable videos are preprocess and feature extracted directly at this stage)
3) hyperparameter_pretesting.ipynb:
   notebook used to conduct exhaustive hyperparameter search across all ablation studies
4) final_evaluation.ipynb:
   script used to train final model and perform 5-fold cross validation on final evaluation metrics
