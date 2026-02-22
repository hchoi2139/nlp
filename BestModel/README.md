# Stage 4 Submission: Hard Voting Ensemble

The predictions in `dev.txt` (F1: 0.5252) and `test.txt` were generated using a Hard Voting Ensemble across 5 K-Fold DeBERTa-v3-base models. 

Due to standard GitHub Git LFS storage quotas (1GB limit), pushing all 5 model weights (~3.5GB total) is not possible. Therefore, only the weights for the single highest-performing internal model (Fold 2) are provided in this directory as `model.pth`. 

The full inference logic used to generate the 0.5252 predictions across all five folds is preserved in `src/generate_submission.py`.
