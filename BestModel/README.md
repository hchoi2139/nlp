# Stage 4 Submission: Hard Voting Ensemble

The predictions in `dev.txt` (F1: 0.5289) and `test.txt` were generated using a Hard Voting Ensemble across 5 K-Fold DeBERTa-v3-base models.

With discussion with Dr. Lala about the format of submitting, I was permitted to 1) remain the source code of the model under `src/` directory and 2) upload the best performing model via Google Drive.

Due to standard GitHub Git LFS storage quotas (2GB limit), pushing all 5 model weights is not possible. Therefore, the 5 ensemble checkpoints are hosted on Google Drive. Please look here: LINK_HERE

The full inference logic used to generate the 0.5289 predictions across all five folds is preserved in `src/generate_submission.py`. As I don't have gpu in my local machine, I used `submit_inference.sh` with the help of gpucluster. Then the evaluation was done by `src/evaluate_ensemble.py`. Again, the actual running has been done by the help of gpucluster, with the bash script `submit_ensemble.sh`.
