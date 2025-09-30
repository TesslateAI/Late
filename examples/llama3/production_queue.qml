# Example production training queue for Llama 3 models
# This queue demonstrates a typical workflow:
# 1. Quick sweep to find optimal learning rate
# 2. Full training with best parameters
# 3. Evaluation runs with different datasets

# Note: Replace these with absolute paths on your system
/workspace/configs/llama3_lr_sweep_config_0.yml
/workspace/configs/llama3_lr_sweep_config_1.yml
/workspace/configs/llama3_lr_sweep_config_2.yml
/workspace/configs/llama3_lr_sweep_config_3.yml
/workspace/configs/llama3_lr_sweep_config_4.yml
/workspace/configs/llama3_full_training.yml
/workspace/configs/llama3_eval_dataset1.yml
/workspace/configs/llama3_eval_dataset2.yml