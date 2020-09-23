MY_VAR=$(grep WANDB_LOGIN credentials.ini | xargs)
MY_VAR=${MY_VAR#*=}

pip install --upgrade wandb
wandb login $MY_VAR