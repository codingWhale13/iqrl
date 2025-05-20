import wandb


api = wandb.Api()
all_runs = api.runs("kielen1-aalto-university/iqrl")
with open("WANDB_RUN_IDS.txt", "w") as f:
    for run in all_runs:
        f.write(f"{run.name} ({run.notes}) => {run.id}\n")
