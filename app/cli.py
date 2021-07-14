from typing import Optional
import typer
from pathlib import Path
import yaml
import json
from argparse import Namespace
import tempfile

import torch
import mlflow
from cv import main, train, utils

app = typer.Typer()

BASEDIR = Path(__file__).parent.parent.absolute()
CONFIGDIR = Path(BASEDIR, 'config')
MODELDIR = Path(BASEDIR, 'model')


@app.command()
def trainmodel(
    paramsfile: Optional[Path] = Path(CONFIGDIR, 'config.json'),
    experimentname: Optional[str] = 'best',
    runname: Optional[str] = 'model',
    modeldir: Optional[Path] = Path(MODELDIR)
):
    configfile = open(Path(paramsfile), 'r')
    params = Namespace(**utils.load_dict(filepath=paramsfile))

    #Start Run
    mlflow.set_experiment(experiment_name=experimentname)
    with mlflow.start_run(run_name=runname):
        runid = mlflow.active_run().info.run_id

        #Train the model
        artifacts = main.trainmodel(params)
        
        #store artifacts
        performance = artifacts['performance']
        print(json.dumps(performance, indent=2))
        metrics = {
            'precision': performance['overall']['precision'],
            'recall': performance['overall']['recall'],
            'f1score': performance['overall']['f1score'],
            'top1accuracy': performance['overall']['top1']

        }

        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(performance, Path(dp, "performance.json"))
            torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
            mlflow.log_artifacts(dp)


    # Save for repo
    open(Path(modeldir, "run_id.txt"), "w").write(runid)
    utils.save_dict(performance, Path(modeldir, "performance.json"))

    



    


    

