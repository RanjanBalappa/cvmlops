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

#add stores registry
STORESDIR = Path(BASEDIR, 'stores')
BLOBSTORE = Path(STORESDIR, 'blob')
MODELSTORE = Path(STORESDIR, 'model')
mlflow.set_tracking_uri(f'file://{MODELSTORE.absolute()}')




@app.command()
def trainmodel(
    paramsfile: Optional[Path] = Path(CONFIGDIR, 'params.json'),
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

        tags = {'runid': runid}
        mlflow.set_tags(tags)

        #Train the model
        artifacts = main.trainmodel(params)
        print('Training Done')
        
        #store artifacts
        performance = artifacts['performance']
        metrics = {
            'precision': performance['overall']['precision'],
            'recall': performance['overall']['recall'],
            'f1score': performance['overall']['f1score'],
            'top1accuracy': artifacts['top1accuracy'],
            'bestloss': artifacts['bestloss']

        }

        mlflow.log_metrics(metrics)
        mlflow.log_params(vars(params))

        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(params), Path(dp, 'params.json'))
            utils.save_dict(performance, Path(dp, 'performance.json'))
            torch.save(artifacts["model"].state_dict(), Path(dp, 'model.pt'))
            mlflow.log_artifacts(dp)


    # Save for repo
    open(Path(modeldir, 'runid.txt'), 'w').write(runid)
    utils.save_dict(vars(params), Path(modeldir, 'params.json'))
    utils.save_dict(performance, Path(modeldir, "performance.json"))


@app.command()
def evalmodel(
    runid: Optional[str] = open(Path(MODELDIR, 'runid.txt')).read(),
    datafolder: Optional[str] = 'test'
):
    artifacts = main.loadartifacts(runid)
    params = artifacts['params']
    model = artifacts['model']
    device = artifacts['device']
    lossfn = artifacts['lossfn']
    optimizer = artifacts['optimizer']

    performance = main.evaluate(model, device, lossfn, optimizer, params, datafolder)
    print(json.dumps(performance, indent=2))


@app.command()
def predict():
    pass