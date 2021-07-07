import typer
from pathlib import Path
import torch
from cv import main, train

app = typer.Typer()


@app.command()
def hello():
    print('Inside hello')


