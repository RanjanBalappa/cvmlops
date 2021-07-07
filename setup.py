from pathlib import Path
from setuptools import setup

setup(
    name="cv",
    version="0.1",
    python_requires=">=3.8",
    entry_points={
        "console_scripts":[
            "cv = app.cli:app"
        ]
    }
)