import click
from click.testing import CliRunner
import dill
import numpy as np

from pipeline_steps.income_classifier.pipeline_step import run_pipeline as income_cli

import sys
sys.path.append("..")

def test_pipeline():

    runner = CliRunner()
    
    with runner.isolated_filesystem():

        # # Test Data Downloader
        # result = runner.invoke(download_cli, [
        #     '--labels-path', 'labels.data',
        #     '--features-path', 'raw_text.data'])

        # with open('raw_text.data', "rb") as f:
        #     assert f
        # with open('labels.data', "rb") as f:
        #     assert f


        # Test Income Classifier model
        result = runner.invoke(income_cli, [
            '--preprocessor-path', 'preprocessor.model',
            '--model-path', 'income_class.model',
            '--out-path', 'clf_prediction.data',
            '--action', 'train'])

        with open("income_class.model", "rb") as f:
            assert f

