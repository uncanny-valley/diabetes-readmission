# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from os import path
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    process_diabetes_data(logger, input_filepath, output_filepath, "diabetes")

def process_diabetes_data(logger, input_filepath, output_filepath, target_filestem):
    logger.info('processing and exporting diabetes dataset')
    input_csv_path = path.join(input_filepath, target_filestem + '.csv')
    output_csv_path = path.join(output_filepath, target_filestem + '.csv')
    output_pickle_path = path.join(output_filepath, target_filestem + '.pkl')

    df = pd.read_csv(input_csv_path, index_col=0)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
