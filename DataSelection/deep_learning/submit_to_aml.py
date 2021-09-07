#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import sys

from DataSelection.deep_learning.utils import load_model_config
from DataSelection.utils.aml import submit_aml_job
from DataSelection.utils.default_paths import PROJECT_ROOT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file characterising trained CNN model/s')
    args, unknown_args = parser.parse_known_args()
    config = load_model_config(args.config)
    submit_aml_job(script_args=sys.argv[1:],
                   script_path=PROJECT_ROOT_DIR / "DataSelection" / "deep_learning" / "train.py",
                   experiment_name="DS_supervised",
                   script_config=config)
