#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import sys

from DataSelection.deep_learning.utils import load_ssl_model_config
from DataSelection.utils.aml import submit_aml_job
from DataSelection.utils.default_paths import PROJECT_ROOT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a self-supervised model')
    parser.add_argument('--config', dest='config', type=str, required=True,
                        help='Path to config file characterising trained CNN model/s')
    parser.add_argument('--tag', type=str, required=False, default="",
                        help='A string tag that is attached to the AzureML run')
    args, unknown_args = parser.parse_known_args()
    config_path = args.config
    script_config = load_ssl_model_config(config_path)
    tags = {"tag": args.tag} if args.tag else None
    submit_aml_job(
        script_args=sys.argv[1:],
        script_path=PROJECT_ROOT_DIR / "DataSelection" / "deep_learning" / "self_supervised" / "main.py",
        experiment_name="DS_unsupervised",
        script_config=script_config,
        tags=tags)
