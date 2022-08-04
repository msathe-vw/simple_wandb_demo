# W&B Demo Repo and Utilities

## Setup

 1. Create a conda environment
 2. Install [PyTorch](https://pytorch.org/get-started/locally/)
 3. Install the project requirements
  `pip install -r src/requirements.txt`
 4. Log in to W&B, and then `wandb login` using this [authorization key](https://wandb.ai/authorize)
 5. Download [PennFudan Dataset](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip) and extract to a convenient location

## Example training command

`python src/train.py --config src/default_config.json --train PATH/TO/PennFudanDataset`

## Useful Notes

- `utils/config_utils.py` has a parser that merges CLI, JSON and SageMaker Environment configuration options. It uses [addict](https://github.com/mewwts/addict) to provide dot syntax. Note: CLI doesn't override JSON currently, will fix this.
- Searching for "# W&B" without the quotes at the repo level in your favorite IDE will identify every line modified for use with W&B

## Future Work
- Add demo of multiple logs per step in different log statements
- Add graceful resuming of crashed runs across mixed environments
- Add hyperparameter tuning configuration example YAML
- Investigate artifact tracking in more detail
- Investigate usage in distributed processes
- Fix naming of runs
- Add demo in sagemaker
