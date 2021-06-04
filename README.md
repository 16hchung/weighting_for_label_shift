# Importance Weighting for Label Shift
_Evaluating Black Box Shift Correction (Lipton et al., 2018) on WILDS_

Scripts under `examples/` are adapted from the same directory in the [WILDS codebase](https://github.com/p-lambda/wilds/tree/50b26775e5ffdf6fc3cd0da4c7b899eb43aefdfd)

### Command to run experiments
```python examples/run_expt.py --version "1.0" --root_dir data --log_dir <LOG DIR> --dataset camelyon17 --algorithm BBSE --model densenet121 --seed 0 --bbse_knockout <KNOCK-OUT RATIO> --n_epochs 3```
