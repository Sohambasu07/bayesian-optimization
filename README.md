# Bayesian Optimization

### Instructions to run

```bash
python -m bayesian_optimization \
    --device cuda \ # Optional
    --seed 1 \ # Optional
    --epochs 20 \ #Optional
```

### Installation from source

```bash
git clone https://github.com/Sohambasu07/bayesian-optimization.git
cd bayesian-optimization

python3 -m venv bayesopt_env
source bayesopt_env/bin/activate  # Linux

pip install -e . # For editable install
```

### Results

The final HP config (optimal learning rate) can be found at `./results/best_config.yaml`

### Plots

The plots for every iteration of Bayesian Optimization can be found at `./plots/...`
Each plot contains 2 subplots: 
- The top one showing the Posterior mean and uncertainty estimates, along with the observations on which the GP was fitted
- The bottom one showing the Acquisition function at that iteration