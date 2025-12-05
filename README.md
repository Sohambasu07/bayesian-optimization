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