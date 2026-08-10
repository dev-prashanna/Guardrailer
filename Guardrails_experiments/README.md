# Experiments

Sandboxed area for testing new approaches. Each experiment is self-contained and does not modify any code in the main project.

## Structure

```
experiments/
├── approaches/          # Each experiment in its own subfolder
│   └── <experiment_name>/
│       ├── config.yaml  # Experiment config
│       ├── run.py       # Experiment script
│       ├── requirements.txt
│       └── README.md
├── results/             # Experiment outputs (auto-generated)
├── logs/                # Run logs
└── configs/             # Shared experiment configs
```

## Rules

1. **No imports from `guardrailer_security/`** — each experiment is standalone
2. **No writes outside `experiments/`** — results stay here
3. **Each experiment has its own `requirements.txt`** — no dependency conflicts
4. **All outputs go to `experiments/results/<experiment_name>/`**

## Usage

```bash
# Create new experiment
mkdir approaches/my_experiment
touch approaches/my_experiment/{config.yaml,run.py,requirements.txt,README.md}

# Run experiment
cd approaches/my_experiment
pip install -r requirements.txt
python run.py
```

## Notes

- Experiments should be reproducible (fixed seeds, saved configs)
- Delete old experiments when no longer needed
- Move promising results to the main project after validation
