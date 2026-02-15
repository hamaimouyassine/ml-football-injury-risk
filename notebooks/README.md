# Notebooks

Place your exploratory notebooks here (EDA, visualizations, experiments).

The main pipeline is run via:

```bash
python -m src.pipelines.pipeline
```

Or from Python:

```python
from src.pipelines.pipeline import run_full_pipeline
df = run_full_pipeline(run_all=True)
```
