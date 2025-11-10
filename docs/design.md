# Design Notes

## Modules
- `data.py`: I/O, de-duplication, stratified K-fold creation.
- `prompt.py`: Template rendering for prefix and suffix.
- `mask.py`: FlexMask construction to isolate suffixes.
- `model.py`: LLM wrapper extracting last-token embeddings, linear scorer, and training step.
- `train.py`: Orchestrates one-fold/one-seed training with logging.
- `infer.py`: Runs scoring with optional 8-bit loading for memory efficiency.
- `ensemble.py`: Aggregates logits from multiple seeds into final predictions.

## Reproducibility
- Global seeding for Python/NumPy/PyTorch.
- Deterministic dataloader and shuffling.
- Config-driven runs via YAML.
