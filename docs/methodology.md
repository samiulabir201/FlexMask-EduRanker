# Methodology

## Problem Framing
We model the task as **suffix classification**. Given a shared **prefix** (question, choices, correct answer,
misconception candidates, student answer & explanation), we predict the **correct suffix** among candidates.

### Prefix Template

```
<|im_start|>user
**Question:** {QuestionText}
**Choices:** {MC_Choices}
**Correct Answer:** {Answer}
**Common Misconceptions:** {MisconceptionCandidates}
**Student Answer:** {MC_Answer}
**Student Explanation:** {StudentExplanation}
<|im_end|>
<|im_start|>assistant
```

### Suffix Formatting
Suffixes are in submission format, e.g. `False_Correct:NA`

Each `QuestionId` has 8/10/12 candidates depending on misconceptions. We extract the **last-token features** of
`[prefix ++ suffix0, prefix ++ suffix1, ...]` and score them via a linear head `nn.Linear(hidden_size, 1)` to obtain logits.
Training uses cross-entropy over the candidate set.

We adopt **prefix-shared packaging**: `prefix ++ suffix0 ++ suffix1 ++ ...` with a **custom attention mask** that enforces:
- causal flow,
- visibility from suffix tokens to their own suffix and the prefix,
- no leakage across different suffixes,
- same-document constraints.

### FlexMask (Custom Mask)

```
causal = q_idx >= kv_idx
is_prefix = suffix_ids[kv_idx] == -1
same_suffix = (suffix_ids[q_idx] == suffix_ids[kv_idx])
same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
allow = causal & (same_suffix | is_prefix) & same_doc
```

## Data
- Deduplicate; final samples: **35,960**.
- 5-fold cross-validation stratified by **Category**.

## Modeling
- Backbone LLMs (Qwen/GLM/DeepSeek family).
- Larger models generally perform better.
- Prefer **loss** over MAP@3 for early stopping.

## Training
- Efficient full-parameter training possible with high-memory GPUs (A100 80G / RTX 6000 Blackwell).
- Typical hyperparameters: `epochs=1`, `batch_size=32`, `lr=1e-5`.
- **Multi-seed ensembling** stabilizes validation (label noise suspected).

## Inference
- Compute bottleneck: dense `nn.Linear` ops during short-sequence prefilling.
- **INT8 W8A8** inference (e.g., SmoothQuant `alpha=0.75`) preserves ensemble performance.
- Layer-wise streaming can enable 32B inference on smaller GPUs.

## Validation & Results
- Multi-seed > multi-fold for stability.
- Example MAP@3 and loss per model family (best ~0.9496). See `docs/results.md` for details.

## Environment Notes (Kaggle)
- `/kaggle/input` is remote & slow, `/tmp` is CoW, `/kaggle/working` is the safe local path.
- Avoid filling `/tmp` with large checkpoints.
