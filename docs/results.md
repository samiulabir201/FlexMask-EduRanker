# Results

**Main Validation (3-seed ensembles)**

| Model | Loss | MAP@3 |
|---|---:|---:|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 0.2716 | 0.9444 |
| Qwen/Qwen3-8B | 0.2677 | 0.9455 |
| zai-org/GLM-Z1-9B-0414 | 0.2627 | 0.9469 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | 0.2621 | 0.9464 |
| Qwen/Qwen3-14B | 0.2614 | 0.9477 |
| Qwen/Qwen3-32B | 0.2589 | 0.9484 |
| zai-org/GLM-Z1-32B-0414 | 0.2560 | 0.9480 |
| Qwen/Qwen3-32B + GLM-Z1-32B | **0.2530** | **0.9496** |

**Key Takeaways**

- Do **not** trust single-seed validation scores.
- Larger models improve performance.
- Loss correlates more reliably with generalization than MAP@3.
