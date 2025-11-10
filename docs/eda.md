# Exploratory Data Analysis (EDA)

**Rows:** 36696

This analysis connects our approach—**suffix classification with FlexMask**—to empirical data patterns in the MAP dataset.

## Key snapshots

- **Category distribution** is imbalanced across true/false and explanation assessments.
- **Answer correctness × explanation assessment** shows strong structure (e.g., True_Misconception vs False_Correct).
- **Explanation length** varies by assessment and loosely correlates with misconception rate.
- **Misconception taxonomy** is long-tailed; a few misconceptions dominate.

### Figures

1. **Category distribution**  
   ![Category distribution](figs/category_distribution.png)

2. **Answer Correctness vs Explanation Assessment**  
   ![Heatmap](figs/answer_vs_expl_heatmap.png)

3. **Explanation length (words)**  
   ![Length histogram](figs/explanation_length_hist.png)

4. **Explanation length by assessment**  
   ![Length by assessment](figs/explanation_length_by_assessment_box.png)

5. **Misconception rate vs length deciles**  
   ![Misconception rate vs length](figs/mis_rate_vs_len_deciles.png)

6. **Top misconceptions**  
![Top misconceptions](figs/top_misconceptions.png)
