# SHAP Explainability Report — German Credit (skeleton)

## 1. Project Summary
- Task: Binary classification (credit risk)
- Model: XGBoost (Stratified K-Fold cross-validation)
- Dataset: German credit (source: Kaggle notebook you provided). Dataset version: [add link/slug]
- Samples / features: [fill after running]

## 2. Validation Metrics
- OOF AUC: [fill]
- Fold AUCs: [list]
- Notes on model stability and overfitting.

## 3. Global SHAP Summary
- Top features by mean(|SHAP value|): [list top 10]
- Interpretation: [narrative on directionality and interactions]

## 4. Feature Dependence Highlights
- For top features, include SHAP dependence descriptions and any non-linear patterns.

## 5. Local Explanations (3 instances)
- Instance 1 (highest predicted risk): top 5 contributing features and explanation.
- Instance 2 (lowest predicted risk): top contributors and explanation.
- Instance 3 (borderline ~0.5): explanation.

## 6. Global vs Local Comparison
- Where do local explanations align or contradict the global picture?

## 7. Potential Biases & Limitations
- Data quality issues, proxy variables, sampling or label problems.

## 8. Three Actionable Recommendations (derived from SHAP)
1. Recommendation 1 — what to change operationally and why (based solely on model explanations).
2. Recommendation 2
3. Recommendation 3

## 9. Reproducibility & Appendix
- Exact commands used, environment, and where artifacts are stored.
