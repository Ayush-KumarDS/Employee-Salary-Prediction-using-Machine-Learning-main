4. The system executes vectorized inference and appends a `PredictedClass` column.
5. Click **"⬇️ Download Predictions CSV"** to save your results.
---
## 💡 Engineering Highlights & Best Practices
1. **Zero Data Leakage**: Transformations (`StandardScaler`, `OneHotEncoder`) are fitted strictly within cross-validation folds and training partitions via `sklearn.pipeline.Pipeline`.
2. **Encapsulated Artifact**: The saved `model.pkl` is a complete composite pipeline. The web application feeds raw DataFrames directly into `model.predict(input_df)` without needing manual scaling or one-hot encoding code inside the web server.
3. **Resilient Categorical Encoding**: Categorical encoders use `handle_unknown='ignore'`, preventing runtime crashes when novel or rare categories are introduced in user input.
4. **Vectorized Batch Processing**: Uses Pandas vectorized operations for batch uploads, allowing processing of thousands of records in seconds.
---
## 🔮 Future Roadmap
- [ ] **Class Imbalance Optimization**: Implement SMOTE (Synthetic Minority Over-sampling Technique) or cost-sensitive learning to boost recall on the `>50K` minority class.
- [ ] **Hyperparameter Optimization**: Conduct Bayesian optimization with Optuna to tune Gradient Boosting estimators and tree depths.
- [ ] **Explainable AI (XAI)**: Integrate **SHAP** (SHapley Additive exPlanations) into the Streamlit dashboard to explain individual feature contributions for each prediction.
- [ ] **Containerization & Cloud Deployment**: Add `Dockerfile` and deploy the service on cloud platforms (e.g., Streamlit Community Cloud, Render, or AWS ECS).
---
## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
---
## 🤝 Acknowledgements
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) for providing the Adult Census Income dataset.
* [scikit-learn](https://scikit-learn.org/) and [Streamlit](https://streamlit.io/) open-source communities.
