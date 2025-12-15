# thesis_revival_project
# Logit-to-Learning-NGSIM
"From Statistic to System: A Machine Learning Approach to Driver Behavior Prediction for Autonomous Vehicles". "From Logit to Learning: A Modern Re-Implementation and Enhancement of a Driver Behavior Model". Revived and modernized master's thesis on driver lane-changing behavior; re-implemented the multinomial logit model in Python, improving data processing efficiency and model interpretability compared to the original statistical software.


Project 1: Driver Behavior Modeling (Thesis Revival)
    Technologies: Python, Pandas, Scikit-learn, Statsmodels

* For my master's thesis in 2020, I used the statistical tools available to me (SPSS) to build a robust multinomial logit model, proving I understand the underlying theory. I am now modernizing this work by re-implementing it in Python with libraries like Pandas and Statsmodels, which allows for greater scalability and integration with modern ML pipelines.
* I'm aware of the discussions around the NGSIM dataset's precision. Working with real-world, imperfect data was a valuable lesson in data hygiene and critical analysis. My current projects using the newer highD or Waymo Open Datasets have further honed these skills.
* My thesis established a strong baseline using traditional statistical methods. This foundational understanding is crucial. I am now expanding on this by applying machine learning techniques like XGBoost and LSTMs to the same problem, comparing their performance against my original logit model to see where deep learning provides an edge.
* My thesis successfully identified a behavioral cluster. The logical next step, which I'm exploring now, is to build a predictive model that can, in real-time, classify a driver into this 'Gap Precision Class' based on their recent trajectory, which could be a powerful feature for an AV's predictive policy.



Step 1: The Rebuild: Re-implement the Data Pipeline in Python.
    * Action: Download the NGSIM data again. Write a Python script using Pandas to load, clean, and merge the trajectory files. Write a Python script using Pandas to replicate all the data cleaning, filtering, and feature engineering you did in Alteryx. Implement a Latent Class Model or use a Clustering Algorithm (like Gaussian Mixture Models) before the classification step. This is an unsupervised learning technique that will algorithmically discover different driver types in your data. You can then build a separate Multinomial Logit model for each discovered class. This would be a significant upgrade and a very impressive portfolio piece.
    * Goal: Demonstrate mastery of the primary data science library. This will be messy and will build immense character (and skill).
Step 2: Rebuild the Multinomial Logit Model.
    * Action: Use the statsmodels library in Python to recreate your original model. Reproduce your original Multinomial Logit Model in Python using statsmodels. Acknowledge the separation issue you find.
    * Goal: Validate your original findings and prove you can implement the same logic with modern code.
Step 3: Enhance with Machine Learning.
    * Action: This is the crucial step. Treat gap acceptance as a classification problem.
       *  Model 1: Use XGBoost or Random Forest to predict gap acceptance (Accept/Reject).
        * Model 2 (Advanced): Frame it as a time-series problem. Use an LSTM network to predict a driver's next action based on their last 5-10 seconds of trajectory.
        * Build a Regularized Multinomial Logistic Regression model using scikit-learn. Compare the results. The new, stable coefficients will be your "real" findings.
        * Discover Driver Classes: Use a Clustering Algorithm (Gaussian Mixture Models) on driver behavior metrics (mean speed, max acceleration, average headway) to find natural groupings in the data. Visualize these clusters.
        *  Build Class-Specific Models: Train a separate Multinomial Logit model for each driver cluster. This will show that "aggressive" drivers have different decision parameters than "cautious" ones. This is a much more powerful demonstration of "driver heterogeneity."
        * Visualization: Create an animation using matplotlib or plotly that shows a vehicle trajectory and your model's prediction of its lane-change probability over time.
    * Goal: Compare the performance (Accuracy, F1-Score) of your new ML models against your original logit model. Which one is better? Why do you think that is?
Step 4: Publish and Explain.
    * Action: Create a GitHub repository with:
        * The clean, well-commented code.
        * A Jupyter Notebook that walks through your entire process, from data loading to model comparison.
        * A README.md file that explains the project, your findings, and the business/AV implications.

* "This model can serve as a baseline for predicting human driver behavior, a
