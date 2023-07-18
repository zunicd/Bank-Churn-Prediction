The app is deployed to Render, and can be found at this [URL](https://bank-churn-predictions.onrender.com)

Our full presentation deck [can be found here](https://docs.google.com/presentation/d/1AiL9Zo9WBySLQkjjMcIHuZp4wEC9UnLtXjSaclaICBE/edit?usp=sharing).

# Churn Modelling - How to predict if a bank’s customer will stay or leave the bank

Using a source of 10,000 bank records,  we created an app to demonstrate the ability to apply machine learning models to predict the likelihood of customer churn. We accomplished this using the following steps:

## 1. Clean the data

By reading the dataset into a dataframe using pandas,  we removed unnecessary data fields including individual customer IDs and names. This left us with a list of columns for Credit Score, Geography, Gender, Age, Length of time as a Bank customer, Balance, Number Of Bank Products Used, Has a Credit Card, Is an Active Member, Estimated Salary and Exited. 

## 2. Analyze initial DataFrame

Utilizing Matplotlib, Seaborn and Pandas, we next analyzed the data. We can see that our dataset was imbalanced. The majority class, "Stays" (0), has around 80% data points and the minority class, "Exits" (1), has around 20% datapoints. To address this, we utilized SMOTE in our machine learning algorithms (Synthetic Minority Over-sampling Technique). More on that later on. 

In percentage, female customers are more likely to leave the bank at 25%, compared to 16% of males.

The smallest number of customers are from Germany, and they are also the most likely to leave the bank. Almost one in three German customers in our sample left the bank.

## 3. Machine Learning using 7 different models

We tested seven different machine learning models (and used six in the final application) to predict customer churn, including Logistic Regression, Decision Tree, Random Forest, Deep Learning (TensorFlow), K-Nearest Neighbor, Support Vector Machine and XGBoost. 

As mentioned earlier, we also used SMOTE to handle issues with the imbalanced data on the Support Vector Machine model. SMOTE (Synthetic Minority Over-sampling Technique) is an over-sampling method that creates new (synthetic) samples based on the samples in our minority classes. It finds the k-nearest-neighbors of each member of the minority classes. The new samples should be generated only in the training set to ensure our model generalizes well to unseen data. We used imblearn python package. Using SMOTE gave us better recall results which is a general goal for customer churning tasks.

## 4. Load models to display predictions on app

Finally, using Flask and HTML/CSS, we created the user-facing app to add information to our data set matching our initial dataframe to predict the likelihood of a customer departing the bank. This was then deployed to Heroku. In November 2022 it was moved to Render and it can be found at this URL: [https://bank-churn-predictions.onrender.com](https://bank-churn-predictions.onrender.com)

## References

- [Kaggle - Churn Modelling Calssification Data Set](https://www.kaggle.com/shrutimechlearn/churn-modelling)
- https://github.com/zunicd/T2D-Predictions
- [How to save a scikit-learn pipline with keras regressor inside to disk?](https://stackoverflow.com/questions/37984304/how-to-save-a-scikit-learn-pipline-with-keras-regressor-inside-to-disk)
- [Problem with serializing and restoring scikit-learn pipelines](https://rebeccabilbro.github.io/module-main-has-no-attribute/)
- [Edit seaborn legend](https://stackoverflow.com/questions/45201514/edit-seaborn-legend)
- [How to Easily Deploy Machine Learning Models Using Flask](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4)
- [Keras Hyperparameter Tuning using Sklearn Pipelines & Grid Search](https://medium.com/@am.benatmane/keras-hyperparameter-tuning-using-sklearn-pipelines-grid-search-with-cross-validation-ccfc74b0ce9f)
- [SciKeras Documentation](https://www.adriangb.com/scikeras/stable/index.html)
- [Scikit-Learn Tutorial: Machine Learning in Python Examples](https://www.guru99.com/scikit-learn-tutorial.html)



For a deeper dive into our analysis and process, take a look at [our full presentation](https://docs.google.com/presentation/d/1AiL9Zo9WBySLQkjjMcIHuZp4wEC9UnLtXjSaclaICBE/edit?usp=sharing).

 