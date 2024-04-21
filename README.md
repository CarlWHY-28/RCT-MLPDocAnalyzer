# RCT_MLPDocAnalyzer Intro

RCT_MLPDocAnalyzer is a paper `analysis tool` to classify whether a paper is a randomized controlled trial based on some of the paper's properties, `RCT`for short. It is designed to reduce the workload of people in classifying papers, and is a small helper based on `machine learning` algorithms.
My application to help a colleague who can't write code solve this problem stems from my new enthusiasm for work XD.
Because of the small amount of data, only 1000 in total, the colleague requires an accuracy rate of more than 90%, and the final accuracy reached 95%.

## Research method

**Information processing**

+ Feature engineering
    + Reading document 
    + Labeling
    + Feature extraction
+ Vectorization and Segmentation
    + Vectorization
        + `CountVectorizer` is wildly used in `Natural Language Processing`, The reason for my choice is that humans use keyword analysis in most cases when judging whether an paper is RCT.
    + Segmentation of datasets


**Machine learning algorithm**

1. Selected method and related parameters
    1. After comparing various methods (decision tree, logistic regression, transformer, etc.), I chose neural network `MLPClassifier` based on the comparison of accuracy
    2. Hidden_layer_sizes: Considering this is a network for small datasets, increasing it doesn't improve performance but raises the possibility of overfitting. Thus, (100,50) is reasonable in this case.
    3. Activation: Oddly, `relu`, which is supposed to be better suited for binary classification problems, `does not perform as well on this problem as identity`. I firmly believe the best activation function can depend on the specific characteristics of your dataset and the problem I am trying to solve.
    4. Solver: lbfgs is better for small datasets
``` python
    # Train the model
    clf = MLPClassifier(hidden_layer_sizes=(100, 50,), activation='identity', solver='lbfgs') 
    clf.fit(X_train, y_train)
```


+ Prediction and Evaluation
    + Printing the accuracy
    + Comparing results to see which predictions are wrong

+ Prediction and Evaluation
    + Model dump
    + Simulations of others using the model
        + My colleague can still use these methods to overcome such problems after I leave the company.

## Result analysis

> Finally, the accuracy of the algorithm reached 95%

> The obvious disadvantage of the word vector approach as a feature analysis is that sometimes the information in the paper may just mention the RCTS of other studies rather than actually using them. And the confusion matrix also reflects that, the misjudgment rate in the case of Yes is much greater than that in the case of No.

![image](https://raw.githubusercontent.com/whyBFU/RCT-MLPDocAnalyzer/main/p1.jpg)
