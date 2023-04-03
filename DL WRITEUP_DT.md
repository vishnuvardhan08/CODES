## Mastering Decision Tree and Random Forest: Techniques, Hyperparameters, and Best Practices

## **DESICION TREE**

**Decision trees** are one of the most widely used algorithms in machine learning for classification and regression tasks. It is a tree-like model of decisions and their possible consequences. Each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label or a numerical value.

A decision tree looks like a flowchart or an inverted tree. It grows from root to leaf but in an upside down manner. We can easily interpret the decision making /prediction process of a decision tree, due to it’s structure.

![image.png](attachment:image.png)

The starting node is called as the **Root Node**.

It splits further by making a decision based on some criterion into what are called as **Internal Nodes**. The internal nodes that split further are called as **Parent nodes** and the nodes formed out of parent nodes are called as **Child nodes**. 

The nodes that do not split further are called **Leaf Nodes**.

A Decision Tree can be a **Classification Tree** or a **Regression Tree**, based upon the type of target variable. The class in case of classification tree is based upon the majority prediction in leaf nodes. In case of regression, the final predicted value is based upon the average values in the leaf nodes.

During **model training** on feature-target relationships, a tree is grown from a root (parent) node (all data containing feature-target relationships), which is then recursively split into child nodes (subset of the entire data) in a binary fashion. 

Each split is performed on a single feature in the parent node, at a desired **threshold value** of the feature. For instance, during each split of the parent node, we go to left node (with the corresponding subset of data) if a feature is less than the threshold, and right node otherwise. 

## But how do we decide on the split? 

### 1 Entropy, Information Gain, Gini Impurity
#### Entropy, Information Gain, and Gini Impurity are three common metrics used to evaluate which feature to split on at each node of a decision tree.

#### **Entropy** 
Entropy is a measure of impurity or disorder in a set of examples. It is given by the following formula

$$H(S) = -\sum_{i=1}^{c} p_i \log_2 p_i$$

where $S$ is a set of examples, $c$ is the number of classes, and $p_i$ is the proportion of examples in class $i$.

#### **Information gain**
Information gain is a measure of the reduction in entropy achieved by splitting a dataset on a given feature. It is defined as:

$$IG(S, F) = H(S) - \sum_{f \in F} \frac{|S_f|}{|S|} H(S_f)$$

where $S$ is a set of examples, $F$ is a set of features, $S_f$ is the subset of examples where feature $f$ has value $v$, and $H(S_f)$ is the entropy of subset $S_f$.

#### **Gini impurity**
The **cost function** that determines the split of the parent node is called the Gini Impurity, which is basically a concept to quantify how homogeneous or “pure” a node is, with relation to the distribution of the targets in the node. 

It is calculated by **summing the squared probabilities** of each outcome in a distribution and subtracting the result from 1.
A node is considered pure (G=0) if all training samples in the node belong to the same class, while a node with many training samples from many different classes will have a Gini Impurity close to 1.

A lower Gini Index indicates a more homogeneous or pure distribution, while a higher Gini Index indicates a more heterogeneous or impure distribution

The Decision Tree grows in the direction of decreasing Gini Impurity, and the root node is the most impure. Hence, at each node the optimum feature, and the corresponding optimum threshold value, are searched and selected, such that the weighted average Gini Impurity of the 2 child nodes is the least possible.

$$G(S) = 1 - \sum_{i=1}^{c} p_i^2$$

where $S$ is a set of examples, $c$ is the number of classes, and $p_i$ is the proportion of examples in class $i$.

One of the advantages of using Gini Impurity is that it is less computationally intensive compared to Entropy. However, it tends to be less sensitive to changes in the dataset and may not always result in the most optimal split.

Overall, Gini Impurity is a useful measure for evaluating the quality of a split in a decision tree and can help us in building accurate models.

###    EXAMPLE
Here’s an example: Suppose we have a dataset with 10 examples that we want to split based on whether they are red or blue. If 6 are red and 4 are blue, then:

**Entropy** = -0.6 log2(0.6) - 0.4 log2(0.4) = 0.971

**Gini impurity** = 1 - (0.6^2 + 0.4^2) = 0.48

**Information gain** for splitting on color = 0.971 - (6/10 * (-3/5 log2(3/5) - 2/5 log2(2/5))) - (4/10 * (-1/4 log2(1/4) - 3/4 log2(3/4))) = 0.171

### 2. Working For Categorical and Numerical Features

For **categorical features**, the tree splits the data based on the different values of the **feature**.

For **example**, consider a dataset of animals with features such as type (mammal, bird, reptile), habitat (terrestrial, aquatic, arboreal), and diet (carnivore, herbivore, omnivore). The decision tree algorithm would split the data based on the values of each feature. The tree could split the data based on type, separating out mammals, birds, and reptiles into different branches. Then each of these branches could be further split based on habitat, separating out terrestrial and arboreal animals from aquatic ones. Finally, the tree could split the data based on diet, separating out carnivores, herbivores, and omnivores.

For **numerical features**, the decision tree algorithm will split the data based on a **threshold value**. 

For **example**, consider a dataset of housing prices with a feature of square footage. The decision tree algorithm would select a threshold value, such as 1500 square feet, and split the data into two branches. The branch with square footage values less than or equal to 1500 would be on one side of the split, and the branch with square footage values greater than 1500 would be on the other side. This process is repeated recursively for each resulting branch until a stopping criterion is met.

The decision tree algorithm can also handle datasets with both categorical and numerical features. In this case, the algorithm will first split the data based on the categorical features and then further split the data based on the numerical features within each resulting branch.

### 3. Overfitting

**Overfitting** is a common problem in decision trees, where the model is too complex and fits the training data too closely, resulting in poor generalization and high error rates on new, unseen data.

**Overfitting** in decision trees can occur when the tree is grown too deep, with too many branches and leaves, and can easily happen when there are too many features or the tree is not pruned properly. A complex decision tree that is too deep can perfectly fit the training data, but fail to generalize to new data.

For example, consider a decision tree that is used to classify handwritten digits. The decision tree is trained on a dataset of handwritten digit images, where each image has 784 pixels (28x28) representing the grayscale intensity of each pixel. The decision tree is trained to predict the correct digit from the image.

If the decision tree is allowed to grow too deep, it may start to overfit the training data, resulting in poor generalization and high error rates on new, unseen images. This is because the decision tree is learning to classify based on specific features and patterns in the training data, which may not be present in new, unseen images.


To prevent overfitting, several techniques can be used, such as:

==> Setting a maximum depth for the tree

==> Setting a minimum number of samples required to split a node

==> Setting a minimum number of samples required to be at a leaf node

==> Pruning the tree after it has been built






### 4. Hyperparameter Techniques

Hyperparameter techniques are used to tune the decision tree model to improve its performance. Hyperparameters are the parameters that are not learned from data, but rather set prior to model training. Some common hyperparameters that can be tuned in a decision tree model include:

1)**Maximum depth**(max_depth): This hyperparameter limits the maximum depth of the decision tree. A deeper tree may fit the training data more closely, but can lead to overfitting.

2)**Minimum samples split**(min_samples_split): This hyperparameter specifies the minimum number of samples required to split an internal node. Increasing this value can prevent the tree from creating too many branches, which can also lead to overfitting.

3)**Minimum samples leaf**(min_samples_leaf): This hyperparameter specifies the minimum number of samples required to be at a leaf node. If a split results in a leaf node with fewer samples than this value, the split is not allowed. Increasing this value can help prevent overfitting by stopping the tree from creating leaves with very few samples.

4)**Maximum number of features**(max_features): This hyperparameter specifies the maximum number of features to consider when looking for the best split. This can be used to reduce the impact of irrelevant features on the model.

5)**Criterion**: This hyperparameter specifies the function to measure the quality of a split. The two most common options are Gini impurity and information gain (entropy).

Hyperparameter tuning can be done using a **validation set** or by using **cross-validation**. By trying out different combinations of hyperparameters and evaluating their performance on the validation set, we can find the best set of hyperparameters to use for the final model.

### 5. Impact of Outliers and missing values

Decision trees can be sensitive to **outliers and missing values**, which can significantly affect the accuracy and reliability of the model.

#### **Outliers**

**Outliers** are data points that lie far away from the other observations in the dataset. These can occur due to measurement errors, data entry errors, or due to the natural variability of the data. Outliers can have a significant impact on the structure of a decision tree, as the algorithm tries to create splits based on the most significant feature to minimize the impurity in the resulting nodes. Outliers can distort the structure of the tree and result in overfitting, as the tree tries to capture the noise in the data instead of the underlying patterns.

For **example**, let's consider a dataset of **housing prices** in a particular city, where the features include the number of bedrooms, square footage, and the location of the house. The target variable is the price of the house. Suppose there is an outlier in the dataset where a house with only one bedroom is listed at an extremely high price. This outlier could significantly impact the structure of the decision tree, as it would create a split based on the number of bedrooms that may not be representative of the overall pattern in the data.

#### **Missing values**

**Missing values**, on the other hand, are data points where one or more features are not available. This can occur due to data entry errors, data preprocessing errors, or due to the nature of the data itself. Missing values can also affect the structure of a decision tree, as the algorithm tries to create splits based on the available features to minimize the impurity in the resulting nodes.

For **example**, let's consider a dataset of **customer demographics** and their purchasing behavior. The features include age, gender, income, and the type of product purchased. Suppose there are missing values for the income feature for some of the customers. This missing data could affect the structure of the decision tree, as the algorithm may create splits based on the available features, such as age and gender, instead of income, which may be the most significant predictor of purchasing behavior.

#### Handling outliers and missing values

To handle outliers and missing values in decision trees, one approach is to preprocess the data before training the model. Outliers can be identified and removed or transformed using techniques such as **Winsorizing**, where the extreme values are replaced with the maximum or minimum value in the dataset.

Missing values can be imputed using techniques such as **mean imputation**, where the missing values are replaced with the mean value of the feature in the dataset. Alternatively, decision trees can be adapted to handle missing values by treating them as a separate category or by assigning them to the most frequent value in the dataset.

## **RANDOM FOREST**

**Random Forest** is an ensemble learning method used for both classification and regression problems. It builds a multitude of decision trees and then aggregates their predictions to improve the accuracy and avoid overfitting.Each Decision Tree is trained on a random subset of the data and a random subset of the input variables. The final prediction is made by taking the average of the predictions of all the Decision Trees.

Step 1: Select n random subsets from the training set

Step 2: Train n decision trees

one random subset is used to train one decision tree
the optimal splits for each decision tree are based on a random subset of features 

Step 3: Each individual tree predicts the records/candidates in the test set, independently.

Step 4: Make the final prediction

For each candidate in the test set, Random Forest uses the class (e.g. cat or dog) with the majority vote as this candidate’s final prediction.

### 1. Ensemble Techniques(Boosting And Bagging)
**Ensemble learning**, in general, is a model that makes predictions based on a number of different models. By combining individual models, the ensemble model tends to be more flexible (less bias) and less data-sensitive (less variance).

Two most popular ensemble methods are bagging and boosting.

a. **Boosting**: Boosting is a technique that works by iteratively improving weak models to create a strong model. In boosting, each model is trained on the same dataset, but the weights of the observations are adjusted based on the errors made by the previous model. The final prediction is made by combining the predictions of all the weak models.AdaBoost (Adaptive Boosting) is a popular boosting algorithm used in random forests.

Boosting:
The predicted output of a single tree in a boosted random forest is given by:

$$\hat{y}{i}=\sum_{b=1}^{B} \gamma f_{b}(x_i)$$

where,

$\hat{y}_{i}$ is the predicted output for the i-th data point.

$f_{b}(x_i)$ is the predicted output of the b-th tree for the i-th data point.

$\gamma$ is the learning rate or step size.

B is the number of trees in the forest.


b. **Bagging**: In bagging, multiple models are trained independently on different subsets of the data. The final model is an average of all models. Random Forest uses bagging to build multiple decision trees.Random forest is an ensemble model using bagging as the ensemble method and decision tree as the individual model.

The predicted output of a single tree in a random forest is given by:

$$\hat{y_i}=\frac{1}{B} \sum_{b=1}^{B} f_{b}(x_i)$$

where,

$\hat{y}_{i}$ is the predicted output for the i-th data point.

$f_{b}(x_i)$ is the predicted output of the b-th tree for the i-th data point.

B is the number of trees in the forest.

![image.png](attachment:image.png)

#### EXAMPLES OF BOOSTING : ADA BOOSTING AND GRADIENT BOOSTING

**AdaBoost** is a boosting ensemble model and works especially well with the decision tree. Boosting model’s key is learning from the previous mistakes, e.g. misclassification data points.

AdaBoost learns from the mistakes by increasing the weight of misclassified data points.

**Gradient boosting** is another boosting model. Remember, boosting model’s key is learning from the previous mistakes.

Gradient Boosting learns from the mistake — residual error directly, rather than update the weights of data points.

### 2. Working as Classifier and Regresor

**Random Forest** is a versatile algorithm that can work as both a classifier and regressor. In classification tasks, it predicts the class label of the input data point while in regression tasks, it predicts a continuous numerical value.

#### **CLASSIFICATION**

In the case of **classification**, the final prediction is based on the majority vote of the individual decision trees. Each tree in the random forest predicts a class label based on the features of the input data point. The predicted class labels from all trees are counted, and the class with the most number of votes is selected as the final predicted class label.

For **example**, let's consider a random forest with 100 decision trees that are trained on a **dataset of images with different objects**. Each decision tree in the random forest will predict the object class based on the features of the input image, such as shape, color, texture, etc. The final predicted object class will be the class with the most number of votes from all the individual decision trees.

#### **REGRESSION**

In the case of **regression**, the final prediction is the average of the predicted values from all the decision trees. Each decision tree in the random forest predicts a numerical value based on the features of the input data point. The predicted values from all the trees are averaged, and the average value is selected as the final predicted value.

For **example**, let's consider a random forest with 100 decision trees that are trained on a dataset of **housing prices**. Each decision tree in the random forest will predict the price of the house based on the features of the input data point, such as the number of bedrooms, the size of the house, the location, etc. The final predicted price of the house will be the average of the predicted prices from all the individual decision trees.

## 3. Hyperparameter Tuning(Grid Search And RandomSearch)

**Random Forest** has several hyperparameters that need to be tuned for optimal performance. **Hyperparameters** are the parameters of the model that are not learned during the training process, but rather are set before the training process begins. These hyperparameters can significantly impact the performance of the Random Forest model.

There are several ways to tune the hyperparameters of a Random Forest model, but two common methods are Grid Search and Random Search.

#### **Grid Search**

**Grid Search**: This method involves creating a grid of possible values for each hyperparameter and then training the model on each possible combination of hyperparameters. The combination that performs the best on the validation set is selected as the optimal hyperparameters.

For example, consider the hyperparameters of the Random Forest model: number of trees, maximum depth of each tree, minimum number of samples required to split an internal node, minimum number of samples required to be a leaf node, and the criterion used to evaluate the quality of a split. We can create a grid of possible values for each hyperparameter and then train the model on each possible combination of hyperparameters to find the optimal combination.

#### **Random Search**

**Random Search**: This method randomly samples the hyperparameters from a specified distribution. It then trains the model on each set of hyperparameters and selects the combination that performs the best on the validation set. This method is often more efficient than Grid Search because it does not need to evaluate every possible combination of hyperparameters.

For example, suppose we want to tune the number of trees hyperparameter. We could specify a uniform distribution between 100 and 1000 for the number of trees hyperparameter, and the Random Search algorithm would randomly sample values from this distribution and evaluate the model's performance for each set of hyperparameters.

Hyperparameter tuning is an essential step in building a robust and accurate Random Forest model. It allows us to find the optimal set of hyperparameters for our specific problem, which can significantly improve the model's performance.

![image.png](attachment:image-2.png)

#### **Grid Search** VS **Random Search**

These two strategies can be compared in terms of dimensionality.

With grid search, the greater the dimensionality, the greater the number of hyperparameter combinations to search for. As such, there is a greater chance of grid search being impractical.

The time taken to search would not justify the use of grid search. The computational resources in use would also prove unfeasible with an increase in the number of parameters.

Each additional parameter would increase the number of evaluations exponentially. With a smaller number of hyperparameters, grid search may edge out the random search.

This is because grid search would guarantee accuracy by exhausting all possible combinations. Similar to grid search, the higher the dimensionality, the greater the time taken to find the right set of hyperparameters. Higher dimensionality also means a greater number of iterations.

Nonetheless, the random search may offer a greater chance of realizing the optimal parameters. Even though random search may not be as accurate as grid search, we also get to control the number of combinations to attempt.

The random search model may be trained on the optimized parameters within a much shorter time than when using grid search. This also results in much more efficient computational power used in comparison to grid search.

### Pros and cons of decision tree and random forest

Decision tree and random forest are both popular machine learning algorithms used for classification and regression tasks. While they have many similarities, they also have some key differences. Here are some pros and cons of each:

#### Decision Tree:

*Pros*:

Easy to understand and interpret: Decision trees are easy to visualize and understand, as they are represented in a tree-like structure.

Non-parametric: Decision trees do not assume any particular distribution of the data or linear relationship between variables, making them versatile.

Can handle both categorical and numerical data: Decision trees can handle both types of data, making them a good choice for a wide range of problems.

Can handle missing data: Decision trees can handle missing data by ignoring the missing values during the decision-making process.

*Cons*:

Prone to overfitting: Decision trees can easily overfit the training data, leading to poor performance on new data.

Unstable: Small changes in the training data can result in a very different decision tree.

Limited accuracy: Decision trees are often not as accurate as other machine learning algorithms, such as neural networks or random forests.

Can be biased towards features with more levels: Decision trees tend to favor features with many levels, which can lead to bias towards those features.

#### Random Forest:

*Pros* :

High accuracy: Random forests often have higher accuracy than decision trees due to the ensemble approach, which averages the predictions of multiple decision trees.

Less prone to overfitting: Random forests are less prone to overfitting than decision trees because of the ensemble approach.
Can handle high-dimensional data: Random forests can handle high-dimensional data, making them useful in many real-world applications.

Feature importance: Random forests can provide a measure of feature importance, which can be useful in feature selection.

*Cons*:

Can be computationally expensive: Random forests can be computationally expensive, especially for large datasets or a large number of trees in the ensemble.

Difficult to interpret: Random forests are harder to interpret than decision trees because they involve an ensemble of trees.
Biased towards features with more levels: Random forests can also be biased towards features with many levels, like decision trees.

Can be sensitive to noisy data: Random forests can be sensitive to noisy data, which can affect their accuracy.


