# Porto-Seguro-Safe-Driver-Prediction
## Project Overview:

![drawing](https://user-images.githubusercontent.com/24396564/34592072-68aa1586-f176-11e7-9f56-62783e87b53f.png)

Supervised learning is the machine learning task of inferring a function from labeled training data.
The training data consist of a set of training examples. In supervised learning, each example is a
pair consisting of an input object (typically a vector) and a desired output value (also called the
supervisory signal). A supervised learning algorithm analyzes the training data and produces an
inferred function, which can be used for mapping new examples. An optimal scenario will allow
for the algorithm to correctly determine the class labels for unseen instances. This requires the
learning algorithm to generalize from the training data to unseen situations in a "reasonable"
way.
Nowadays, machine learning has been used vastly in insurance companies in order to
recommend decent contracts to their clients based on the past information of them. For
example, buying a new car and its insurance need a precise evaluation on both side client and
insurance company to make a decent contract. Nothing ruins the thrill of buying a brand new car
more quickly than seeing your new insurance bill. The sting’s even more painful when you know
you’re a good driver. It doesn’t seem fair that you have to pay so much if you’ve been cautious
on the road for years. Therefore, having a robust predictive model is a key factor for successful
companies.
Fig. 1 shows the road map of the project. The project includes four main parts: data analysis,
feature engineering, modeling and prediction. In the data analysis part, we applied several
statistical techniques to investigate the data in order to get better understanding about our data
and prepare a good sample for our analysis. In the feature engineering section, we will deal with
missing data, creating interaction variables, making new features by frequency encoding and
binary encoding, calculating feature importance and finally select the importance feature and to
reduce the size of the features. In the modeling part we will use several machine learning
algorithms and stack them in three levels to make a prediction. Finally, by the prediction is made
of the mean of the two last prediction models.
The main file that we can run the project is: project.ipynb

## Problem- Statement
Porto Seguro, one of Brazil’s largest auto and homeowner insurance companies. Inaccuracies in
car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce
the price for bad ones. The goal of this project is to build a model that predicts the probability
that a driver will initiate an auto insurance claim in the next year.
An accurate prediction will allow them to further tailor their prices, and hopefully make auto
insurance coverage more accessible to more drivers. Thus, I will predict the probability that an
auto insurance policy holder files a claim. The dataset is available dataset.
