# Machine Learning Challenge
Machine Learning course 2021-2022, Computer science department.<br>
Masters' Degree in Applied Mathematics, 
Sapienza University of Rome. <br>
Exam date: June 13th 2022

### Dataset and task description:
The provided [dataset](https://github.com/SofiaTorchia/ML-challenge/blob/master/OnlineNewsPopularity.csv)
is a modified noisy version of
the original dataset described in [1].<br>
This dataset summarizes a heterogeneous set of
features about articles published by Mashable in 
a period of two years. The goal of the task is to
 predict the number of shares in social networks
(popularity).

### Pre-processing and dataset analysis

The dataset is loaded and cleaned (see the 
[cleaned version](https://github.com/SofiaTorchia/ML-challenge/blob/master/OnlineNewsPopularity_cleaned.csv)). 
In this [notebook](https://github.com/SofiaTorchia/ML-challenge/blob/master/ML_Challenge_21-22.ipynb) 
feature importance analysis in then performed by removing redundant 
features. The data is aferwards scaled to perform model selection.
For this phase the following steps are considered: 

1) The target feature is discretized
(number of classes $=$ 5).

2) The dataset is split to perform cross validation. 

3) The following models are trained: 

- Decision Trees 
- Support Vector Machines
- Random Forest
- MLPNs

4) Hyper-parameter tuning is performed and discussed

<br/><br/>



[1] K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision
    Support System for Predicting the Popularity of Online News. Proceedings
    of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence,
    September, Coimbra, Portugal


