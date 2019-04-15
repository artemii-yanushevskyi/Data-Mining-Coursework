# Abstract

This report demonstrates the full process of developing a data mining solution for marketing campaign dataset, including critical insights and limitation of the solution, and how it could be improved.

As stated in the coursework specification for the solution, the goal of our first model is to predict, as accurately as possible, whether or not a client will subscribe to a term deposit. The goal of the second model for cost-sensitive classification is to make the total cost as small as possible. I apply multiple techniques like Feature Engieering, Parameter Tuning, and Boosting. Alongside the reasons why it should improve the prediction accuracy for equal cost classification and decrease the cost value for cost-sensitive classification.


# Introduction


We are given with the dataset about marketing campaigns that were based on phone calls. Each entry in the dataset corresponds to someone who has subscribed to a bank term deposit or not. Often more than one contact to the same client was required, in order to confirm if the product would (or not) be subscribed to.

The tools that I will use to develop the models are *Python* programming language, equipped with popular data mining library *scikit-learn*, dataframe manipulation library *Pandas*, mathematical library *NumPy*, and visualisation libraries *Plotly* and *Matplotlib*.

# Data Exploration

The dataset was supplied as an `arrf` file. The common statistics for attributes are the following

// common statisics

We convert attribute values month to the number (i.e. 'may' to 5) so that the ordering would have *chronological* sense (unlike alphabetical).

The target attribute `termDeposit` is 'no' approximately 88.4%. That means that this is the baseline accuracy for classification. Now the baseline cost is for cost-sensitive classification is the minimum of (36169-31981) * 10 = 41880 ('no' for all) and 31981 * 1 = 31981 ('yes' for all). The baseline cost is 31981 for all 'yes' classifier. 

Notice that the majority of _yes_'s are at the end of the data. We may want to shuffle rows. Also, we may use this insight to create a new feature. From 34,000 row until the end the half is 'yes'. However, this feature will not be useful since the test dataset doesn't have the same size.

The *feature attributes* are used to predict the *target attribute*. Originally the dataset has 16 feature attributes, 8 _categorical_ (including month) and 8 _numerical_. The quick check shows that there are no missing values in 36k labelled examples. 

// Histograms

Viewing the histograms for each variable showed that there were no variables that were strongly predictive of the class. Some attributes have quite imbalanced distributions of its values. Our algorithms may benefit from the Feature Engineering of some attribute values. 

// Pairplot

Two-dimensional scatter plots don’t show strong class separation; this suggests that several attributes will be needed to separate the two classes.


Job some job attribute values may need to be merged into new groups. 
Some attributes may be irrelevant/redundant. The further exploration is needed to determine the significance of the attributes. The Pearson correlation shows how attributes depend on each other.

In order to see how categorical attributes are influencing `termDeposit` we need to *one-hot-encode* them.

// Pearson correlation

Notice that `duration` and `poutcome_success` are good predictors among others. The student or retired job status has high significance for `termDeposit` being 'yes'.

Existing credits and duration: numeric attributes with a very strongly skewed distribution. These strongly non-Gaussian distributions may affect the naive Bayes model (which fits a Gaussian distribution to numeric variables) so results for this model may be improved by processing the variables.

<!-- So far our analysis showed that the predictabilit

Attribute selection techniques were applied to th -->

<!-- A number of machine learning algorithms were applied to the data and experiments to optimise the meta-parameters were performed. -->

# Feature engineering

Having done preliminary inspection of the features, we will be prosess the features to make them more suitable for the classification algorithms. 

## Job attribute

There are many different attribute values. It will negatively affect our prediction. We will group them by the percentage of 'yeses'.

Sorting attributes by the success rate

// sort jobs by %

So the groups are

// groups


The union of the jobs in such groups is not arbitrary, I chose to group jobs that have very close success rate. Thus the future models would be trained on lagrer groups, it will boost accuracy for the jobs in minority.

<!-- Can't phrase it better -->

We have created new columns in our data

// group names



## Age


Notice that after 62 yo we see that subscription rate is constant being about 50%. Also, as we can see from histogram, the number of entries with age above 60 drops significantly.

So, for age that is above 66, we will set it to be equal to 67. Now the distribution of age reminds Gaussian distribution even more.

// how we do it

education, marital, default, housing, loan, contact, day, month, poutcome
duration, campaign, Pdays, previous,

## Day and Month

I created the new attribute `date`. It is defined as date via `day` and `month` attributes. The success rate by date

// success rate by date

While exploring `month` attribute, I noticed that the success percentage is inversely proportional to the percentage of entries for this month.

// month inv

It turns out that the same holds true even for the days. The more calls are done during the day, the lesser success rate becomes. It may be because the callers are focussing on the number of calls rather than persuading the audience to make term deposit.

// days inv

It would be really nice if we have had the year. This way we can see how success rate depends on the day of the week.

We don't need to preserve the `day` attribute, since it has become redundant.


## Balance

This attribute is the most confusing. From the histogram we can infer that it is quite unlikely that for 10 present the balance attribute is between -8 and 8 pounds. It is either because their actual balance is different since they may use other means to save money. As a result the balance is not a good predictor for target attribute.

### Replacing
I will try to make balance to be more stronger estimation of the wealth by replacing the value between -8 and 8 by its estimation from more stronger predictors, such as: `education`, `marital`, `default`, `housing`, `loan`, `contact`, and `poutcome`.

### Logarithm
The second reason why this will be beneficial is because the distribution is lognormal, like all human generated data usually is. I will apply function $\log$ to the attribute values to oblatin a distribution close to Gaussian.

Now we can freely apply logarithm to the attribute balance.

// logarithm graph

// transformed balance

## Duration, Campaign, and Previous

I will go with the same approach as with `balance`, `campaign`, and `previous`. Apply the function $\log$ to the values

// log graph for duration

Campaign doesn't become

// log graph for campaign

// log graph for previous


## Pdays

It is not quite clear what strategy to use with pdays attribute.

// graph


## Poutcome

We may consider grouping the poutcome attribute values.. 


## PCA for numerical attributes

The application of PCA for our numerical data had output the folowing result

// PCA

This is strengthtening the conclusion from Pearson correlation matrix that the .. attribute is the most predictive of the 'yes' class.

When applying classification we will use first .. PCA components that explain .. of the data.

## Scaling ...

## Result of Feature Engineering

Now, instead of .. attributes we have .. that should improve the predictive power of the classification algorithms.

# Classification

## Desision trees

The score had increased to ..


## Naive Bayes

We apply .. to categorical and Gaussian Naive Bayes to numerical attributes. Then we use the /voting classifier/ for ...

The achieved accuracy is ..


## Logistic Regression

The achieved accuracy is ..

## KNN

The ...

## Random Forest

# Cost-sensitive classification

I use thresholding approach to achieve cost sensitivity

## Desision trees

The cost is ..


## Naive Bayes

We apply .. to categorical and Gaussian Naive Bayes to numerical attributes. Then we use the /voting classifier/ for ...

The cost is ..


## Logistic Regression

The cost is ..

## KNN

The ...

## Random Fores

The cost is ..



# Improving Accuracy: Ensembles 


# Reducing Cost: Ensembles 



# Comparing the models

The best model for accuracy is ..

The lowest cost model is ..

# Conclusions

## Dataset

There are few suggestions about the given data.


## Model evaluation

The model generalises well for new data ..

* some attributes are transformed in order to have the distribution close to Gaussian. It is actually requirement for some classifier algorithms like Naive Bayes
* the models work quick enough to generate predictions in production/real time on the server
* there is a potential for such models to scale without the need to retrain on the entire dataset
...

The model is not perfect, it may be improved in the following ways

* it depends on the number and time of samples, there is a noticable improvement in the addvertisement team as time goes by
* it could be biased towards favoring days when team does many calls (hoping to achieve higer accuracy)
...