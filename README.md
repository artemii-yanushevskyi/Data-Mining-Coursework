# Data Mining at Aston University Course

KDD (Knowledge Discovery in Databases) Process

1. Develop an understanding of the domain
2. Create target data set
3. Data cleaning and preprocessing, reduction and projection
4. Method selection (classification, clustering, association analysis)
5. Extract patterns, models
6. Interpretation
7. Consolidating knowledge

Data


Object | Attribute 1 | Atribute 2 | Atribute 3
--- | --- | --- | ---
Object 1 | Attribute value 1 for object 1 | Attribute value 2 for object 1 | Attribute value 3 for object 1
Object 2 | Attribute value 1 for object 2 | Attribute value 2 for object 2 | Attribute value 3 for object 2
Object 3 | Attribute value 1 for object 3 | Attribute value 2 for object 3 | Attribute value 3 for object 3
Object 4 | Attribute value 1 for object 4 | Attribute value 2 for object 4 | Attribute value 3 for object 4


Attribute: variable, field, characteristic, feature

Objects: record, case, observation, entity, instance

Types of attributes:

* Nominal/Categorical
    * ```{juice, beer, soda, …}```
    * Names, Labels
    * Eye color
* Ordinal
    * Energy efficiency ```{C, B, A, A+, A++}```
    * ```{bad, average, above average, good}```
    * ```{hot > mild > cool}```
* Interval
    * Temperatures
    * Dates, times
* Ratio
    * Distance
    * Real numbers


Discrete | continuos
--- | ---
Countable | Infinite
Usually Integers | Real numbers
Zip codes, set of words, binary | Hight, weight, temperature


# Data mining tasks

Classification

* Predict the class of an object based on its features

Regression

* Estimate the value for unknown variable based for an object on its attributes

Clustering

* Unite similar objects in subgroups (clusters)

Association Rule Discovery

* What things go together

Outlier/Anomaly detection 

* Detect significant deviations from normal behavior

# Data Exploration

> Frequency(attribute value) = proportion of time the value occurs in the data set

> Mode(attribute value) = most frequent attribute values


# Percentiles 

An ordinal or continuous attribute x
A number p between 0 and 100

> The p-th percentile is a value $x_p$ of x such that p% of the observed values of x are smaller than $x_p$.


> Mean(attribute) = sum(attribute values)/m

> Median(attribute) = value in the middle of observations, or average of 2 values in the middle
(Trimmed mean)

> Range(attribute) = difference between the largest and the smallest

> Variance(attribute) = $s^2_x$ = sum(attribute values - mean)$^2$/(m - 1)

# Visualisation Techniques

## Histograms
Distribution of attribute values

How much values fall into the bin of size 10 (or 20).

## Box plots
￼
## Scatter plots

# Data quality
Noise, Outliers, Missing values, Duplicate data

# Data preprocessing

Sampling

* Without replacement
    * Each time item is selected it is removed
* With replacement
    * Non removed
    * One object can be picked more than once
 

Dimensionality reduction
* Less resources needed
* Easy visualize
* Eliminate irrelevant features and reduce noise

* Feature elimination
* Feature extraction: PCA

## PCA
Linear combinations of the original attributes.
Ordered in decreasing amount of variance explained.
Orthonormal (orthogonal with unit norm), independent.
Not easily interpretable.

## Attribute transformation
Apply a function to the attribute values
x^k, log(x), sqrt(x)

## Standardisation
Replace each original attribute by a scaled version of the attribute

Scale all data in the range [0,1] or [-1,1]

$$ x’ = x - x_min / x_max - x_min $$

$$ x’ = -1 + 2 x - x_min / x_max - x_min $$ 

## Normalisation
Zero mean and unit variance

$$ x’ = x - mean(x)/s $$ 

$$ s = 1/(N-1) sum (x - mean)^2 $$

## Similarity and Dissimilarity
* Euclidian distance
* Norm

Binary vectors
x and y have binary attributes
M_ab = Number of attributes where x has value a\in {0,1} and y has value b\in {0,1}

* Simple Matching Coefficient

$$ SMC = M00 + M11 / M00 + M01 + M10 + M11 $$

* Jacard Similarity Coefficient

$$ J = M11 / M01 + M10 + M11 $$ 

* Cosine similarity

$$ sim(d1, d2) = cos(< d1, d2) = d1^T • d2/ norm(d1)norm(d2) $$

* Covariance matrix (for features??)

$$ ∑ = [[s11, s12], [s21, s22]]$$ 

$$ s_ij = cov(x_i, x_j) = (x_i -mean(x_i))^T • (x_j - mean(x_j)) / N-1 $$

* Correlation (for features??)

$$ \rho _ij = s_ij/s_i*s_j $$

correlation ≠ causation, look for 3rd variable

$x_i$ and $x_j$ are features

Gower’s similarity index
(For objects)


$$ sim_G(x,y) = 1/M ∑sim(x_i,y_i) $$

here x, y are objects 

$$ sim(x_i, y_i) = 1 if x_i = y_i and 0 otherwise $$

For interval/ratio

sim(x_i, y_i) = 1 - |x_i - y_i|/R_i
Where R_i is the range of i-th attribute in the data.

??

d_G(x,y) = 1 - sim_G(x, y)



# To Learn

```Keras
Naïve Bayes
Decision tree
Statistics 
Labs answers, md, git
Dim reduction , pca 
R visualisation, tutorials 
Data types
Data similarity, types, covariant
Covariance and dependence 
```