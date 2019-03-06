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

> The p-th percentile is a value x_p of x such that p% of the observed values of x are smaller than x_p.


> Mean(attribute) = sum(attribute values)/m 

> Median(attribute) = value in the middle of observations, or average of 2 values in the middle
(Trimmed mean)

> Range(attribute) = difference between the largest and the smallest

Variance(attribute) = s^2_x = sum(attribute values - mean)^2/(m - 1)

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

$$ x'= \frac{x - x_{min}}{ x_{max} - x_{min} } $$

$$ x'= -1 + 2\ \frac{x - x_{min}}{x_{max} - x_{min} } $$ 

## Normalisation
Zero mean and unit variance

$$ x' = \frac{x - mean(x)}s  $$ ,

where

$$s = \frac1{(N-1)}\sum(x - mean(x))^2  $$

## Similarity and Dissimilarity
### Euclidian distance
### Norm

Binary vectors
x and y have binary attributes
M_ab = Number of attributes where x has value a\in {0,1} and y has value b\in {0,1}

### Simple Matching Coefficient

$$SMC = (M00 + M11 )/( M00 + M01 + M10 + M11 ) $$

### Jacard Similarity Coefficient

$$J = M11 / (M01 + M10 + M11) $$ 

### Cosine similarity

$$ sim (d_1 ,d_2) = \cos(d_1, d_2) = \frac{d1^T • d2}{norm(d_1)norm(d_2)}$$

### Covariance matrix (for features??)

$$\sum = [[s_{11}, s_{12}], [s_{21}, s_{22}]]  $$ 

$$ s_{ij} = cov(x_i, x_j) = \frac{(x_i - mean(x_i))^T \cdot (x_j - mean(x_j))}{N-1} $$

### Correlation (for features??)

$$\rho _{ij} = \frac{s_{ij}}{s_{i}*s_{j}}  $$

correlation ≠ causation, look for 3rd variable

x_i and x_j are features

Gower’s similarity index
(For objects)


$$ sim_G(x,y) = \frac{1}{M} \sum sim(x_i,y_i)  $$

here x, y are objects 

$$ sim(x_i, y_i) = 1\text{ if }x_i = y_i\text{ and }0\text{ otherwise } $$

For interval/ratio

$$ sim(x_i, y_i) = 1 - \frac{|x_i - y_i|}{R_i}  $$ 

Where R_i is the range of i-th attribute in the data.

??

$$d_G(x,y) = 1 - sim_G(x, y) $$

# ...
In linear regression: betas are how influential are attributes

$$r^2\text{ -- how well model captures variation}=\frac{\text{variation accounted}}{\text{total variation}} = 1 - \frac{SS_E}{SS_T} $$

$$SS_T = \text{total spread }\sum(y_i - mean(y)) $$

Higher r^2 the better (we can trick it by not including irrelevant variables, that is why we need r_adj^2

r2 is how much (*100% percents) is explained by the model



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