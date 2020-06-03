# Big-Data

### <p align="center" > TECNOLÓGICO NACIONAL DE MÉXICO <br> INSTITUTO TECNOLÓGICO DE TIJUANA <br> SUBDIRECCIÓN ACADÉMICA <br>DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN <br>PERIODO: Enero - Junio 2020 </p>

###  <p align="center">  Carrera: Ing. En Sistemas Computacionales. <br>Materia:     Datos Masivos (BDD-1704 IF9A    ).</p>
### <p align="center">  Maestro: Jose Christian Romero Hernandez    </p>
### <p align="center">  No. de control y nombre del alumno: <br> 15211275 - Camacho Paniagua Luis Angel <br> 16210585 - Valenzuela Rosales Marco Asael </p>


# Unit 2

## Index
&nbsp;&nbsp;&nbsp;[Practice 1](#practice-1)  
&nbsp;&nbsp;&nbsp;[Practice 2](#practice-2)  
&nbsp;&nbsp;&nbsp;[Practice 3](#practice-3)    
&nbsp;&nbsp;&nbsp;[Practice 4](#practice-4)    
&nbsp;&nbsp;&nbsp;[Practice 5](#practice-5)  
&nbsp;&nbsp;&nbsp;[Practice 6](#practice-6)  
&nbsp;&nbsp;&nbsp;[Practice 7](#practice-7)  
&nbsp;&nbsp;&nbsp;[Practice 8](#practice-8)   
&nbsp;&nbsp;&nbsp;[Homework 1](#Homework-1)   
&nbsp;&nbsp;&nbsp;[Homework 2](#Homework-2)  
&nbsp;&nbsp;&nbsp;[Homework 3](#Homework-3)  
&nbsp;&nbsp;&nbsp;[Exam 1](#Exam-1)	 


### &nbsp;&nbsp;Practice 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
   1. Import the LinearRegression
   2. Use the following code to configure errors
   3. Start a simple Spark Session
  4. Use Spark for the Clean-Ecommerce csv file
   5. Print the schema on the DataFrame
  6. Print an example row from the DataFrame
 7. Transform the data frame so that it takes the form of ("label", "features")
 8. Rename the Yearly Amount Spent column as "label"
 9. The VectorAssembler Object 
10. Use the assembler to transform our DataFrame to two columns: label and features 
11. Create an object for line regression model 
 12. Fit the model for the data and call this model lrModel
 13. Print the coefficients and intercept for the linear regression
 
 
14. Summarize the model on the training set and print the output of some metrics
15. Show the residuals values, the RMSE, the MSE, and also the R^2
 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala   
  1. import org.apache.spark.ml.regression.LinearRegression
 
```   	 
```scala	 
  2.import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```   	 
```scala	
3.import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
```   	 
```scala	 
  4. val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")
 ```   	 
```scala	
5. data.printSchema
 ```   	 
```scala	
6. data.head(1)
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(0, colnames.length)){
   println(colnames(ind))
   println(firstrow(ind))
   println("\n")
}
 ```   	 
```scala	
 
7.   Import VectorAssembler and Vectors:
 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```   	 
```scala	

 
8. val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership", $"Yearly Amount Spent")
 ```   	 
```scala	
9. val new_assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership", "Yearly Amount Spent")).setOutputCol("features")
 ```   	 
```scala	
10. val output = new_assembler.transform(df).select($"label",$"features")
 ```   	 
```scala	
 
11. val lr = new LinearRegression()
 ```   	 
```scala	
 
12.val lrModel = lr.fit(output)
 ```   	 
```scala	
 
13. println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
 ```   	 
```scala	
 
14.val trainingSummary = lrModel.summary
 ```   	 
```scala	
 
15. trainingSummary.residuals.show()
val RMSE = trainingSummary.rootMeanSquaredError
val MSE = scala.math.pow(RMSE, 2.0)
val R2 = trainingSummary.r2 

```










### &nbsp;&nbsp;Practice 2.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
   	 
    	1.-Create a list called "list" with the elements "red", "white", "black"
    	2.-Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
    	3.-Bring the "list" "green", "yellow", "blue" items
    	4.-Create a number array in the 1-1000 range in 5-in-5 steps
    	5.-What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets
    	6.-Create a mutable map called names containing the following"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
    	6.-a. Print all map keys
    	7.-b. Add the following value to the map ("Miguel", 23)
   	 
#### In this practice what we did we created lists with different colors, we added elements to the lists, we created an array with ranges and that they count from 5 to 5, we made a mutable map, we printed them and we added element to that map
   	 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala
     	/*1.-Create a list called "list" with the elements "red", "white", "black"*/
     	var lista = collection.mutable.MutableList("rojo","blanco","negro") 	 
```   	 
```scala
     	/*2.-Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"*/
      	lista += ("verde","amarillo", "azul", "naranja", "perla")

```

```scala
     	/*3.-Bring the "list" "green", "yellow", "blue" items*/
         	lista(3)
         	lista(4)
         	lista(5)

```

```scala
     	/*4.-Create a number array in the 1-1000 range in 5-in-5 steps*/
           	var v = Range(1,1000,5)

```

```scala
     	/*5.-What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets*/
          	var l = List(1,3,3,4,6,7,3,7)
           	l.toSet

```

```scala
     	/*6.-Create a mutable map called names containing the following"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27*/
      	var map=collection.mutable.Map(("Jose", 20),("Luis", 24),("Ana", 23),("Susana", "27"))

```
```scala
     	/*6.-a. Print all map keys*/
       	map.keys

```
``` scala
     	/*7.-b. Add the following value to the map ("Miguel", 23)*/
      	map += ("Miguel"->23)

```

### &nbsp;&nbsp;Practice 3.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

1. Import libraries.
 2. Import a Spark Session.
 3. Create a Spark session.
  4. Load the data stored in LIBSVM format as a DataFrame.
 5. Index labels, adding metadata to the label column.
 6. Automatically identify categorical features, and index them.
7. Split the data into training and test sets.
 8. Train a DecisionTree model. 
9. Convert indexed labels back to original labels.
10. Chain indexers and tree in a Pipeline.
11. Train model.
12. Make predictions.
 13. Select example rows to display.
 14. Select (prediction, true label) and compute test error.
 

 
15. Print the tree obtained from the model


#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

  	 
``` scala
  1. import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
 
```
``` scala
2.import org.apache.spark.sql.SparkSession
 ```
```scala

3. def main(): Unit = {
   val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

 ```
``` scala
4.    val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
```
```scala
 
5. val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
```
``` scala

6. 
 val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
```
```scala
 
7.    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
```
```scala
8.    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
```
```scala 
9.   val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
 ```
```scala
10   val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
 ```
```scala
11.
   val model = pipeline.fit(trainingData)
 ```
```scala
12.    val predictions = model.transform(testData)
 ```
```scala
13.    predictions.select("predictedLabel", "label", "features").show(5)
  ```
```scala
14. 
   val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
   val accuracy = evaluator.evaluate(predictions)
   println(s"Test Error = ${(1.0 - accuracy)}")
 ```
```scala
15.    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
   println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
 ```
 

### &nbsp;&nbsp;Practice 4.

 #### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
1. Import libraries.
 2. Import a Spark Session.
 3. Create a Spark session.
4. Load the data and create a Dataframe.
 5. Index labels
 6. Automatically identify categorical features, and index them.
 7. Split the data
8. Train a RandomForest mode
9. Convert indexed labels.
10. Chain indexers and forest in a Pipeline.
11. Train model.
 12. Make predictions.
13. Select example rows to display.
14. Select (prediction, true label) and compute test error.
15. Print the trees obtained from the model.
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala
1. import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
  ```
```scala
2. import org.apache.spark.sql.SparkSession
  ```
```scala
3.  def main(): Unit = {
   val spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()
 ```
```scala 
4. val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
 ```
```scala 
5.    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
 ```
```scala 
6.    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
  ```
```scala
7.    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
 ```
```scala
8.    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees
 ```
```scala 
9.    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
 ```
```scala 
10.   val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
  ```
```scala
11.    val model = pipeline.fit(trainingData)
  ```
```scala
12.   val predictions = model.transform(testData)
  ```
```scala
13.   predictions.select("predictedLabel", "label", "features").show(5)
  ```
```scala
14.   val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
   val accuracy = evaluator.evaluate(predictions)
   println(s"Test Error = ${(1.0 - accuracy)}")
 ```
```scala 
15.   val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
   println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
 ```



### &nbsp;&nbsp;Practice 5.
#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
1. Import libraries.
2. Import a Spark Session.
3. Use the Error reporting code.
4. Create a Spark session.
5. Load the file and create a DataFrame.
6. Index labels, adding metadata to the label column.
7. Automatically identify categorical features, and index them.
8. Split the data into training and test sets.
 9. Train a GBT model.
10. Convert indexed labels back to original labels.
11. Chain indexers and GBT in a Pipeline.
12. Train model.
13. Make predictions.
 14. Select example rows to display.
15. Select (prediction, true label) and compute test error.
16. Print result of Trees using GBT .
 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala
1.
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
  ```
```scala
2. import org.apache.spark.sql.SparkSession
 ```
```scala
 
3. import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
  ```
```scala
4. def main(): Unit = {
   val spark = SparkSession.builder.appName("GradientBoostedTreeClassifierExample").getOrCreate()
  ```
```scala
5.
   val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
   data.printSchema()

 ```
```scala
 6.   val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
 ```
```scala 
7.      val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
 ```
```scala 
8.  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
 ```
```scala 
9.    val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter

 ```
```scala 
10. val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
 ```
```scala
 
11. val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
 
 ```
```scala
12.  val model = pipeline.fit(trainingData)
 ```
```scala
 
13.  val predictions = model.transform(testData)
 ```
```scala
 
14. predictions.select("predictedLabel", "label", "features").show(5)
 ```
```scala
 
15.    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
 
   val accuracy = evaluator.evaluate(predictions)
   println(s"Test Error = ${1.0 - accuracy}")
  ```
```scala
16.    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
   println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
  ```

### &nbsp;&nbsp;Practice 6.
#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
1. Import libraries and package
2. Import a Spark Session.
3. Load the data stored in LIBSVM format as a DataFrame.
4. Split the data into train and test
5. specify layers for the neural network:
6. create the trainer and set its parameters
7. train the model
8. Compute accuracy on the test set
9. Print result 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
 
```scala
1. 
package org.apache.spark.examples.ml
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
  ```
```scala
2. 
 
import org.apache.spark.sql.SparkSession
  ```
```scala
3.
 
val data = spark.read.format("libsvm")
 .load("data/mllib/sample_multiclass_classification_data.txt")
  ```
```scala
4. 
 
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
  ```
```scala
5. 
val layers = Array[Int](4, 5, 4, 3)
  ```
```scala
6. val trainer = new MultilayerPerceptronClassifier()
 .setLayers(layers)
 .setBlockSize(128)
 .setSeed(1234L)
 .setMaxIter(100)
 ```
```scala 
7. val model = trainer.fit(train)
  ```
```scala
8.
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
 .setMetricName("accuracy")
  ```
```scala
9. println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels
 ```
 
 ### &nbsp;&nbsp;Practice 7.
#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
1. Import libraries and package 
2. Import a Spark Session.
3.Load the data from the file and add it to a variable to train it.
4. Load the data stored in LIBSVM format as a DataFrame.
5. Create an object of type LinearSVC.
6. Fit the model
7. Print result
 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala
1.package org.apache.spark.examples.ml
import org.apache.spark.ml.classification.LinearSVC
  ```
```scala 
2.import org.apache.spark.sql.SparkSession
   ```
```scala


3.val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()
   ```
```scala
4. val training = spark.read.format("libsvm").load("/usr/local/spark-2.3.4-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")
   ```
```scala
5. val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
   ```
```scala
6. val lsvcModel = lsvc.fit(training)
   ```
```scala
7. println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
  ```
### &nbsp;&nbsp;Practice 8.
#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
1. Import libraries.
2. Import a Spark Session.
3. Create a Spark session.
4. Load data file.
5. Generate the train/test split.
6. Instantiate the base classifier
7. Instantiate the One Vs Rest Classifier.
8. Train the multiclass model.
9. Score the model on test data.
10. Obtain evaluator.
11. Compute the classification error on test data.
12. Print result


#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
  
```scala
1.import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
   ```
```scala 
2.import org.apache.spark.sql.SparkSession
    ```
```scala
3. def main(): Unit = {
   val spark = SparkSession.builder.appName("MulticlassClassificationEvaluator").getOrCreate()
   ```
```scala 
4.val inputData = spark.read.format("libsvm")load("data/mllib/sample_multiclass_classification_data.txt")
   ```
```scala 
5.val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
 
   ```
```scala
 
6.val classifier = new LogisticRegression()
.setMaxIter(10)
.setTol(1E-6)
.setFitIntercept(true)
    ```
```scala
7val ovr = new OneVsRest().setClassifier(classifier)
    ```
```scala
8. val ovrModel = ovr.fit(train)
    ```
```scala
9. val predictions = ovrModel.transform(test)
    ```
```scala
10. val evaluator = new MulticlassClassificationEvaluator()
.setMetricName("accuracy")
   ```
```scala 
11.val accuracy = evaluator.evaluate(predictions)
   ```
```scala 
12.println(s"Test Error = ${1 - accuracy}")


```
### &nbsp;&nbsp;Homework 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

                   
Main Types of Machine Learning Algorithms

### &nbsp;&nbsp;&nbsp;&nbsp; Investigation

```
Machine Learning came a long way from a science fiction fancy to a reliable and diverse business tool that amplifies multiple elements of the business operation.
Its influence on business performance may be so significant that the implementation of machine learning algorithms is required to maintain competitiveness in many fields and industries.
The implementation of machine learning into business operations is a strategic step and requires a lot of resources. Therefore, it's important to understand what do you want the ML to do for your particular business and what kind of perks different types of ML algorithms bring to the table.

Machine learning algorithms can divide into three broad categories: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning is useful in cases where a property (label) is available for a specific set of data, but must be predicted for other instances. Unsupervised learning is useful in cases where the challenge is to discover implicit relationships in an unlabelled dataset (elements are not previously assigned). Reinforcement learning falls between these two extremes: there is some form of feedback available for each step or predictive action, but there is no precise label or error message.

SUPERVISED LEARNING

1. Decision trees: A decision tree is a decision support tool that uses a graph or model similar to a decision tree and its possible consequences, including the results of fortuitous events, resource costs, and utility . They have an appearance like this:
Class Session ![Big Data]https://github.com/Camacho18/Big-Data/blob/Unit_2/HomeWorks/1%20arbol.png

From a business decision-making point of view, a decision tree is the minimum number of yes / no questions that one has to ask, to assess the probability of making a correct decision, most of the time. This method allows you to approach the problem in a structured and systematic way to reach a logical conclusion.




2. Naïve Bayes Classification: Naïve Bayes classifiers are a family of simple probabilistic classifiers based on the application of Bayes ‘theorem with strong (Naïve) assumptions of independence between characteristics’. The featured image is the equation - with P (A | B) being posterior probability, P (B | A) being probability, P (A) being class prior probability, and P (B) being prior probability predictor.


3. Ordinary Least Squares Regression: If you've been in contact with statistics, you've probably heard of linear regression before. Ordinary Least Squares Regression is a method of performing linear regression. Linear regression can be thought of as the task of fitting a straight line through a set of points. There are several possible strategies for doing this, and the "ordinary least squares" strategy goes like this: you can draw a line and then, for each of the data points, measure the vertical distance between the point and the line and add them together; The fitted line would be the one in which this sum of distances is as small as possible.
Class Session![BigData]https://github.com/Camacho18/Big-Data/blob/Unit_2/HomeWorks/3%20ordinary.png

4. Logistic Regression: Logistic regression is a powerful statistical way to model a binomial result with one or more explanatory variables. Measure the relationship between the categorical dependent variable and one or more independent variables by estimating the probabilities using a logistic function, which is the cumulative logistic distribution.

Class Session ![BigData]https://github.com/Camacho18/Big-Data/blob/Unit_2/HomeWorks/4.png




5. Support Vector Machines: SVM is a binary classification algorithm. Given a set of points of 2 types at the N-dimensional location, SVM generates a dimensional (N-1) hyperlane to separate those points into 2 groups. Let's say you have some points of 2 types on a piece of paper that are linearly separable. SVM will find a straight line separating those points into 2 types and located as far as possible from all those points.








6. Métodos Ensemble: Ensemble methods are learning algorithms that build a set of classifiers and then classify new data points by taking a weighted vote of their predictions. The original set method is Bayesian averaging, but the latest algorithms include encoding output correction error.

UNSUPERVISED LEARNING

 
7. Clustering algorithms: Clustering is the task of grouping a set of objects such that the objects in the same group (cluster) are more similar to each other than to those of other groups.

 

8. Principal Component Analysis: PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of linearly uncorrelated variable values ​​called principal components.

Class Session![BigData]https://github.com/Camacho18/Big-Data/blob/Unit_2/HomeWorks/8%20Analysis.png



Principal component analysis

Some of the PCA applications include compression, data simplification for easier learning, visualization. Keep in mind that domain knowledge is very important when choosing whether to go ahead with PCA or not. Not suitable in cases where the data is noisy (all PCA components have a fairly high variance).

 
9. Singular Value Decomposition: In linear algebra, SVD is a factorization of a real complex matrix. For a given M * n matrix, there is a decomposition such that M = UΣV, where U and V are unit matrices and Σ is a diagonal matrix.

Class Session![BigData]https://github.com/Camacho18/Big-Data/blob/Unit_2/HomeWorks/9%20Singular.png


10. Independent Component Analysis: ICA is a statistical technique to reveal the hidden factors underlying sets of variables, measurements or random signals. ICA defines a generative model for the observed multivariate data, which is usually given as a large sample database. In the model, the data variables are assumed to be linear mixtures of some unknown latent variables, and the mixing system is also unknown. Latent variables are assumed to be non-Gaussian and mutually independent, and are called independent components of the observed data.



https://www.google.com/search?client=ubuntu&channel=fs&q=Main+Types+of+Machine+Learning+Algorithms&ie=utf-8&oe=utf-8
https://www.raona.com/los-10-algoritmos-esenciales-machine-learning/
https://www.raona.com/los-10-algoritmos-esenciales-machine-learning/
```

### &nbsp;&nbsp;Homework 2.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
VectorAssembler Library

### &nbsp;&nbsp;&nbsp;&nbsp; Investigation
```
                   
VectorAssembler is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees. VectorAssembler accepts the following input column types: all numeric types, boolean type, and vector type. In each row, the values of the input columns will be concatenated into a vector in the specified order.
Examples
Assume that we have a DataFrame with the columns id, hour, mobile, userFeatures, and clicked:
id | hour | mobile | userFeatures     | clicked
----|------|--------|------------------|---------
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0
userFeatures is a vector column that contains three user features. We want to combine hour, mobile, and userFeatures into a single feature vector called features and use it to predict clicked or not. If we set VectorAssembler’s input columns to hour, mobile, and userFeatures and output column to features, after transformation we should get the following DataFrame:
id | hour | mobile | userFeatures     | clicked | features
----|------|--------|------------------|---------|-----------------------------
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0     | [18.0, 1.0, 0.0, 10.0, 0.5]

Vectors Library

Factory methods for org.apache.spark.ml.linalg.Vector. We don't use the name Vector because Scala imports scala.collection.immutable.Vector by default.
```
### &nbsp;&nbsp;Homework 3.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
                   
Pipeline and Confusion Matrix
### &nbsp;&nbsp;&nbsp;&nbsp; Investigation

```

                   
Main concepts in Pipelines
MLlib standardizes APIs for machine learning algorithms to make it easier to combine multiple algorithms into a single pipeline, or workflow. This section covers the key concepts introduced by the Pipelines API, where the pipeline concept is mostly inspired by the scikit-learn project.
DataFrame:     This ML API uses DataFrame from Spark     SQL as an ML dataset, which can hold a variety of data types. E.g.,     a DataFrame could have different     columns storing text, feature vectors, true labels, and predictions.
     
Transformer:     A Transformer is an     algorithm which can transform one DataFrame     into another DataFrame. E.g., an ML     model is a Transformer which transforms     a DataFrame with features into a     DataFrame with predictions.
     
Estimator:     An Estimator is an algorithm which can     be fit on a DataFrame to produce a     Transformer. E.g., a learning algorithm     is an Estimator which trains on a     DataFrame and produces a model.
     
Pipeline:     A Pipeline chains multiple     Transformers and Estimators     together to specify an ML workflow.
     
Parameter:     All Transformers and Estimators     now share a common API for specifying parameters.


Pipeline components
Transformers

A Transformer is an abstraction that includes feature transformers and learned models. Technically, a Transformer implements a method transform(), which converts one DataFrame into another, generally by appending one or more columns. For example:

A feature transformer might take a DataFrame, read a column (e.g., text), map it into a new column (e.g., feature vectors), and output a new DataFrame with the mapped column appended.
A learning model might take a DataFrame, read the column containing feature vectors, predict the label for each feature vector, and output a new DataFrame with predicted labels appended as a column.

Estimators

An Estimator abstracts the concept of a learning algorithm or any algorithm that fits or trains on data. Technically, an Estimator implements a method fit(), which accepts a DataFrame and produces a Model, which is a Transformer. For example, a learning algorithm such as LogisticRegression is an Estimator, and calling fit() trains a LogisticRegressionModel, which is a Model and hence a Transformer.
Properties of pipeline components

Transformer.transform()s and Estimator.fit()s are both stateless. In the future, stateful algorithms may be supported via alternative concepts.

Each instance of a Transformer or Estimator has a unique ID, which is useful in specifying parameters (discussed below).
Pipeline

In machine learning, it is common to run a sequence of algorithms to process and learn from data. E.g., a simple text document processing workflow might include several stages:


Class Session![BigData]https://github.com/Camacho18/Big-Data/blob/Unit_2/Unit_2/Practices/HomeWorks/pipe1.png


Split     each document’s text into words.
     
Convert     each document’s words into a numerical feature vector.
     
Learn a     prediction model using the feature vectors and labels.
     
MLlib represents such a workflow as a Pipeline, which consists of a sequence of PipelineStages (Transformers and Estimators) to be run in a specific order. We will use this simple workflow as a running example in this section.
How it works

A Pipeline is specified as a sequence of stages, and each stage is either a Transformer or an Estimator. These stages are run in order, and the input DataFrame is transformed as it passes through each stage. For Transformer stages, the transform() method is called on the DataFrame. For Estimator stages, the fit() method is called to produce a Transformer (which becomes part of the PipelineModel, or fitted Pipeline), and that Transformer’s transform() method is called on the DataFrame.

We illustrate this for the simple text document workflow. The figure below is for the training time usage of a Pipeline.
Class Session![BigData]https://github.com/Camacho18/Big-Data/blob/Unit_2/Unit_2/Practices/HomeWorks/pipe2.png

Above, the top row represents a Pipeline with three stages. The first two (Tokenizer and HashingTF) are Transformers (blue), and the third (LogisticRegression) is an Estimator (red). The bottom row represents data flowing through the pipeline, where cylinders indicate DataFrames. The Pipeline.fit() method is called on the original DataFrame, which has raw text documents and labels. The Tokenizer.transform() method splits the raw text documents into words, adding a new column with words to the DataFrame. The HashingTF.transform() method converts the words column into feature vectors, adding a new column with those vectors to the DataFrame. Now, since LogisticRegression is an Estimator, the Pipeline first calls LogisticRegression.fit() to produce a LogisticRegressionModel. If the Pipeline had more Estimators, it would call the LogisticRegressionModel’s transform() method on the DataFrame before passing the DataFrame to the next stage.

A Pipeline is an Estimator. Thus, after a Pipeline’s fit() method runs, it produces a PipelineModel, which is a Transformer. This PipelineModel is used at test time; the figure below illustrates this usage.


CONFUSION MATRIX
after data cleaning, pre-processing and wrangling, the first step we do is to feed it to an outstanding model and of course, get output in probabilities. But hold on! How in the hell can we measure the effectiveness of our model. Better the effectiveness, better the performance and that’s exactly what we want. And it is where the Confusion matrix comes into the limelight. Confusion Matrix is a performance measurement for machine learning classification.

Well, it is a performance measurement for machine learning classification problem where output can be two or more classes. It is a table with 4 different combinations of predicted and actual values.

It is extremely useful for measuring Recall, Precision, Specificity, Accuracy and most importantly AUC-ROC Curve.
The confusion matrix of a class problem is an nx matrix in which the rows are named according to the real classes and the columns, according to the classes stopped by the model. It is used to explicitly show when one class is confused with another. Therefore, it allows working separately with different types of errors.

For example, in a binary model that seeks to predict whether a mushroom is poisonous or not, specify in the precision physical characteristics considered the real classes p (ositive = the mushroom is poisonous) and n (egative = the mushroom is edible), and the classes predicted by the model, S (í, is poisonous), or N (o, is edible). In this way, the confusion matrix for this model has its rows labeled with the real classes, and its columns with those predicted by the model. It would look like this:




In this way, the main diagonal contains the sum of all the correct predictions (the model says "S" and it is correct, it is poisonous, or it says "N" and it is correct, it is edible). The other diagonal reflects classifier errors: false positives or “true positives” (says that it is poisonous “S”, but in reality it is not “n”), or false negatives or “false negatives” (says that it is edible "N", but actually poisonous "p").
However, when the different “classes” are very unbalanced, this way of classifying the “goodness” of the operation of a classifier is of little use. For example, if the churn rate is 10% per month (that is, 10 people out of 100 unsubscribe per month), and we consider the customer who unsubscribes as a class "Positive", the expected positive: negative ratio would be 1: 9. So if we directly assigned all the clients the negative class (= no churn), we would be achieving a base precision of 90%, but… it would be useless.

