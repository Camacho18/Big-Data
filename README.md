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
