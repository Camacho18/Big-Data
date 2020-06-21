import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder,IndexToString}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

// Minimiza los erorres mostrados 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Valores para medir el rendimiento
val runtime = Runtime.getRuntime
val startTimeMillis = System.currentTimeMillis()

// Se inicia una sesion en spark
val spark = SparkSession.builder().getOrCreate()

// Se cargan los datos en la variable "data" en el formato "csv"
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

// Se imprime el schema del dataFrame
data.printSchema()

//Convertimos los valores de la columna "y" en numerico
val change1 = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),0).otherwise(col("y")))
val newDF = change2.withColumn("y",'y.cast("Int"))


// Generamos un vector con los nombres de las columnas a evaluar
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))

// Se rellena el vector con los valores
val features = vectorFeatures.transform(newDF)

//Renombramos la columna "y" a Label
val featuresLabel = features.withColumnRenamed("y", "label")

//Indexamos las columnas label y features
val dataIndexed = featuresLabel.select("label","features")

// Indexamos las etiquetas
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed) 

// Dividimos los datos, 70% entrenamiento y 30% prueba
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))

// Creamos el modelo y le indicamos las columnas a utilizar
val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")

// Metemos todo en una tuberia
val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))

// Ahora si entrenamos el modelo con el 70% de los datos
val model = pipeline.fit(trainingData)

// Realizamos la predicción de los datos con el 30% de la data
val predictions = model.transform(testData)

// imprimiemos los primero s5 registros
predictions.select("prediction", "label", "features").show(5)

// Seleecionamos las columnas y el valor del error
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println("Learned regression tree model:\n" + treeModel.toDebugString)

// 1mb -> 1e6 bytes
var mb = 0.000001
println("Used Memory: " + ((runtime.totalMemory - runtime.freeMemory) * mb) + " mb")
println("Free Memory: " + (runtime.freeMemory * mb) + " mb")
println("Total Memory: " + (runtime.totalMemory * mb) + " mb")
println("Max Memory: " + (runtime.maxMemory * mb)+ " mb")

val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (endTimeMillis - startTimeMillis) / 1000 + "s"