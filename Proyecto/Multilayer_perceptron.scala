//Importamos las librerias que utilizaremos para realizar nuestro ejercicio
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Minimizamos los errores 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


//Creamos una sesion de spark 
val spark = SparkSession.builder().getOrCreate()

//cargamos nuestro archivo CSV

val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

// Se imprime el schema
df.printSchema()

// Visualizamos nuestro Dataframe 
df.show()




//Modificamos la columna de strings a datos numericos 
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))

//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)

//Cambiamos la columna a label 
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")

//Multilayer perceptron
//Dividimos los datos en un arreglo en partes de 70% y 30%
val split = feat.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = split(0)
val test = split(1)

// Especificamos las capas para la red neuronal
val layers = Array[Int](5, 2, 2, 4)


//Creamos el entrenador con sus parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

//Entrenamos el modelo
val model = trainer.fit(train)

//Imprimimos la exactitud
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
