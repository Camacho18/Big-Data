//Importar una simple sesión Spark.
//Importar Vector Assembler y Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors

//Utilizaos las lineas de código para minimizar errores 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Cree una instancia de la sesión Spark
val spark = SparkSession.builder().getOrCreate()

// Importar la librería de Kmeans para el algoritmo de agrupamiento.
import org.apache.spark.ml.clustering.KMeans

// Cargamos  el dataset de Wholesale Customers Data
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Wholesale customers data.csv")

// Se imprime el dataFrame
data.printSchema()

// Seleccione las siguientes columnas: Fres, Milk, Grocery, Frozen, Detergents_Paper,Delicassen 
//y llamar a este conjunto feature_data
val f_data= (data.select($"Fresh", $"Milk",$"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen"))

// Se limpia el dataframe los campos vacios 
val f_data_clean = f_data.na.drop()

// Crea un nuevo objeto Vector Assembler para las columnas de caracteristicas 
//como un conjunto de entrada, recordando que no hay etiquetas
val f_Data = (new VectorAssembler().setInputCols(Array("Fresh","Milk", "Grocery","Frozen", "Detergents_Paper","Delicassen")).setOutputCol("features"))

//Utilice el objeto assembler para transformar feature_data
val features = f_Data.transform(f_data_clean)

// Se ejecuta el modelo Kmeans con k = 3
val kmeans = new KMeans().setK(3).setSeed(1L).setPredictionCol("cluster")
val model = kmeans.fit(features)

//Evaluamos los grupos utilizando Within Set Sum of Squared Errors WSSSE e 

val WSSE = model.computeCost(features)
println(s"Within set sum of Squared Errors = $WSSE")

// Imprimimos los clusters 
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

Bueno despues de mi companero yo seguire con la explicacion del codigo

// Se limpia el dataframe los campos vacios  usando la funcion na.drop se borra algo en especifico en nuestro archivo  csv
creando una variable f_data_clean igual que f_data.na.drop() podemos ejecutar esta funcion
val f_data_clean = f_data.na.drop()


// Crea un nuevo objeto Vector Assembler  este sirve para que combina una lista dada de columnas en una sola columna vectorial. Es útil para combinar características en bruto y características generadas por diferentes transformadores de características en un solo vector de características, con el fin de entrenar modelos Machine Learning en este caso usaremoslas columnas de caracteristicas usando los datos igual de Fresh,Milk,Grocery,frozen, detergents paper
Delicassen //como un conjunto de entrada, recordando que no hay etiquetas
val f_Data = (new VectorAssembler().setInputCols(Array("Fresh","Milk", "Grocery","Frozen", "Detergents_Paper","Delicassen")).setOutputCol("features"))


//Utilizamos aqui creando otra variable  el objeto assembler para transformar feature_data_clean
val features = f_Data.transform(feature_data_clean)

// Se ejecuta el modelo Kmeans que es uno de los algoritmos de agrupación más utilizados que agrupa los puntos de datos en un número predefinido de agrupaciones. con k = 3 k es el número de grupos deseados. Tenga en cuenta que es posible que se devuelvan menos de k grupos, por ejemplo, si hay menos de k puntos distintos para agrupar.
val kmeans = new KMeans().setK(3).setSeed(1L).setPredictionCol("cluster")
val model = kmeans.fit(features)

//Evaluamos los grupos utilizando Within Set Sum of Squared Errors WSSSE el  objetivo  es minimizar la suma de cuadrados de la distancia entre los puntos de cada conjunto: la distancia euclidiana al cuadrado. Este es el objetivo de WCSS

val WSSE = model.computeCost(features)
println(s"Within set sum of Squared Errors = $WSSE")

// Imprimimos los clusters
println("Cluster Centers: ")
model.clusterCenters.foreach(println)