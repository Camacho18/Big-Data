# Big-Data  

### <p align="center" > TECNOLÓGICO NACIONAL DE MÉXICO INSTITUTO TECNOLÓGICO DE TIJUANA SUBDIRECCIÓN ACADÉMICA DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN PERIODO: Enero - Junio 2020 </p>

###  <p align="center">  Carrera: Ing. En Sistemas Computacionales. Materia: 	Datos Masivos (BDD-1704 IF9A	).</p>

### <p align="center">  Maestro: Jose Christian Romero Hernandez	</p>
### <p align="center">  No. de control y nombre del alumno: 15211275 - Camacho Paniagua Luis Angel </p>
### <p align="center">  No. de control y nombre del alumno:16210585 - Valenzuela Rosales Marco Asael </p>


# Unit 1

## Index
&nbsp;&nbsp;&nbsp;[Practice 1](#practice-1)  
&nbsp;&nbsp;&nbsp;[Practice 2](#practice-2)  
&nbsp;&nbsp;&nbsp;[Practice 3](#practice-3)  
&nbsp;&nbsp;&nbsp;[Practice 4](#practice-4)  
&nbsp;&nbsp;&nbsp;[Practice 5](#practice-5)  
&nbsp;&nbsp;&nbsp;[Practice 6](#practice-6)  
&nbsp;&nbsp;&nbsp;[Practice 7](#practice-7)  
&nbsp;&nbsp;&nbsp;[Exam](#Exam)


### &nbsp;&nbsp;Practice 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
   
        1.-Develop an algorithm in scala that calculates the radius of a circle
        2.-Develop an algorithm in scala that tells me if a number is a cousin
        3.-Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet"
        4.-Given the variable message = "Hi Luke, I'm your father!" use slilce to extract the  sequence "Luke"
        5.-What is the difference in value and a variable in scala?
        6.-Given the tuple ((2,4,5), (1,2,3), (3,116,23))) the number 3,141 returns
        
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala   
      /*1.Develop an algorithm in scala that calculates the radius of a circle*/
            var a=3
     var r=math.sqrt(a/math.Pi)        
```        
```scala     
      /*2. Develop an algorithm in scala that tells me if a number is a cousin*/
        var t = ((2,4,5),(1,2,3),(3.1416,23))
        t._3._1
``` 

```scala  
      /*3. Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet"n*/
        var bird="tweet"
        printf(s"Estoy ecribiendo un %s",bird)
``` 

```scala   
        /*4. Given the variable message = "Hi Luke, I'm your father!" use slilce to extract the  sequence "Luke"*/
        var mensaje = "Hola Luke yo soy tu padre!"
        mensaje.slice(5,9)
``` 

```scala  
      /*5. What is the difference in value and a variable in scala?*/
       Value (val) is immutable once assigned the value this cannot be changed
       Variable (var) once assigned you can reassign the value, as long as the new value
       sea of the same type
``` 

```scala  
      /*6. Given the tuple ((2,4,5), (1,2,3), (3,116,23))) the number 3,141 returns*/
       var t = ((2,4,5),(1,2,3),(3.1416,23))
       t._3._1
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

        1.- Recursive version descending.
        2.- Version with explicit formula.
        3.- Iterative version.
        4.- Iterative version 2 variables.
        5.- Iterative vector version.
        6.- Divide and Conquer version.

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.


```scala
    /* 1.- Recursive version descending.*/
    def fib1( n : Int) : Int =
    {
        if(n<2) n
        else fib1( n-1 ) + fib1( n-2 )
    }
```
```scala
    /* 2.- Version with explicit formula.*/
    def fib2( n : Int ) : Int = {
        if(n<2)n
        else
        {
            var y = ((1+math.sqrt(5))/2)
            var j = ((math.pow(y,n)-math.pow((1-y),n))/math.sqrt(5))      
            j.toInt
        }
}
```
```scala
    /* 3.- Iterative version.*/
    def fib3(n : Int ): Int = {
        var a = 0
        var b = 1
        var c = 0
        for (k<-Range(0,n)){
            c=b+a
            a=b
            b=c
        }
        a
}
```
```scala
    /* 4.- Iterative version 2 variables.*/
    def fib4(n:Int): Int={
        var a=0
        var b=1
        for(k<-Range(0,n)){
            b=b+a
            a=b-a
        }
        a
}
```
```scala
    /* 5.- Iterative vector version.*/
    def fib5(n:Int): Int={
    if(n<2)n    
    else{        
        var v = List.range(0,n+1)
        for(k<-Range(2,n+1)){            
            v = v.updated(k,(v(k-1)+v(k-2)))            
        }            
        v(n)
     }     
}
```
```scala
    /* 6.- Versión Divide y Vencerás.*/
    def fib6(n:Int): Int={
    if(n<=0)0
    else{
        var i = n-1
        var aO = 0.0
        var aT = 1.0
        var ab = Array(aT,aO)
        var cd = Array(aO,aT)
        while(i>0){
            if(i%2==1){
                aO = (cd(1)*ab(1)+cd(0)*ab(0))
                aT = (cd(1)*(ab(1)+ab(0))+cd(0)*ab(1))
                ab(0) = aO
                ab(1) = aT
            }            
                aO = (math.pow(cd(0),2)+math.pow(cd(1),2))
                aT = (cd(1)*(2*cd(0)+cd(1)))
                cd(0)=aO
                cd(1)=aT
                i = i/2            
        }
        (ab(0)+ab(1)).toInt        
    }
}
```
### &nbsp;&nbsp;Practice 4.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```
¿Cómo ese interpreta el coeficiente de correlación de Pearson?

Su dimensión indica el nivel de asociación entre las variables.

Cuando es menor a cero (r < 0) Se dice que hay correlación negativa: Las variables se correlacionan en un sentido inverso.
A valores altos en una de las variables, le suelen corresponder valores bajos en la otra variable y viceversa. Cuánto el valor esté más próximo a -1 dicho coeficiente de correlación más evidente será la covariación extrema.

Si r= -1 se habla de correlación negativa perfecta, la cual supone una determinación absoluta entre ambas variables, en sentido directo coexiste una relación lineal perfecta de pendiente negativa.

Cuando es mayor a cero (r > 0) Se dice que hay correlación positiva: Ambas variables se correlacionan en un sentido directo.
A valores altos en una de las variables, le corresponden valores altos en la otra variable e igualmente en una situación inversa sucede con los valores bajos. Cuánto más próximo a +1 se encuentre el coeficiente de correlación más evidente será la covariación.

Si r = 1 Se habla de correlación positiva perfecta, la cual supone una determinación absoluta entre las variables, en sentido directo coexiste una relación lineal perfecta de pendiente positiva).

Cuando es igual a cero (r = 0) Se dice que las variables están incorrectamente relacionadas, no puede es posible establecer algún sentido de covariación.
No existe relación lineal, pero esto no implica necesariamente que las variables sean independientes, pudiendo existir relaciones no lineales entre las variables.

Cuando las dos variables son independientes se dice que no están correlacionadas, aunque el resultado de reciprocidad no es necesariamente cierto.

Para concluir se puede decir que se ve más difícil de lo que resulta ser, sobre todo si se cuenta con tecnología avanzada, pues hoy día existen múltiples programas que facilitan esta labor de cálculo e interpretación del coeficiente de Pearson.




```

### &nbsp;&nbsp;Practice 5.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```

//10 functions
```scala   
//1
df.select(first("Sales")).show()

def first(columnName: String): Column
Aggregate function: returns the first value of a column in a group.
The function by default returns the first values it sees. It will return the first non-null
value it sees when ignoreNulls is set to true. If all values are null, then null is returned.
``` 
``` scala
//2
df.select(last("Sales")).show()

def last(columnName: String): Column
Aggregate function: returns the last value of the column in a group.
The function by default returns the last values it sees. It will return the last non-null
 value it sees when ignoreNulls is set to true. If all values are null, then null is returned.
``` 
```scala
//3
df.select(mean("Sales")).show()

def mean(columnName: String): Column
Aggregate function: returns the average of the values in a group. Alias for avg.
``` 
``` scala
//4
df.select(var_pop("Sales")).show()

def var_pop(columnName: String): Column
Aggregate function: returns the population variance of the values in a group
``` 
```scala
//5
df.select(avg("Sales")).show()

def avg(columnName: String): Column
Aggregate function: returns the average of the values in a group.
``` 
``` scala
//6
df.select(var_samp("Sales")).show()

def var_samp(columnName: String): Column
Aggregate function: returns the unbiased variance of the values in a group.
``` 
```scala
//7
df.select(approx_count_distinct("Sales")).show()

def approx_count_distinct(columnName: String): Column
Aggregate function: returns the approximate number of distinct items in a group.
``` 
```scala
//8
df.select(avg("Sales")).show()

def avg(columnName: String): Column
Aggregate function: returns the average of the values in a group.
``` 
```scala
//9
df.select(collect_list("Sales")).show()

def collect_list(columnName: String): Column
Aggregate function: returns a list of objects with duplicates.
``` 
```scala
//10
df.select(kurtosis("Sales")).show()

def kurtosis(columnName: String): Column
Aggregate function: returns the kurtosis of the values in a group.
``` 
```scala
//11
df.select(skewness("Sales")).show()

def skewness(columnName: String): Column
Aggregate function: returns the skewness of the values in a group. 
``` 
```scala
//12
df.select(stddev("Sales")).show()

def stddev(columnName: String): Column
Aggregate function: alias for stddev_samp. 
``` 
```scala
//13
df.select(approxCountDistinct("Sales")).show()

def approxCountDistinct(columnName: String): Column 
``` 
```scala
//14
df.select(count("Sales")).show()

def corr(column1: Column, column2: Column): Column
Aggregate function: returns the Pearson Correlation Coefficient for two columns.
``` 
```scala
//15
df.select(max("Sales")).show()

def max(columnName: String): Column
Aggregate function: returns the maximum value of the column in a group. 
``` 
```scala
//16
df.select(corr("Sales","Sales")).show()
```
```scala

//17
df.select(covar_pop("Sales","Sales")).show()
```
```scala
//18
df.select(covar_samp("Sales","Sales")).show()
```
```scala
//19
df.select(approx_count_distinct("Company")).show()
```
```scala
//20
df.select(mean("Sales")).show()

```


### &nbsp;&nbsp;Practice 6.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

### &nbsp;&nbsp;&nbsp;&nbsp; Code.


//10 functions
```scala   
//1
df.select(first("Price")).show()

def first(columnName: String): Column
Aggregate function: returns the first value of a column in a group.
The function by default returns the first values it sees. It will return the first non-null
value it sees when ignoreNulls is set to true. If all values are null, then null is returned.
``` 
```scala
//2
df.select(last("Price")).show()

def last(columnName: String): Column
Aggregate function: returns the last value of the column in a group.
The function by default returns the last values it sees. It will return the last non-null
 value it sees when ignoreNulls is set to true. If all values are null, then null is returned.
``` 
```scala
//3
df.select(mean("Price")).show()

def mean(columnName: String): Column
Aggregate function: returns the average of the values in a group. Alias for avg.
``` 
``` scala
//4
df.select(var_pop("Price")).show()

def var_pop(columnName: String): Column
Aggregate function: returns the population variance of the values in a group
``` 
```scala
//5
df.select(avg("Price")).show()

def avg(columnName: String): Column
Aggregate function: returns the average of the values in a group.
``` 
``` scala
//6
df.select(var_samp("Price")).show()

def var_samp(columnName: String): Column
Aggregate function: returns the unbiased variance of the values in a group.
``` 
```scala
//7
df.select(approx_count_distinct("Price")).show()

def approx_count_distinct(columnName: String): Column
Aggregate function: returns the approximate number of distinct items in a group.
``` 
```scala
//8
df.select(avg("Price")).show()

def avg(columnName: String): Column
Aggregate function: returns the average of the values in a group.
``` 
```scala
//9
df.select(collect_list("Price")).show()

def collect_list(columnName: String): Column
Aggregate function: returns a list of objects with duplicates.
``` 
```scala
//10
df.select(kurtosis("Price")).show()

def kurtosis(columnName: String): Column
Aggregate function: returns the kurtosis of the values in a group.
``` 
```scala
//11
df.select(skewness("Price")).show()

def skewness(columnName: String): Column
Aggregate function: returns the skewness of the values in a group. 
``` 
```scala
//12
df.select(stddev("Price")).show()

def stddev(columnName: String): Column
Aggregate function: alias for stddev_samp. 
``` 
```scala
//13
df.select(approxCountDistinct("Price")).show()

def approxCountDistinct(columnName: String): Column 
``` 
```scala
//14
df.select(count("Price")).show()

def corr(column1: Column, column2: Column): Column
Aggregate function: returns the Pearson Correlation Coefficient for two columns.
``` 
```scala
//15
df.select(max("Price")).show()

def max(columnName: String): Column
Aggregate function: returns the maximum value of the column in a group. 
``` 
```scala
//16
df.select(corr("Price","Price")).show()
```
```scala

//17
df.select(covar_pop("Price","Price")).show()
```
```scala
//18
df.select(covar_samp("SPrice","Price")).show()
```
```scala
//19
df.select(approx_count_distinct("Price")).show()
```
```scala
//20
df.select(mean("Price")).show()

```

### &nbsp;&nbsp;Practice 7.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.


### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala
// 1
//def approx_count_distinct(columnName: String, rsd: Double): Column
//Aggregate function: returns the approximate number of distinct items in a group.
//rsd
//maximum estimation error allowed (//default = 0.05)
df.select(approx_count_distinct("Sales",2333.3)).show()
```
```scala
// 2
//def collect_set(columnName: String): Column
//Aggregate function: returns a set of objects with duplicate elements eliminated.
df.select(collect_set("Sales")).show()
```
```scala
// 3
//def collect_set(e: Column): Column
//Aggregate function: returns a set of objects with duplicate elements eliminated.
df.select(collect_set("Sales")).show()
```
```scala
// 4
//def countDistinct(columnName: String, columnNames: String*): Columnaprox
//Aggregate function: returns the number of distinct items in a group.
df.select(countDistinct("Sales","Sales2")).show()
```
```scala
// 5
//def first(columnName: String, ignoreNulls: Boolean): Column
//Aggregate function: returns the first value of a column in a group.
df.select(first(first"Sales",1)).show()
```
```scala
// 6
//def grouping(columnName: String): Column
//Aggregate function: indicates whether a specified column in a GROUP BY list is //Aggregated or not, returns 1 for //Aggregated or 0 for not //Aggregated in the result set.
df.select(grouping("Sales")).show()
```
```scala
// 7
//def grouping(e: Column): Column
//Aggregate function: indicates whether a specified column in a GROUP BY list is //Aggregated or not, returns 1 for //Aggregated or 0 for not //Aggregated in the result set.
df.select(grouping("Sales")).show()
```
```scala
// 8
//def grouping_id(colName: String, colNames: String*): Column
//Aggregate function: returns the level of grouping, equals to
df.select(grouping_id("Sales","Company")).show()
```
```scala
// 9
//def grouping_id(cols: Column*): Column
//Aggregate function: returns the level of grouping, equals to
df.select(grouping_id("Sales")).show()
```
```scala
// 10
//def last(columnName: String, ignoreNulls: Boolean): Column
//Aggregate function: returns the last value of the column in a group.
df.select(last("Sales",1)).show()
```
```scala
//11
def min(columnName: String): Column
df.select(min("Sales")).show()
```
```scala
//12
//Aggregate function: returns the minimum value of the column in a group.
def min(e: Column): Column
df.select(min("Sales")).show()
```
```scala
//13

//Aggregate function: returns the minimum value of the expression in a group.
def stddev_pop(columnName: String): Column
df.select(stddev_pop("Sales")).show()
```
```scala

//14
//Aggregate function: returns the population standard deviation of the expression in a group.
def stddev_pop(e: Column): Column
df.select(stddev_pop("Sales")).show()
```
```scala
//15
//Aggregate function: returns the population standard deviation of the expression in a group.
def stddev_samp(columnName: String): Column
df.select(stddev_samp("Sales")).show()
```
```scala
//16
//Aggregate function: returns the sample standard deviation of the expression in a group.
def stddev_samp(e: Column): Column
df.select(stddev_samp("Sales")).show()
```
```scala
//17
//Aggregate function: returns the sample standard deviation of the expression in a group.
def sum(columnName: String): Column

df.select(sum("Sales")).show()
```
```scala
//18
//Aggregate function: returns the sum of all values in the given column.
def sum(e: Column): Column
df.select(sum("Sales")).show()
```
```scala
//19
Aggregate function: returns the sum of all values in the expression.
def var_samp(columnName: String): Column
df.select( var_samp("Sales")).show()
```
```scala
//20
//Aggregate function: returns the unbiased variance of the values in a group.
//def var_samp(e: Column): Column
df.select( var_samp("Sales")).show()
```





## &nbsp;&nbsp;Exam.

### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

    Given a square matrix, calculate the absolute difference between the sums of your diagonals.

        arr = [[11,2,4], [4,5,6], [10,3, -12]]

        diagonal_1 = 11 + 5 - 12 = 4

        diagonal_2 = 4 + 5 + 10 = 19

        Absolute Address = | 4 - 19 | = 15
        
    Develop a function called diagonalDifference in a scrip with the Scala programming language. It must return an integer that represents the difference of the absolute diagonal.
### &nbsp;&nbsp;&nbsp;&nbsp; Code.


```scala
    /* Exam Unit_1*/
    var arr = List(List(11,2,4),List(4,5,6),List(10,8,-12))

    def diagonalDifference(list:List[List[Int]]): Int ={
        var n = list.length
        var suma1 =0;
        var suma2 =0;
        for(x<-0 to n-1)
        {
            for(y<-0 to n-1)
            {
                if (x == y) suma1 += list(x)(y)
                if (x + y == n - 1) suma2 += list(x)(y)           
            }
        }
        println(s"Suma 1: $suma1")
        println(s"Suma 2:$suma2")
        math.abs(suma1-suma2)
}
```
        
