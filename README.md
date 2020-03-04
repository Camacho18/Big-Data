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
&nbsp;&nbsp;&nbsp;[Homework 1](#Homework-1)  
&nbsp;&nbsp;&nbsp;[Practice 5](#practice-5)  
&nbsp;&nbsp;&nbsp;[Practice 6](#practice-6)  
&nbsp;&nbsp;&nbsp;[Homework 2](#Homework-2)  
&nbsp;&nbsp;&nbsp;[Exam](#Exam)


### &nbsp;&nbsp;Practice 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
   
        1.-Develop an algorithm in scala that calculates the radius of a circle
        2.-Develop an algorithm in scala that tells me if a number is a cousin
        3.-Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet"
        4.-Given the variable message = "Hi Luke, I'm your father!" use slilce to extract the  sequence "Luke"
        5.-What is the difference in value and a variable in scala?
        6.-Given the tuple ((2,4,5), (1,2,3), (3,116,23))) the number 3,141 returns
        
        
#### In this practice, what we did was the connotation of marios methods, mathematical to get the radius of a circle, how to print from an interpolation, using a slilce to extract the sequence of a text   

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

        1.- Recursive version descending.
        2.- Version with explicit formula.
        3.- Iterative version.
        4.- Iterative version 2 variables.
        5.- Iterative vector version.
        6.- Divide and Conquer version.
#### In this practice what we did is fibonacci their different ways of doing them in their 6 ways using cycles to give us the 6

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
### &nbsp;&nbsp;Homework 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

#### This task is from Pearson's correlation, where we investigate what it is and how it is interpreted

### &nbsp;&nbsp;&nbsp;&nbsp; Investigation

```
The Pearson Correlation Coefficient is a measure of the correspondence or linear relationship between two random quantitative variables. In simpler words it can be defined as an index used to measure the degree of relationship that two variables have, both quantitative.

Having two variables, the correlation facilitates estimates of the value of one of them, with knowledge of the value of the other variable.

This coefficient is a measure that indicates the relative situation of the events with respect to the two variables, that is, it represents the numerical expression that indicates the degree of correspondence or relationship that exists between the 2 variables. These numbers vary between limits of +1 and -1.

The basis of the Pearson coefficient is as follows: The more intense the concordance (in direct or inverse sense) of the relative positions of the data in the two variables, the product of the numerator takes on greater value (in absolute sense). If the match is exact, the numerator is equal to N (or a -N), and the index takes a value equal to 1 (or -1).

How does that interpret Pearson's correlation coefficient?

Its dimension indicates the level of association between the variables.

When it is less than zero (r <0) It is said that there is a negative correlation: The variables are correlated in an inverse sense.
At high values ​​in one of the variables, low values ​​usually correspond to the other variable and vice versa. How much the value is closer to -1 said more obvious correlation coefficient will be extreme covariation.

If r = -1 there is talk of a perfect negative correlation, which implies an absolute determination between both variables, in a direct sense a perfect linear relationship of negative slope coexists.

When it is greater than zero (r> 0) It is said that there is a positive correlation: Both variables are correlated in a direct sense.
High values ​​in one of the variables correspond to high values ​​in the other variable and also in an inverse situation with low values. The closer to +1 the correlation coefficient is found, the more evident the covariation will be.

If r = 1 There is talk of a perfect positive correlation, which implies an absolute determination between the variables, in a direct sense a perfect linear relationship of positive slope coexists).

When it is equal to zero (r = 0) It is said that the variables are incorrectly related, it is not possible to establish some sense of covariation.
There is no linear relationship, but this does not necessarily imply that the variables are independent, and there may be non-linear relationships between the variables.

When the two variables are independent it is said that they are not correlated, although the result of reciprocity is not necessarily true.

To conclude it can be said that it looks more difficult than it turns out to be, especially if there is advanced technology, because today there are multiple programs that facilitate this work of calculating and interpreting the Pearson coefficient.

```

### &nbsp;&nbsp;Practice 5.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

#### In this practice it was to investigate so that the functions of aggregation work, and to apply them and of which we had already seen to take out 20 more functions and to apply them in a defined cvs

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

#### In this practice it was to apply the functions of aggregation, to apply them in another cvs in this era we lady information on diamonds

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

### &nbsp;&nbsp;Homework 2.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

#### In this practice we investigate how the variance function works and we see what parameters it receives and we apply another 20 functions
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
        
