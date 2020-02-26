# Big-Data  

### <center>  TECNOLÓGICO NACIONAL DE MÉXICO</center> <center> INSTITUTO TECNOLÓGICO DE TIJUANA</center> <center> SUBDIRECCIÓN ACADÉMICA</center> <center> DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN</center><center> PERIODO: Enero - Junio 2020</center>

### <center> Carrera: Ing. En Sistemas Computacionales.</center> <center> Materia: 	Datos Masivos (BDD-1704 IF9A	).</center>

### <center> Maestro: Jose Christian Romero Hernandez		</center> <center> No. de control y nombre del alumno:</center> <center> 15211275 - Camacho Paniagua Luis Angel</center>
<center> 16210585 - Valenzuela Rosales Marco Asael</center>

# Unit 1

## Index
&nbsp;&nbsp;&nbsp;[Practice 1](#practice-1)  
&nbsp;&nbsp;&nbsp;[Practice 2](#practice-2)  
&nbsp;&nbsp;&nbsp;[Practice 3](#practice-3)  
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
```    /* 1.Develop an algorithm in scala that calculates the radius of a circle
            var a=3
     var r=math.sqrt(a/math.Pi)*/        
```        
```    /* 2. Develop an algorithm in scala that tells me if a number is a cousin
        var t = ((2,4,5),(1,2,3),(3.1416,23))
        t._3._1
*/
``` 

```    /* 3. Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet"n
        var bird="tweet"
        printf(s"Estoy ecribiendo un %s",bird)
*/
``` 

```    /*4. Given the variable message = "Hi Luke, I'm your father!" use slilce to extract the  sequence "Luke"
        var mensaje = "Hola Luke yo soy tu padre!"
        mensaje.slice(5,9)
*/
``` 

```    /* 5. What is the difference in value and a variable in scala?
       Value (val) is immutable once assigned the value this cannot be changed
       Variable (var) once assigned you can reassign the value, as long as the new value
       sea of the same type
*/
``` 

```    /* 6. Given the tuple ((2,4,5), (1,2,3), (3,116,23))) the number 3,141 returns*/
       var t = ((2,4,5),(1,2,3),(3.1416,23))
       t._3._1

*/
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
```    /*1.-Create a list called "list" with the elements "red", "white", "black"
         var lista = collection.mutable.MutableList("rojo","blanco","negro")      
```        
```    /* 2.-Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
          lista += ("verde","amarillo", "azul", "naranja", "perla")
*/
``` 

```    /* 3.-Bring the "list" "green", "yellow", "blue" items
             lista(3)
             lista(4)
             lista(5)
*/
``` 

```    /* 4.-Create a number array in the 1-1000 range in 5-in-5 steps
               var v = Range(1,1000,5)
*/
``` 

```    /*  5.-What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets
              var l = List(1,3,3,4,6,7,3,7)
               l.toSet
*/
``` 

```    /* 6.-Create a mutable map called names containing the following"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27
          var map=collection.mutable.Map(("Jose", 20),("Luis", 24),("Ana", 23),("Susana", "27"))
*/
``` 
```    /*6.-a. Print all map keys
           map.keys
*/
``` 
```    /* 7.-b. Add the following value to the map ("Miguel", 23)
           map += ("Miguel"->23)
*/
``` 

### &nbsp;&nbsp;Practice 3.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

        1.- Recursive version descending.
        2.- Version with explicit formula.
        3.- Iterative version.
        4.- Iterative version 2 variables.
        5.- Iterative vector version.
        6.- Versión Divide y Vencerás.

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
        
