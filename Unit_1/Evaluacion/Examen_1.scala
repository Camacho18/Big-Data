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