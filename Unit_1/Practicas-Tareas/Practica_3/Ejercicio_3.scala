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