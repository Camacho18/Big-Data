def fib4(n:Int): Int={
    var a=0
    var b=1
    for(k<-Range(0,n)){
        b=b+a
        a=b-a
    }
    a
}