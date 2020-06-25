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