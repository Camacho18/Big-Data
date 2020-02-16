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