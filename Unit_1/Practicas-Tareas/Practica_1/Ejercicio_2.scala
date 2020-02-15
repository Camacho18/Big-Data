//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
def Prime(n:Int):Boolean={
    if(n%2==0){
        return false
    }
    else{
        for(i <- Range(3,n)){
            if(n%i==0){
                return false
            }
        }
        return true                
    }
}
Prime(2)
Prime(4)
Prime(7)
Prime(13)