def fib2( n : Int ) : Int = {
  if(n<2)n
  else
  {
      var y = ((1+math.sqrt(5))/2)
      var j = ((math.pow(y,n)-math.pow((1-y),n))/math.sqrt(5))      
      j.toInt
  }
}