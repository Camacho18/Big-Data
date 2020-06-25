def fib1( n : Int) : Int =
{
    if(n<2) n
    else fib1( n-1 ) + fib1( n-2 )
}