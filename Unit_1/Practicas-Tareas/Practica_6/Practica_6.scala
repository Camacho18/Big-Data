//Diamantes 
//1
df.select(max("Price")).show()

//2
df.select(countDistinct("Price")).show()

//3
df.select(sumDistinct("Price")).show()

//4
df.select(variance("Price")).show()

//5
df.select(stddev("Price")).show()

//6
df.select(collect_set("Price")).show()

//7
df.select(first("Price")).show()

//8
df.select(last("Price")).show()

//9
df.select(mean("Price")).show()

//10
df.select(var_pop("Price")).show()

