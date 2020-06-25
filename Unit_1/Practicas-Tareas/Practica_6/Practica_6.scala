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

/11
df.select(avg("Price")).show()

//12
df.select(var_samp("Price")).show()

//13
df.select(approx_count_distinct("Price")).show()

//14
df.select(first("Price")).show()


//15
df.select(last("Price")).show()


//16
df.select(mean("Price")).show()


//17
df.select(var_pop("Price")).show()


//18
df.select(avg("Price")).show()


//19
df.select(var_samp("Price")).show()


//20
df.select(approx_count_distinct("Price")).show()
