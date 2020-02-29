// 1
//def approx_count_distinct(columnName: String, rsd: Double): Column
//Aggregate function: returns the approximate number of distinct items in a group.
//rsd
//maximum estimation error allowed (//default = 0.05)
df.select(approx_count_distinct("Sales",2333.3)).show()

// 2
//def collect_set(columnName: String): Column
//Aggregate function: returns a set of objects with duplicate elements eliminated.
df.select(collect_set("Sales")).show()

// 3
//def collect_set(e: Column): Column
//Aggregate function: returns a set of objects with duplicate elements eliminated.
df.select(collect_set("Sales")).show()

// 4
//def countDistinct(columnName: String, columnNames: String*): Columnaprox
//Aggregate function: returns the number of distinct items in a group.
df.select(countDistinct("Sales","Sales2")).show()

// 5
//def first(columnName: String, ignoreNulls: Boolean): Column
//Aggregate function: returns the first value of a column in a group.
df.select(first(first"Sales",1)).show()

// 6
//def grouping(columnName: String): Column
//Aggregate function: indicates whether a specified column in a GROUP BY list is //Aggregated or not, returns 1 for //Aggregated or 0 for not //Aggregated in the result set.
df.select(grouping("Sales")).show()

// 7
//def grouping(e: Column): Column
//Aggregate function: indicates whether a specified column in a GROUP BY list is //Aggregated or not, returns 1 for //Aggregated or 0 for not //Aggregated in the result set.
df.select(grouping("Sales")).show()

// 8
//def grouping_id(colName: String, colNames: String*): Column
//Aggregate function: returns the level of grouping, equals to
df.select(grouping_id("Sales","Company")).show()

// 9
//def grouping_id(cols: Column*): Column
//Aggregate function: returns the level of grouping, equals to
df.select(grouping_id("Sales")).show()

// 10
//def last(columnName: String, ignoreNulls: Boolean): Column
//Aggregate function: returns the last value of the column in a group.
df.select(last("Sales",1)).show()

//11
def min(columnName: String): Column
df.select(min("Sales")).show()
//12
//Aggregate function: returns the minimum value of the column in a group.
def min(e: Column): Column
df.select(min("Sales")).show()
//13

//Aggregate function: returns the minimum value of the expression in a group.
def stddev_pop(columnName: String): Column
df.select(stddev_pop("Sales")).show()

//14
//Aggregate function: returns the population standard deviation of the expression in a group.
def stddev_pop(e: Column): Column
df.select(stddev_pop("Sales")).show()

//15
//Aggregate function: returns the population standard deviation of the expression in a group.
def stddev_samp(columnName: String): Column
df.select(stddev_samp("Sales")).show()

//16
//Aggregate function: returns the sample standard deviation of the expression in a group.
def stddev_samp(e: Column): Column
df.select(stddev_samp("Sales")).show()

//17
//Aggregate function: returns the sample standard deviation of the expression in a group.
def sum(columnName: String): Column

df.select(sum("Sales")).show()

//18
//Aggregate function: returns the sum of all values in the given column.
def sum(e: Column): Column
df.select(sum("Sales")).show()

//19
Aggregate function: returns the sum of all values in the expression.
def var_samp(columnName: String): Column
df.select( var_samp("Sales")).show()

//20
//Aggregate function: returns the unbiased variance of the values in a group.
//def var_samp(e: Column): Column
df.select( var_samp("Sales")).show()


