//10 functions
//1
df.select(first("Sales")).show()

def first(columnName: String): Column
Aggregate function: returns the first value of a column in a group.
The function by default returns the first values it sees. It will return the first non-null 
value it sees when ignoreNulls is set to true. If all values are null, then null is returned. 

//2
df.select(last("Sales")).show()

def last(columnName: String): Column
Aggregate function: returns the last value of the column in a group.
The function by default returns the last values it sees. It will return the last non-null
 value it sees when ignoreNulls is set to true. If all values are null, then null is returned. 

//3
df.select(mean("Sales")).show()

def mean(columnName: String): Column
Aggregate function: returns the average of the values in a group. Alias for avg. 

//4
df.select(var_pop("Sales")).show()

def var_pop(columnName: String): Column
Aggregate function: returns the population variance of the values in a group

//5
df.select(avg("Sales")).show()

def avg(columnName: String): Column
Aggregate function: returns the average of the values in a group.

//6
df.select(var_samp("Sales")).show()

def var_samp(columnName: String): Column
Aggregate function: returns the unbiased variance of the values in a group.

//7
df.select(approx_count_distinct("Sales")).show()

def approx_count_distinct(columnName: String): Column
Aggregate function: returns the approximate number of distinct items in a group. 

//8
df.select(avg("Sales")).show()

def avg(columnName: String): Column
Aggregate function: returns the average of the values in a group. 

//9
df.select(collect_list("Sales")).show()

def collect_list(columnName: String): Column
Aggregate function: returns a list of objects with duplicates. 

//10
df.select(kurtosis("Sales")).show()

def kurtosis(columnName: String): Column
Aggregate function: returns the kurtosis of the values in a group. 
