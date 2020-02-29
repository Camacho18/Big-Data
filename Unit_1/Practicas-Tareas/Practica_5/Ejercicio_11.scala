
//11
df.select(skewness("Sales")).show()

def skewness(columnName: String): Column
Aggregate function: returns the skewness of the values in a group. 
