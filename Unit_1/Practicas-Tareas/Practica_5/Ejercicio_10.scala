// Pract_10
// def kurtosis(columnName: String): Column
// Aggregate function: returns the kurtosis of the values in a group. 
df.select(kurtosis("Sales")).show()