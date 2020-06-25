// Pract_8
// def avg(columnName: String): Column
// Aggregate function: returns the average of the values in a group. 
df.select(avg("Sales")).show()