// Pract_12
// def stddev(columnName: String): Column
// Aggregate function: alias for stddev_samp. 
df.select(stddev("Sales")).show()