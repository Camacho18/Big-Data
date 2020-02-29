// Pract_6
// def var_samp(columnName: String): Column
// Aggregate function: returns the unbiased variance of the values in a group.
df.select(var_samp("Sales")).show()