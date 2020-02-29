// Pract_4
// def var_pop(columnName: String): Column
// Aggregate function: returns the population variance of the values in a group
df.select(var_pop("Sales")).show()