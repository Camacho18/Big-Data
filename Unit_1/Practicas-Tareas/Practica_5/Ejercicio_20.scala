// Pract_20
// mean(columnName: String): Column
// Aggregate function: returns the average of the values in a group.
df.select(mean("Sales")).show()