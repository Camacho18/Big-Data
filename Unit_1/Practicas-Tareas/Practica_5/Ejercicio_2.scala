// Pract_2
// def last(columnName: String): Column
// Aggregate function: returns the last value of the column in a group.
// The function by default returns the last values it sees. It will return the last non-null
// value it sees when ignoreNulls is set to true. If all values are null, then null is returned. 
df.select(last("Sales")).show()