// Pract_14
// def count(columnName: String): TypedColumn[Any, Long]
// Aggregate function: returns the number of items in a group.
df.select(count("Sales")).show()