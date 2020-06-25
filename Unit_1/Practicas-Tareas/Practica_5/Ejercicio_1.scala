//10 functions
//1
df.select(first("Sales")).show()

def first(columnName: String): Column
Aggregate function: returns the first value of a column in a group.
The function by default returns the first values it sees. It will return the first non-null 
value it sees when ignoreNulls is set to true. If all values are null, then null is returned. 










