//15
df.select(max("Sales")).show()

def max(columnName: String): Column
Aggregate function: returns the maximum value of the column in a group. 