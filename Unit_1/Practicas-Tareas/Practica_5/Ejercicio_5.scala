//5
df.select(avg("Sales")).show()

def avg(columnName: String): Column
Aggregate function: returns the average of the values in a group.
