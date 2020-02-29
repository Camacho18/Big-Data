//7
df.select(approx_count_distinct("Sales")).show()

def approx_count_distinct(columnName: String): Column
Aggregate function: returns the approximate number of distinct items in a group. 