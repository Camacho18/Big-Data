//9
df.select(collect_list("Sales")).show()

def collect_list(columnName: String): Column
Aggregate function: returns a list of objects with duplicates. 

