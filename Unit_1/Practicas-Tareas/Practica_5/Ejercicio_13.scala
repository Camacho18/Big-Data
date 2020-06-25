//13
df.select(approxCountDistinct("Sales")).show()

def approxCountDistinct(columnName: String): Column 
