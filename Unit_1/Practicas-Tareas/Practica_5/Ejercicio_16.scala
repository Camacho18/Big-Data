// Pract_16
// def corr(columnName1: String, columnName2: String): Column
// Aggregate function: returns the Pearson Correlation Coefficient for two columns.
df.select(corr("Sales","Sales2")).show()