// Pract_18
// def covar_samp(columnName1: String, columnName2: String): Column
// Aggregate function: returns the sample covariance for two columns.
df.select(covar_samp("Sales","Sales2")).show()