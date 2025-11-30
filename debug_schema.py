import pathway as pw

schema = pw.schema_from_csv("data/stream/transactions.csv")
print(schema)
