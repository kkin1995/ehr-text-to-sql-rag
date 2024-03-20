import clickhouse_connect
import os
import glob
import pandas as pd

path_to_data_csv = "/Users/karankinariwala/OneDrive/Medeva LLM Internship/data/synthea_sample_data_csv_apr2020/csv"

client = clickhouse_connect.get_client()

create_allergies_table = "CREATE TABLE allergies (start Date, stop Nullable(Date), patient String, encounter String, code String, description String) ENGINE MergeTree ORDER BY (patient, code)"
create_careplans_table = "CREATE TABLE careplans (id String, start Date, stop Date, patient String, encounter String, code String, description String, reasoncode Float32, reasondescription String) ENGINE MergeTree ORDER BY (id)"

client.command(create_allergies_table)
client.command(create_careplans_table)

df = pd.read_csv(os.path.join(path_to_data_csv, "allergies.csv"))
client.command("INSERT INTO allergies VALUES", df.to_dict("records"))
# client.insert_df(
#     "allergies",
#     df,
#     column_names=["start", "stop", "patient", "encounter", "code", "description"],
# )

result = client.query("SHOW TABLES")
for table in result.result_rows:
    print(table)

result = client.query("SELECT * FROM allergies LIMIT 5;")
for table in result.result_rows:
    print(table)
