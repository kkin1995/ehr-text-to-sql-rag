import sqlite3
import os
import glob
import pandas as pd

path_to_data_csv = "/Users/karankinariwala/OneDrive/Medeva LLM Internship/data/synthea_sample_data_csv_apr2020/csv"
csv_files = glob.glob(os.path.join(path_to_data_csv, "*.csv"))
filenames = []
for i in range(len(csv_files)):
    filenames.append(csv_files[i].split("/")[-1])

conn = sqlite3.connect("patient_health_data.db")

n_csv_files = len(filenames)
for i in range(n_csv_files):
    name_of_table = filenames[i].split(".")[0]
    print(f"Creating Table: {name_of_table}")
    df = pd.read_csv(os.path.join(path_to_data_csv, filenames[i]))
    df.to_sql(name=name_of_table, con=conn)

conn.close()
