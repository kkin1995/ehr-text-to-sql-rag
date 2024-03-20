from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

system_message = """
Your task is to convert a doctor's natural language query into a specific SQL query, leveraging the schema of an Electronic Health Record (EHR) system's database. This process involves understanding the intent and details of the doctor's request and accurately mapping it to a SQL query that interacts with the EHR database schema to retrieve the desired information. Follow these steps:

Understand the EHR Database Schema: Begin with a thorough analysis of the provided EHR system's SQL database schema. Familiarize yourself with the tables, columns, relationships, and data types to understand how the database stores and relates medical data.

Translate Natural Language to SQL: Convert the doctor's natural language query into an SQL query. This step requires interpreting medical terminology and query intent to construct a SQL statement that accurately targets the relevant data within the EHR system.

Ensure SQL Query Accuracy and Efficiency: Your SQL query should not only accurately reflect the doctor's request but also be optimized for efficient execution. Consider the best practices for query performance, such as selecting only necessary columns and using appropriate JOINs.

Prepare for Iterative Refinement: Anticipate the need for adjustments. Based on feedback or further clarifications, be ready to refine your SQL query to better match the doctor's information needs or to accommodate any additional insights about the database schema.

Output: In your output, you should only give the SQL query. Do not give any more information or text in addition to the SQL query.

Table Schema:

CREATE TABLE patients (
	Id, String,
	BIRTHDATE, Date,
	DEATHDATE, Date,
	SSN, String,
	DRIVERS, Int32,
	PASSPORT, String,
	PREFIX, String,
	FIRST, String,
	LAST, String,
	SUFFIX, String,
	MAIDEN, String,
	RACE, String,
	ETHNICITY, String,
	GENDER, String,
	BIRTHPLACE, String,
	ADDRESS, String,
	CITY, String,
	STATE, String,
	COUNTRY, String,
	ZIP, Int32,
	LAT, Float32,
	LON, Float32,
	HEALTHCARE_EXPENSES, Float32,
	HEALTHCARE_COVERAGE, Float32
)

CREATE TABLE conditions (
	START Date,
	STOP Date,
	PATIENT String,
	ENCOUNTER String,
	CODE Int32,
	DESCRIPTION String
)
"""

user_message = "How does the prevalence of specific conditions vary across different age groups and ethnicities within our patient population?"

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ],
)

print(completion.choices[0].message)

# ChatCompletionMessage(
# content=
# "SELECT COUNT(*) AS condition_count, conditions.DESCRIPTION,
# FLOOR(DATEDIFF('YEAR', patients.BIRTHDATE, CURRENT_DATE) / 10) * 10 AS age_group,
# patients.ETHNICITY
# FROM conditions
# JOIN patients ON conditions.PATIENT = patients.Id
# GROUP BY conditions.DESCRIPTION, age_group, patients.ETHNICITY;",
# role='assistant', function_call=None, tool_calls=None)
