import sqlite3

connection = sqlite3.connect("patient_health_data.db")
cursor = connection.cursor()

# cursor.execute(
#     """
#     SELECT
#         p.ETHNICITY,
#         CASE
#         WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) BETWEEN 0 AND 17 THEN '0-17'
#         WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) BETWEEN 18 AND 34 THEN '18-34'
#         WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) BETWEEN 35 AND 50 THEN '35-50'
#         WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) BETWEEN 51 AND 69 THEN '51-69'
#             ELSE '70+' END AS AgeGroup,
#         c.DESCRIPTION AS Condition,
#         COUNT(DISTINCT p.Id) AS PatientCount
#     FROM
#         patients p
#     JOIN
#         conditions c ON p.Id = c.PATIENT
#     WHERE
#         c.STOP IS NULL
#     GROUP BY
#         p.ETHNICITY, AgeGroup, Condition
#     ORDER BY
#         p.ETHNICITY, AgeGroup, Condition;
#     """
# )

# cursor.execute(
#     """
#     SELECT
#         p.FIRST || ' ' || p.LAST AS PatientName,
#         c.DESCRIPTION AS Condition,
#         c.START AS DiagnosisDate,
#         c.STOP AS ResolutionDate
#     FROM
#         patients p
#     INNER JOIN
#         conditions c ON p.Id = c.PATIENT
#     WHERE
#         p.Id = 'f5dcd418-09fe-4a2f-baa0-3da800bd8c3a'
#     ORDER BY
#         c.START;
#     """
# )

# cursor.execute(
#     """
#     SELECT
#         cp.DESCRIPTION AS CarePlan,
#         COUNT(DISTINCT c.PATIENT) AS TotalConditions,
#         COUNT(DISTINCT CASE WHEN c.STOP IS NOT NULL THEN c.PATIENT ELSE NULL END) AS ResolvedConditions,
#         ROUND((COUNT(DISTINCT CASE WHEN c.STOP IS NOT NULL THEN c.PATIENT ELSE NULL END) * 1.0 / COUNT(DISTINCT c.PATIENT)) * 100, 2) AS ResolutionRate
#     FROM
#         careplans cp
#     INNER JOIN
#         conditions c ON cp.PATIENT = c.PATIENT
#     WHERE
#         c.DESCRIPTION LIKE '%asthma%'
#         AND c.START <= cp.STOP
#         AND (c.STOP IS NULL OR c.STOP >= cp.START)
#     GROUP BY
#         cp.DESCRIPTION
#     ORDER BY
#         ResolutionRate DESC, TotalConditions DESC;
#     """
# )

# cursor.execute(
#     """
#     SELECT
#         m.DESCRIPTION AS Medication,
#         COUNT(m.PATIENT) AS Prescriptions,
#         AVG(m.BASE_COST) AS AverageCost,
#         AVG(m.PAYER_COVERAGE) AS AveragePayerCoverage
#     FROM
#         medications m
#     INNER JOIN
#         conditions c ON m.PATIENT = c.PATIENT
#     WHERE
#         c.DESCRIPTION LIKE '%heart%'
#         OR c.DESCRIPTION LIKE '%cardiac%'
#     GROUP BY
#         m.DESCRIPTION
#     ORDER BY
#         Prescriptions DESC, AverageCost DESC
#     LIMIT 10;
#     """
# )

# cursor.execute(
#     """
#     SELECT
#         p.Id AS PatientID,
#         p.FIRST || ' ' || p.LAST AS PatientName,
#         a.DESCRIPTION AS AllergyDescription,
#         a.START AS AllergyStart,
#         a.STOP AS AllergyStop
#     FROM
#         allergies a
#     JOIN
#         patients p ON a.PATIENT = p.Id
#     ORDER BY
#         p.LAST, p.FIRST;
#     """
# )

# cursor.execute(
#     """
#     SELECT
#         d.DESCRIPTION AS DeviceType,
#         COUNT(d.PATIENT) AS NumberOfUsers,
#         AVG(JULIANDAY(d.STOP) - JULIANDAY(d.START)) AS AverageUsageDuration
#     FROM
#         devices d
#     INNER JOIN
#         patients p ON d.PATIENT = p.Id
#     GROUP BY
#         d.DESCRIPTION
#     ORDER BY
#         NumberOfUsers DESC
#     """
# )

# cursor.execute(
#     """
#     SELECT
#         CASE
#             WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) < 18 THEN '0-17'
#             WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) BETWEEN 18 AND 49 THEN '18-49'
#             WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) >= 50 THEN '50+'
#         END AS AgeGroup,
#         i.DESCRIPTION AS Vaccine,
#         COUNT(DISTINCT p.Id) AS ImmunizedPatients
#     FROM
#         immunizations i
#     JOIN
#         patients p ON i.PATIENT = p.Id
#     WHERE
#         i.DESCRIPTION LIKE '%flu%' OR i.DESCRIPTION LIKE '%HPV%'
#     GROUP BY
#         AgeGroup, Vaccine
#     ORDER BY
#         AgeGroup, Vaccine;
#     """
# )

# cursor.execute(
#     """
#     SELECT
#         CASE
#             WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) < 18 THEN '0-17'
#             WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) BETWEEN 18 AND 49 THEN '18-49'
#             WHEN (strftime('%Y', 'now') - strftime('%Y', p.BIRTHDATE)) >= 50 THEN '50+'
#         END AS AgeGroup,
#         c.DESCRIPTION AS Condition,
#         COUNT(DISTINCT p.Id) AS PatientsWithCondition
#     FROM
#         conditions c
#     JOIN
#         patients p ON c.PATIENT = p.Id
#     WHERE
#         c.DESCRIPTION LIKE '%flu%' OR
#         c.DESCRIPTION LIKE '%HPV%'
#     GROUP BY
#         AgeGroup, Condition
#     ORDER BY
#         AgeGroup, Condition;
#     """
# )

cursor.execute(
    """
    SELECT 
    COUNT(*) AS condition_count, 
    conditions.DESCRIPTION,
    CASE
    WHEN (strftime('%Y', 'now') - strftime('%Y', patients.BIRTHDATE)) BETWEEN 0 AND 17 THEN '0-17'
    WHEN (strftime('%Y', 'now') - strftime('%Y', patients.BIRTHDATE)) BETWEEN 18 AND 34 THEN '18-34'
    WHEN (strftime('%Y', 'now') - strftime('%Y', patients.BIRTHDATE)) BETWEEN 35 AND 50 THEN '35-50'
    WHEN (strftime('%Y', 'now') - strftime('%Y', patients.BIRTHDATE)) BETWEEN 51 AND 69 THEN '51-69'
    ELSE '70+' END AS age_group,
    patients.ETHNICITY
    FROM conditions
    JOIN patients ON conditions.PATIENT = patients.Id
    GROUP BY conditions.DESCRIPTION, age_group, patients.ETHNICITY
    """
)

rows = cursor.fetchall()

for row in rows:
    print(row)
