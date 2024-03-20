input_queries = []
output_queries = []

with open(
    "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/data.txt",
    "r",
) as f:
    text = f.read()

lines = text.split("\n")

current_section = None

for line in lines:
    if line.startswith("input:"):
        current_section = "input"
        input_queries.append(line[len("input:") :].strip())
    elif line.startswith("output:"):
        current_section = "output"
        output_queries.append(line[len("output:") :].strip())
    else:
        if current_section == "input" and input_queries:
            input_queries[-1] += " " + line.strip()
        elif current_section == "output" and output_queries:
            output_queries[-1] += " " + line.strip()

print(input_queries)
print(output_queries)
