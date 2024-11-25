import csv


def get_unique_airways(file_path):
    unique_airways = set()  # Use a set to store unique airways

    # Open the CSV file
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')  # Assuming tab-separated values
        print("Column names:", reader.fieldnames)
        for row in reader:
            airways = row['Airways'].split()  # Split the 'Airways' column by space
            unique_airways.update(airways)  # Add each airway to the set

    return sorted(unique_airways)  # Return sorted list of unique airways


# Specify the path to your CSV file
file_path = 'airways_db.csv'

# Get unique airways
unique_airways = get_unique_airways(file_path)

# Print the results
# print("Unique Airways:")
# for airway in unique_airways:
#     print(airway)
print(len(unique_airways))
