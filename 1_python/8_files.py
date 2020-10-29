import csv

# Text file reading and writing
txt_filename = './fruits_input.txt'
FAV_FRUITS = ['Apple', 'Pepper', 'Orange', 'Watermelon', 'Tomatoes']

with open(txt_filename) as f:
    fruits = f.read().split('\n')
    found_fruits = [fruit for fruit in fruits if fruit in FAV_FRUITS]
    print(found_fruits)
    f.close()

# CSV file reading and writing
csv_filename = './sample_csv_file.csv'

with open(csv_filename) as f:
    data = list(csv.reader(f, delimiter=','))
    headers = data[0]
    users = []

    for row in data[1:]:
        user = {}
        for j in range(len(headers)):
            user[headers[j]] = row[j]
        users.append(user)

    print(users)
    f.close()

# Challenge
csv_filename = './S_P500_Stock_data.csv'

with open(csv_filename) as f:
    data = list(csv.reader(f, delimiter=','))
    print(f'Headers: {data[0]}')
    selected = data[1:6]
    print(selected)
    f.close()
