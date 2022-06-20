# List all subdirectories using os.listdir
import csv 


def passwd_to_dict(filename):
    with open(filename, 'r') as f:
        data =list(csv.reader(f,delimiter=','))
    data= [element for lista in data for element in lista if not element.startswith('#')] # list flattening and filtering comments
    data={item.split(':')[0]:item.split(':')[2] for item in data}
    return data

print(passwd_to_dict('passwd.txt'))
