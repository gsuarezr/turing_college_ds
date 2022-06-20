from typing import Dict
Item = str
Price = int
menu: Dict[str, int] = {'Quesadillas': 15, 'Chilaquiles': 20,
                        'Tacos de pastor': 18, 'Tacos de suadero': 18, 'Taco de chistorra': 20}

condition = True
total = 0


def show():
    print('============= Menu ===============')
    for i, v in menu.items():
        print(i, v)
    print('==================================')


while condition:
    try:
        order: str = input("Please input your order :")
        if order == 'show':
            show()
        elif order == '':
            condition = False
            print(f'Your total is {total}')
        else:
            total += menu[order]
            print(order+f' costs {menu[order]}, the total is now {total}')
    except KeyError:
        print('We don\'t sell that, if you want to see the menu write show')
