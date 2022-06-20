__all__ = ["Calculator"]


class Calculator:
    def __init__(self):
        """
        This function initializes the Calculator class setting it's
        memory to 0, and then prints the value in memory
        Example:
        to create a calculator object called calc
        >>> calc=Calculator()
        0
        """
        self.state = 0
        print(self.state)

    def add(self, a: float) -> float:
        """
        This function performs addition of the value stored in memory with a,
        and saves it as a new value in memory, then it prints the value in memory
        before that it checks that a is a number using isnumber(), if it is not
        it returns an error
        Example:
        If the value in memory is 0 and the object is called calc
        >>> calc=Calculator()
        0
        >>> calc.add(5)
        5.0
        """
        a = self.is_number(a)
        self.state += a
        return self.state

    def subs(self, a: float) -> float:
        """
        This function performs substraction of the value stored in memory with a,
        and saves it as a new value in memory, then it prints the value in memory
        before that it checks that a is a number using isnumber(), if it is not
        it returns an error
        Example:
        >>> calc=Calculator()
        0
        >>> calc.subs(5)
        -5.0
        """
        a = self.is_number(a)
        self.state -= a
        return self.state

    def mul(self, a: float) -> float:
        """
        This function performs multiplication of the value stored in memory with a,
        and saves it as a new value in memory, then it prints the value in memory
        before that it checks that a is a number using isnumber(), if it is not
        it returns an error
        Example:
        If the value in memory is 5 and the object is called calc
        >>> calc=Calculator()
        0
        >>> calc.add(5)
        5.0
        >>> calc.mul(5)
        25.0
        """
        a = self.is_number(a)
        self.state *= a
        return self.state

    def div(self, a: float) -> float:
        """
        This function performs addition of the value stored in memory with a,
        and saves it as a new value in memory, then it prints the value in memory
        before that it checks that a is a number using isnumber(), if it is not
        it returns an error, the function also returns an error if a=0
        Example:
        If the value in memory is 10 and the object is called calc
        >>> calc=Calculator()
        0
        >>> calc.state=10
        >>> calc.div(5)
        2.0
        """
        a = self.is_number(a)
        if abs(a) > 0:
            self.state /= a
            return self.state
        else:
            self.reset()
            print(
                "Division by zero is not defined, resetting to last value in memory ...."
            )

    def reset(self) -> float:
        """
        This function resets the memory of the calculator, it sets the current value to 0
        Example:
        If the value in memory is 10 and the object is called calc
        >>> calc=Calculator()
        0
        >>> calc.state=12
        >>> calc.reset()
        0
        """
        self.state = 0
        return self.state

    def root(self, n: float) -> float:
        """
        This function performs the nth root of the value stored in memory,
        and saves it as a new value in memory, then it prints the value in memory
        before that it checks that n is a number using isnumber()
        Example:
        If the value in memory is 100 and the object is called calc
        >>> calc=Calculator()
        0
        >>> calc.state=100
        >>> calc.root(2)
        10.0
        """
        n = self.is_number(n)
        if (n < 0) and (self.state == 0):
            self.reset()
            print(
                "Division by zero is not defined, resetting to last value in memory ...."
            )
            return self.state
        if (0 < n < 0.001) and (self.state > 1):
            self.reset()
            print(
                "The numerical value is too big for this calculator to handle, resetting to last value in memory ...."
            )
            return self.state
        self.state **= 1 / n

        return self.state

    def is_number(self, a: float) -> float:
        """
        It checks whether a is a number, if the number is pass as a string it converts it to a float,
        if not it raises an error
        Example:
        >>> calc=Calculator()
        0
        >>> calc.is_number(5)
        5.0
        >>> calc.is_number('6')
        6.0
        >>> calc.is_number('n')
        The calculator just handles numerical values, you used a <class 'str'>
        """
        if (a == float) or (a == int):
            pass
        else:
            try:
                a = float(a)
                return a
            except ValueError:
                print(
                    f"The calculator just handles numerical values, you used a {type(a)}"
                )


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
