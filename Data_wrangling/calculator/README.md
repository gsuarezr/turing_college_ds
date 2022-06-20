
This is a calculator done for the Turing college sprint 1. It can perform the following actions

- Addition / Subtraction.
- Multiplication / Division.
- Take (n) root of a number.
- Reset memory (Calculator must have its own memory, meaning it should manipulate its starting number 0 until it is reset.).

After each action the result is stored in memory and returned


# Installation:

To install this package you need to use the following
commands:

Windows
```
pip install -i https://test.pypi.org/simple/ gerardocalc
```

# Usage


To create a new calculator object you can use:

```
from gerardocalc.calculator import Calculator
calc=Calculator()
```

Once the object is created you can perform all of the functions of the calculator:

### Addition:

To add x to the value in memory use:

```
calc.add(x)
```

### Substraction:

To substract x to the value in memory use:

```
calc.subs(x)
```

### Multiplication:

To multiply x to the value in memory use:

```
calc.mul(x)
```

### Division:

To divide the value in memory by x use:

```
calc.div(x)
```

Division by 0 is not defined, it prints a warning a resets the memory

### Nth root:

To obtain the nth root of the value in memory use:

```
calc.div(n)
```
 If the value in memory is 0 and n is negative then it throws a warning and resets the memory


# Dockerized

If you want to run the package from a docker container you can use this [Dockerfile](Dockerfile), to use it just input this command from the same directory that contains the docker file

```
docker build -t gerardocalc .
docker run -i -t gerardocalc
```


