# NOTES ABOUT PYTHON CLASSES

## Class Structure

A Python class is defined using the `class` keyword:

```python
class MyClass(ParentClass):
    # Class attributes
    class_variable = value

    def __init__(self, value):
        # Instance attributes
        self.value = value

    def my_method(self):
        # Method
        pass


```

### super()

 - super allows you to access parent implimentaiton

 - this allows for things like accesssing overridden classes or values 

 * in this example we using the constructor from our parent class and our child class!


 ```python
    def __init__(self, config: MyTeleopConfig):
        super().__init__(config) #<-- parent constructor done here!

        self.device_id = config.device_id
        self.sensitivity = config.sensitivity

 ```

