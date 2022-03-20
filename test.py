class Person:
    def __init__(self,name,age):
        self.name = name
        self.__age = age
    def __test(self):
        print("这是父类的私有方法")
    def test(self):
        self.__test()
        print("这是父类的公有方法")
    def setAge(self,age):
        self.__age = age
    def getAge(self):
        return self.__age
class Student(Person):
    def __init__(self,school,name,age):
        super(Student, self).__init__(name=name,age=age)
        self.school = school
    def stuTest(self):
        super().test()
        print("所在学校为：",self.school)
stu = Student("一中","tom",12)
stu.stuTest()
print("学生的姓名是：",stu.name)
print("学生的年龄是：",stu.getAge())