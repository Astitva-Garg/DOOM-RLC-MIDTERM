import random

"""
We'll try to understand classes in python.
Check the resources on google classroom to ensure you have gone through everything expected.

"""

###### THESE LISTS HAVE ALREADY BEEN DEFINED FOR YOU ###############
engineer_roster = []  # A list of all instantiated engineer objects
sales_roster = []  # List of all instantiated sales objects
branchmap = {  # A dictionary of dictionaries -> Maps branchcodes to cities and branch names
    0: {"city": "NYC", "name": "Hudson Yards"},
    1: {"city": "NYC", "name": "Silicon Alley"},
    2: {"city": "Mumbai", "name": "BKC"},
    3: {"city": "Tokyo", "name": "Shibuya"},
    4: {"city": "Mumbai", "name": "Goregaon"},
    5: {"city": "Mumbai", "name": "Fort"},
}
####################################################################


class Employee:
    name: str
    age: int
    ID: int
    city: str
    branches: list[
        int
    ]  # This is a list of branches (as branch codes) to which the employee may report
    salary: int

    def __init__(self, name, age, ID, city, branchcodes, salary=None):
        self.name = name
        self.age = age
        self.ID = ID
        self.city = city
        self.branches = branchcodes
        if salary is not None:
            self.salary = salary
        else:
            self.salary = 10_000

    def change_city(self, new_city: str) -> bool:
        if self.city != new_city:
            self.city = new_city
            return True  # Change the city
        return False  # Return true if city change, successful, return false if city same as old city
        pass

    def migrate_branch(self, new_code: int) -> bool:
        if len(self.branches) != 1:
            return False  # Should work only on those employees who have a single
        # branch to report to. Fail for others.
        if branchmap[self.branches[0]]["city"] != branchmap[new_code]["city"]:
            return False  # Change old branch to new if it is in the same city, else return false.
        self.branches = [new_code]
        return True
        pass

    def increment(self, increment_amt: int) -> None:
        self.salary += increment_amt
        pass


class Engineer(Employee):
    position: str  # Position in organization Hierarchy
    positions_list: list[str]  # List of all available positions

    positions_list = ["Junior", "Senior", "Team Lead", "Director"]

    def __init__(
        self, name, age, ID, city, branchcodes, position="Junior", salary=None
    ):
        # Call the parent's constructor
        super().__init__(name, age, ID, city, branchcodes, salary)

        if (
            position in self.positions_list
        ):  # Check if position is one of  "Junior", "Senior", "Team Lead", or "Director"
            self.position = position  # Only then set the position.
        else:
            raise ValueError("Invalid Position")

    def increment(self, amt: int) -> None:
        super().increment(
            amt // 10 + amt
        )  # While other functions are the same for an engineer,
        # and increment to an engineer's salary should add a 10% bonus on to "amt"
        pass

    def promote(self, position: str) -> bool:
        if position not in self.positions_list:
            return False  # Return false for a demotion or an invalid promotion
        if self.positions_list.index(position) <= self.positions_list.index(
            self.position
        ):
            return False  # Promotion can only be to a higher position and
        self.increment(
            (3 * super().salary) // 10
        )  # it should call the increment function with 30% of the present salary
        return True  # as "amt". Thereafter return True.
        pass


class Salesman(Employee):
    """
    This class is to be entirely designed by you.

    Add increment (this time only a 5% bonus) and a promotion function
    This time the positions are: Rep -> Manager -> Head.

    Add an argument in init which tracks who is the superior
    that the employee reports to. This argument should be the ID of the superior
    It should be None for a "Head" and so, the argument should be optional in init.
    """

    def increment(self, amt: int) -> None:
        super().increment(amt // 20 + amt)
        #  increment to a Salesman's salary should add a 5% bonus on to "amt"
        pass

    # An extra member variable!
    superior: int  # EMPLOYEE ID of the superior this guy reports to
    positions_list: list[str]

    positions_list = ["Rep", "Manager", "Head"]

    def __init__(
        self,
        name,
        age,
        ID,
        city,
        branchcodes,
        position="Rep",
        salary=None,
        superior=None,
    ):  # Call the parent's constructor
        super().__init__(name, age, ID, city, branchcodes, salary)

        if position in self.positions_list:  # Check if position is valid
            self.position = position  # Only then set the position.
        else:
            raise ValueError("Invalid Position")
        pass

    def promote(self, position: str) -> bool:
        if position not in self.positions_list:
            return False  # Return false for a demotion or an invalid promotion
        if self.positions_list.index(position) <= self.positions_list.index(
            self.position
        ):
            return False  # Promotion can only be to a higher position and
        self.increment(
            (3 * super().salary) // 10
        )  # it should call the increment function with 30% of the present salary
        return True  # as "amt". Thereafter return True.
        pass

    # def increment

    def find_superior(self):
        superior_found = next(
            (employee for employee in sales_roster if employee.ID == self.superior),
            None,
        )
        return (
            (superior_found.ID, superior_found.name)
            if superior_found is not None
            else (None, None)
        )
        # Return the employee ID and name of the superior
        # Report a tuple of None, None if no superior.
        pass

    def add_superior(self) -> bool:
        try:
            superior_position = self.positions_list[
                self.positions_list.index(self.position) + 1
            ]  # Add superior of immediately higher rank.
        except IndexError:
            return False  # If superior doesn't exist return false,
        else:
            self.superior = random.choice(
                [
                    person
                    for person in sales_roster
                    if person.position == superior_position
                ]
            ).ID  # taking a random choice of a person in sales roster whose position is just ahead self
            return True

    def migrate_branch(self, new_code: int) -> bool:
        if new_code in super().branches:
            return False
        super().branches.append(
            new_code
        )  # This should simply add a branch to the list; even different cities are fine
        return True
