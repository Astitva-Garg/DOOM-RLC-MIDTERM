"""
This is where the actual working of the program will happen!
We'll be Querying stuff for testing whether your classes were defined correctly

Whenever something inappropriate happens (like a function returning false in people.py),
raise a Value Error.
"""

from people import *  # import everything!

if __name__ == "__main__":  # Equivalent to int main() {} in C++.
    last_input = 99
    while last_input != 0:
        last_input = int(input("Please enter a query number:"))

        if last_input == 1:
            name = input("Name:")
            age = int(input("Age:"))
            ID = int(input("ID:"))
            city = input("City:")
            branchcodes = [int(i) for i in input("Branch(es):").split(",")]
            position = input("Position:")
            salary = int(input("Salary: "))
            # Create a new Engineer with given details.
            engineer = Engineer(
                name, age, ID, city, branchcodes, position, salary
            )  # Change this

            engineer_roster.append(
                engineer
            )  # Add him to the list! See people.py for definiton

        elif last_input == 2:
            name = input("Name:")
            age = int(input("Age:"))
            ID = int(input("ID:"))
            city = input("City:")
            branchcodes = input("Branch(es):").split(",")
            position = input("Position:")
            salary = int(input("Salary:"))
            superior = int(input("ID of Superior:"))
            salesman = Salesman(
                name, age, ID, city, branchcodes, position, salary, superior
            )  # Gather input to create a Salesperson
            sales_roster.append(salesman)
            # Then add them to the roster
            pass

        elif last_input == 3:
            ID = int(input("ID: "))
            # Print employee details for the given employee ID that is given.

            found_employee = None
            for employee in engineer_roster + sales_roster:
                if employee.ID == int(ID):
                    found_employee = employee
                    break

            if not found_employee:
                print("No such employee")
            else:
                print(f"Name: {found_employee.name} and Age: {found_employee.age}")
                print(f"City of Work: {found_employee.city}")

                ## Write code here to list the branch names to
                ## which the employee reports as a comma separated list
                ## eg> Branches: Goregaon,Fort
                branches = [branchmap[n]["city"] for n in found_employee.branchcodes]

                ## ???? what comes here??
                print("Branches: ", *branches, sep=",")

                print(f"Salary: {found_employee.salary}")

        elif last_input == 4:
            ID = int(input("Enter Employee ID to migrate/change branch of: "))
            branchcode = int(input("Enter Branchcode to migrate/add Employee to"))
            found_employee = next(
                (emp for emp in sales_roster + engineer_roster if emp.ID == ID), None
            )

            if not found_employee:
                print("Employee not Found")
            else:
                found_employee.migrate_branch(branchcode)
            #### NO IF ELSE ZONE ######################################################
            # Change branch to new branch or add a new branch depending on class
            # Inheritance should automatically do this.
            # There should be no IF-ELSE or ternary operators in this zone
            pass
            #### NO IF ELSE ZONE ENDS #################################################

        elif last_input == 5:
            ID = int(input("Enter Employee ID to promote: "))
            found_employee = next(
                (emp for emp in sales_roster + engineer_roster if emp.ID == ID), None
            )

            if not found_employee:
                print("Employee not Found")
            else:
                found_employee.promote()
            # promote employee to next position

        elif last_input == 6:
            ID = int(input("Enter Employee ID to give increment: "))
            amt = int(input("Enter salary increment amount: "))
            found_employee = next(
                (emp for emp in sales_roster + engineer_roster if emp.ID == ID), None
            )

            if not found_employee:
                print("Employee not Found")
            else:
                found_employee.increment(amt)
            # Increment salary of employee.

        elif last_input == 7:
            ID = int(input("Enter Employee ID to find superior: "))
            found_employee = next(
                (emp for emp in sales_roster if emp.ID == ID), None
            )
            if not found_employee:
                print("Employee not Found")
            else:
                print("Superior ID:" ,found_employee.superior)
            # Print superior of the sales employee.

        elif last_input == 8:
            ID_E = int(input("Enter Employee ID to add superior: "))
            ID_S = int(input("Enter Employee ID of superior: "))
            # Add superior of a sales employee
            found_employee = next(
                (emp for emp in sales_roster if emp.ID == ID_E), None
            )

            if not found_employee:
                print("Employee not Found")
            else:
                found_employee.add_superior(ID_S)
        else:
            raise ValueError("No such query number defined")
