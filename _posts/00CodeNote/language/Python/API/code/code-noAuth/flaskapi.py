from flask import Flask, json


class User:
    def __init__(self, username, FLname, password, Engine):
        self.username = username
        self.FLname = FLname
        self.password = password
        self.Engine = Engine

    def serialize(self):
        return {
            "username": self.username,
            "Firstname Lastname": self.FLname,
            "password": self.password,
            "Mother’s Favorite Search Engine": self.Engine,
        }

    # def get_usr_name(self):
    #     long_name = f"{self.year} {self.manufacturer} {self.model}"
    #     return long_name.title()

    # def read_odometer(self):
    #     print(f"This car has {self.odometer_reading} miles on it.")

    # def update_odometer(self, mileage):
    #     if mileage >= self.odometer_reading:
    #         self.odometer_reading = mileage else:
    #         print("You can't roll back an odometer!")

    # def increment_odometer(self, miles):
    #     self.odometer_reading += miles


usr1 = {
    "username": "gh0st",
    "Firstname Lastname": "William L. Simon",
    "password": "",
    "Mother’s Favorite Search Engine": "Searx",
}

usr2 = {
    "username": "jet-setter",
    "Firstname Lastname": "Frank Abignale",
    "password": "r0u7!nG",
    "Mother’s Favorite Search Engine": "Bing",
}

usr3 = {
    "username": "kvothe",
    "Firstname Lastname": "Patrick Rothfuss",
    "password": "3##Heel7sa*9-zRwT",
    "Mother’s Favorite Search Engine": "Duck Duck Go",
}

usr4 = {
    "username": "tpratchett",
    "Firstname Lastname": "Terry Pratchett",
    "password": "Thats Sir Terry to you!",
    "Mother’s Favorite Search Engine": "Google",
}

usr5 = {
    "username": "lmb",
    "Firstname Lastname": "Lois McMaster Bujold",
    "password": "null",
    "Mother’s Favorite Search Engine": "Yandex",
}

usrlist = {
    "gh0st": usr1,
    "jet-setter": usr2,
    "kvothe": usr3,
    "tpratchett": usr4,
    "lmb": usr5,
}
# print(usrlist)

# initialize Flask
api = Flask(__name__)

# declare a route for endpoint.
# When a consumer visits /usrlist using a GET request, the list of two usrlist will be returned.
@api.route("/usr/usrlist", methods=["GET"])
def get_usrlist():
    return json.dumps(usrlist)
    # status code wasn’t required because 200 is Flask’s default.


@api.route("/usr/usrlist", methods=["POST"])
def post_usrlist():
    return json.dumps({"success": True}), 201


if __name__ == "__main__":
    api.run()
