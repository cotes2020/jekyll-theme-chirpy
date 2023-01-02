import dataset
from flask import Flask, jsonify, make_response, request
from flask_bcrypt import Bcrypt

api = Flask(__name__)
bcrypt = Bcrypt(api)
db = dataset.connect("sqlite:///api.db")
table = db["userlist"]

# function that fetch the db


def fetch_one_db(user_username):
    return table.find_one(username=user_username)


def fetch_all_db():
    usersinfo = []
    for user in table:
        usersinfo.append(user)
    return usersinfo


# add the user in to the table
@api.route("/api/creatdefaultuser", methods=["GET"])
def db_creatdefaultuser():
    table.insert(
        {
            "username": "gh0st",
            "Firstname Lastname": "William L. Simon",
            "password": "",
            "Mother’s Favorite Search Engine": "Searx",
        }
    )
    table.insert(
        {
            "username": "jet-setter",
            "Firstname Lastname": "Frank Abignale",
            "password": "r0u7!nG",
            "Mother’s Favorite Search Engine": "Bing",
        }
    )
    table.insert(
        {
            "username": "kvothe",
            "Firstname Lastname": "Patrick Rothfuss",
            "password": "3##Heel7sa*9-zRwT",
            "Mother’s Favorite Search Engine": "Duck Duck Go",
        }
    )
    table.insert(
        {
            "username": "tpratchett",
            "Firstname Lastname": "Terry Pratchett",
            "password": "Thats Sir Terry to you!",
            "Mother’s Favorite Search Engine": "Google",
        }
    )
    table.insert(
        {
            "username": "lmb",
            "Firstname Lastname": "Lois McMaster Bujold",
            "password": "null",
            "Mother’s Favorite Search Engine": "Yandex",
        }
    )
    return make_response(jsonify(fetch_all_db()), 200)


# operation on the all user information
@api.route("/api/userlist", methods=["GET", "POST"])
def for_userlist():
    # GET all the user information
    if request.method == "GET":
        return make_response(jsonify(fetch_all_db()), 200)
    # POST new user information
    if request.method == "POST":
        content = request.json
        table.insert(content)
        user_username = content["username"]
        return make_response(jsonify(fetch_one_db(user_username)), 201)


# operation on a singal user information
@api.route("/api/userlist/<user_username>", methods=["GET", "PUT", "DELETE"])
def singal_user(user_username):
    # GET one user information
    if request.method == "GET":
        user = fetch_one_db(user_username)
        if user:
            return make_response(jsonify(user), 200)
        else:  # username not exits
            return make_response(jsonify({}), 404)
    # PUT update to one user information
    if request.method == "PUT":
        content = request.json
        table.update(content, ["username"])
        user = fetch_one_db(user_username)
        return make_response(jsonify(user), 200)
    # DELETE one user information
    if request.method == "DELETE":
        table.delete(username=user_username)
        return make_response(jsonify({}), 204)


if __name__ == "__main__":
    api.run()
