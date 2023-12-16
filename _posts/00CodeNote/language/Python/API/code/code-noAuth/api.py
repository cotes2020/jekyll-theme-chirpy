from flask import Flask, jsonify, make_response, request

api = Flask(__name__)

user1 = {
    "username": "gh0st",
    "Firstname Lastname": "William L. Simon",
    "password": "",
    "Mother’s Favorite Search Engine": "Searx",
}

user2 = {
    "username": "jet-setter",
    "Firstname Lastname": "Frank Abignale",
    "password": "r0u7!nG",
    "Mother’s Favorite Search Engine": "Bing",
}

user3 = {
    "username": "kvothe",
    "Firstname Lastname": "Patrick Rothfuss",
    "password": "3##Heel7sa*9-zRwT",
    "Mother’s Favorite Search Engine": "Duck Duck Go",
}

user4 = {
    "username": "tpratchett",
    "Firstname Lastname": "Terry Pratchett",
    "password": "Thats Sir Terry to you!",
    "Mother’s Favorite Search Engine": "Google",
}

user5 = {
    "username": "lmb",
    "Firstname Lastname": "Lois McMaster Bujold",
    "password": "null",
    "Mother’s Favorite Search Engine": "Yandex",
}

userlist = {
    "gh0st": user1,
    "jet-setter": user2,
    "kvothe": user3,
    "tpratchett": user4,
    "lmb": user5,
}


# operation on the all user information
@api.route("/api/userlist", methods=["GET", "POST"])
def for_userlist():
    if request.method == "GET":
        return make_response(jsonify(userlist), 200)
    if request.method == "POST":
        # ensure we get the response form the request
        content = request.json
        user_username = content["username"]
        userlist[user_username] = content
        user_new = userlist.get(user_username, {})
        return make_response(jsonify(user_new), 201)


<<<<<<< HEAD
# operation on a singal user information
=======
# operation on a single user information
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
@api.route("/api/userlist/<user_username>", methods=["GET", "PUT", "DELETE"])
def singal_user(user_username):
    if request.method == "GET":
        user = userlist.get(user_username, {})
        if user:
            return make_response(jsonify(user), 200)
        else:  # username not exits
            return make_response(jsonify({}), 404)
    if request.method == "PUT":
        content = request.json
        userlist[user_username] = content
        user = userlist.get(user_username, {})
        return make_response(jsonify(user), 200)
    if request.method == "DELETE":
        if user_username in userlist.keys():
            del userlist[user_username]
            return make_response(jsonify({}), 204)
        else:  # username not exits
            return make_response(jsonify({}), 404)


if __name__ == "__main__":
    api.run()
