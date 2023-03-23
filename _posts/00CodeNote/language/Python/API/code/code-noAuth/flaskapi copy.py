from flask import Flask, json

companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]

# initialize Flask
# create a Flask object, and assign it to the variable name api.
api = Flask(__name__)

# declare a route for endpoint.
# When a consumer visits /companies using a GET request, the list of two companies will be returned.


@api.route("/companies", methods=["GET"])
def get_companies():
    return json.dumps(companies)
    # status code wasn’t required because 200 is Flask’s default.


@api.route("/companies", methods=["POST"])
def post_companies():
    return json.dumps({"success": True}), 201


if __name__ == "__main__":
    api.run()
