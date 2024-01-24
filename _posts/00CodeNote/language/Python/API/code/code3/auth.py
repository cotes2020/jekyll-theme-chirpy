from flask import (
    Blueprint,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from . import db
from .models import User

auth = Blueprint("auth", __name__)


@auth.route("/adduser", methods=["GET"])
def adduser():
    # return render_template('adduser.html')
    return make_response("Add user Page", 200)


# http GET https://127.0.0.1:5000/adduser


@auth.route("/adduser", methods=["POST"])
def adduser_post():
    # got the adduser info
    content = request.json
    username = content["username"]
    fLname = content["Firstname Lastname"]
    password = content["password"]
    engine = content["Mother’s Favorite Search Engine"]

    # check the username, usery the database, get the first one
    user = User.query.filter_by(db_username=username).first()
    if user:
        print("username already Exists")
        return make_response(
            jsonify(
                {
                    "status": 400,
                    "Add user": False,
                    "Error Message": "username not available or already exsisted",
                }
            ),
            400,
        )
    new_user = User(
        db_username=username,
        db_flname=fLname,
        db_password=generate_password_hash(password, method="sha256"),
        db_engine=engine,
    )
    db.session.add(new_user)
    db.session.commit()
    # return render_template('adduser.html')
    # return redirect(url_for('auth.login'))
    return make_response(jsonify({"status": 200, "new_user": content}), 201)


# echo '{"username":"a",  "Firstname Lastname":"x", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST https://127.0.0.1:5000/adduser
# echo '{"username":"ab",  "Firstname Lastname":"doubleuser", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST https://127.0.0.1:5000/adduser


@auth.route("/login", methods=["GET"])
def login():
    # return render_template('login.html')
    return make_response("User Login Page", 200)


# http GET https://127.0.0.1:5000/login


@auth.route("/login", methods=["POST"])
def login_post():
    content = request.json
    username = content["username"]
    password = content["password"]
    # remember = True
    user = User.query.filter_by(db_username=username).first()
    # correct username and passwd:
    if user and check_password_hash(user.db_password, password):
        # login_user(user, remember=remember)
        login_user(user)
        username = user.db_username
        flname = user.db_flname
        engine = user.db_engine
        userinfo = {
            "username": username,
            "Firstname Lastname": flname,
            "Mother’s Favorite Search Engine": engine,
        }
        # return redirect(url_for('main.profile'))
        return make_response(
            jsonify(
                {
                    "status": 200,
                    "Successful login": True,
                    "current_user.is_authenticated": current_user.is_authenticated,
                    "userinfo": userinfo,
                }
            ),
            200,
        )
    # wrong username and passwd:
    else:
        # return redirect("auth.login")
        return make_response(
            jsonify({"status": 401, "reason": "Username or Password Error"}), 401
        )


# echo '{"username":"a", "password":"123"}' | http POST https://127.0.0.1:5000/login
# echo '{"username":"a", "password":"wrongpasswd"}' | http POST https://127.0.0.1:5000/login


@auth.route("/logout", methods=["GET"])
# @login_required
def logout():
    logout_user()
    return make_response(jsonify({"status": 200, "Session": "Successful logout"}), 200)
    # if current_user.is_authenticated:
    #    logout_user()
    #    # return redirect('main.index')
    #    return make_response(jsonify({"status": 200, "Session": "Successful logout"}), 200)
    # else:
    #    return make_response(jsonify({"status": 403, "Session": "you are not even login"}), 403)


# http GET https://127.0.0.1:5000/logout
