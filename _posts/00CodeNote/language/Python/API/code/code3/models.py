from flask_login import UserMixin

from . import db


class User(db.Model, UserMixin):
    db_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    db_username = db.Column(db.String(256), unique=True)
    db_flname = db.Column(db.String(100))
    db_password = db.Column(db.String(100))
    db_engine = db.Column(db.String(100))

    def get_id(self):
        return self.db_id


# echo '{"username":"a",  "Firstname Lastname":"ab", "password":"123",  "Mother’s Favorite Search Engine":"c"}' | http POST http://127.0.0.1:5000/adduser
