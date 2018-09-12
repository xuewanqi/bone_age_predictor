import random
import string

import app
import errors


class User(app.db.Model):
    id = app.db.Column(app.db.Integer, primary_key=True)
    name = app.db.Column(app.db.String(80), unique=True, nullable=False)
    token = app.db.Column(app.db.String(64), unique=True, nullable=False)
    total_quota = app.db.Column(app.db.Integer, unique=False, nullable=False)
    quota_left = app.db.Column(app.db.Integer, unique=False, nullable=False)

    def __repr__(self):
        return "[User]\t{}\t{}\t{}/{}".format(self.name, self.token, self.quota_left, self.total_quota)

    def as_dict(self):
        return {
            "name": self.name,
            "token": self.token,
            "total_quota": self.total_quota,
            "quota_left": self.quota_left
        }

    def make_quotas_dict(self):
        return {
            "total_quota": self.total_quota,
            "quota_left": self.quota_left
        }

    def edit_from_info(self, info):
        if info.get("name") is not None:
            self.name = info["name"]
        if info.get("token") is not None:
            self.token = info["token"]
        if info.get("total_quota") is not None:
            self.total_quota = info["total_quota"]
        if info.get("quota_left") is not None:
            self.quota_left = info["quota_left"]
        return

    @staticmethod
    def construct_from_info(info):
        return User(
            name=info["name"],
            token=info["token"],
            total_quota=info["total_quota"],
            quota_left=info["quota_left"]
        )


def initialize():
    app.db.create_all()
    return


def get_user_quotas(token):
    return get_user_by_token(token).make_quotas_dict()


def decrement_user_quota(token):
    user = get_user_by_token(token=token)

    if user.quota_left > 0:
        user.quota_left = user.quota_left - 1
        commit_database()
        return user.make_quotas_dict()
    else:
        raise errors.UserAuthenticationError("No more request quota.")

    return


def get_all_users_info():
    users_info = []
    users = User.query.all()
    for user in users:
        users_info.append(user.as_dict())
    return users_info


def get_user_info(name):
    return get_user_by_name(name).as_dict()


def add_user_from_info(info):
    valid_info = value_check_all_info(info)
    user = User.construct_from_info(valid_info)
    app.db.session.add(user)
    commit_database()
    return user.as_dict()


def update_user_from_info(name, info):
    user = get_user_by_name(name)
    valid_partial_info = value_check_info_against_user(info, user)
    user.edit_from_info(valid_partial_info)
    commit_database()
    return user.as_dict()


def delete_user_by_name(name):
    user = get_user_by_name(name=name)
    app.db.session.delete(user)
    commit_database()
    return user.as_dict()


def commit_database():
    try:
        app.db.session.commit()
    except Exception as user_conflict_error:
        if user_conflict_error.args[0][-4:] == "name":
            message = "Name conflict with an existing user in the database."
        elif user_conflict_error.args[0][-5:] == "token":
            message = "Token conflict with an existing user in the database."
        else:
            message = "Name or token conflict with an existing user."
        app.db.session.rollback()

        raise errors.UserConflictError(message=message)

    return


def get_user_by_name(name):
    user = User.query.filter_by(name=name).first()
    if user is None:
        raise errors.UserNotFoundError("No user associated with that name.")
    return user


def get_user_by_token(token):
    user = User.query.filter_by(token=token).first()
    if user is None:
        raise errors.UserNotFoundError("No user associated with your token.")
    return user


def value_check_all_info(info):
    if info.get("name") is None:
        raise errors.UserNameError("The name field cannot be blank.")

    if info.get("token") is None:
        info["token"] = generate_token()

    if info.get("total_quota") is None:
        info["total_quota"] = 10

    if info.get("quota_left") is None:
        info["quota_left"] = info["total_quota"]

    return {
        "name": value_check_name(raw_name=info["name"]),
        "token": value_check_token(raw_token=info["token"]),
        "total_quota": value_check_total_quota(raw_total_quota=info["total_quota"]),
        "quota_left": value_check_quota_left(raw_quota_left=info["quota_left"], clean_total_quota=info["total_quota"])
    }


def value_check_info_against_user(info, user):
    result = {}

    if info.get("name") is not None:
        result["name"] = value_check_name(raw_name=info["name"])

    if info.get("token") is not None:
        result["token"] = value_check_token(raw_token=info["token"])

    if info.get("total_quota") is not None:
        result["total_quota"] = value_check_total_quota(raw_total_quota=info["total_quota"])

    if info.get("quota_left") is not None:
        if result.get("total_quota") is None:
            result["quota_left"] = value_check_quota_left(raw_quota_left=info["quota_left"],
                                                          clean_total_quota=user.total_quota)
        else:
            result["quota_left"] = value_check_quota_left(raw_quota_left=info["quota_left"],
                                                          clean_total_quota=result["total_quota"])

    return result


def value_check_name(raw_name):

    if not isinstance(raw_name, str):
        raise errors.UserNameError("A user's name must be a string.")

    processed_name = raw_name.strip().lower()

    if processed_name == "":
        raise errors.UserNameError("A user's name cannot cannot comprise only of whitespace.")

    if len(processed_name) > 80:
        raise errors.UserNameError("A user's name cannot be more than 80 characters long.")

    return processed_name


def value_check_token(raw_token):
    if not isinstance(raw_token, str):
        raise errors.UserTokenError("The provided token must be a string.")

    processed_token = raw_token.strip()

    if processed_token == "":
        raise errors.UserTokenError("The token cannot comprise only of whitespace.")

    if len(processed_token) > 64:
        raise errors.UserTokenError("The token cannot be longer than 64 characters.")

    return processed_token


def value_check_total_quota(raw_total_quota):
    if not isinstance(raw_total_quota, int):
        raise errors.UserTotalQuotaError("The provided total_quota may only be an integer.")

    elif raw_total_quota < 0:
        raise errors.UserTotalQuotaError("The total_quota field cannot contain a negative integer.")

    return raw_total_quota


def value_check_quota_left(raw_quota_left, clean_total_quota):
    if not isinstance(raw_quota_left, int):
        raise errors.UserQuotaLeftError("The provided quota_left may only be an integer.")

    elif raw_quota_left < 0:
        raise errors.UserQuotaLeftError("The quota_left cannot be a negative integer.")

    elif raw_quota_left > clean_total_quota:
        raise errors.UserQuotaLeftError("The quota_left must be less than the total_quota of {}."
                                        .format(clean_total_quota))

    return raw_quota_left


def generate_token():
    """
    Generates a random 32-character token. Tokens may only include lowercase letters, uppercase letters, and digits.
    They are hence case-sensitive.
    :return: 32-character case-sensitive token as a string.
    """
    candidate_characters = string.ascii_letters + string.digits
    new_token = ""
    for i in range(0, 32):
        new_token += random.choice(candidate_characters)
    return new_token
