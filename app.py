import admins
import users
import errors
from model_bone_age import BoneAgePredictor

import json
import argparse
import traceback

from flask import Flask, jsonify, request, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_httpauth import HTTPBasicAuth
from passlib.apps import custom_app_context as pwd_context


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
auth = HTTPBasicAuth()

classifier = None


@app.route("/", methods=["GET"])
def show_index_page():
    response_body = {"status": "ok"}
    return jsonify(response_body)


@app.route("/check-quota", methods=["GET"])
def show_user_quota():
    try:
        token = get_token_from_request()
        current_quotas = users.get_user_quotas(token=token)
    except errors.UserAuthenticationError as bad_token_error:
        return errors.unauthorized_response(message=str(bad_token_error))
    except errors.UserNotFoundError:
        return errors.not_found_response("No user associated with your token.")

    if current_quotas["quota_left"] <= 0:
        return errors.unauthorized_response(message="No more request quota.", dict={"quotas": current_quotas})

    response_body = {
        "status": "ok",
        "quotas": current_quotas
    }
    return jsonify(response_body)


@app.route("/echo", methods=["POST"])
def show_echo():
    try:
        token = get_token_from_request()
        users.get_user_by_token(token=token)
        image = parse_info_as_image(request.data)
    except errors.UserAuthenticationError as bad_token_error:
        return errors.unauthorized_response(message=str(bad_token_error))
    except errors.UserNotFoundError:
        return errors.not_found_response("No user associated with your token.")
    except errors.ImageNotFoundError:
        return errors.bad_request_response("Please include an image in your request body.")

    return image


@app.route("/model", methods=["POST"])
def show_model_response():
    try:
        token = get_token_from_request()
        current_quotas = users.get_user_quotas(token=token)
        input_image=request.files.get("image")
        image = parse_info_as_image(input_image)
        gender= request.values.get("gender").encode('utf-8')
    except errors.UserAuthenticationError as bad_token_error:
        return errors.unauthorized_response(message=str(bad_token_error))
    except errors.UserNotFoundError:
        return errors.not_found_response("No user associated with your token.")
    except errors.ImageNotFoundError:
        return errors.bad_request_response("Please include an image in your request body.")

    if current_quotas["quota_left"] <= 0:
        return errors.unauthorized_response(message="No more request quota.", dict={"quotas": current_quotas})
    try:
        new_quotas = users.decrement_user_quota(token=token)
        predictions = classifier.predict(img=image, gender=gender)
    except Exception:
        print('error from classification')
        traceback.print_exc()
        return

    response_body = {
        "status": "ok",
        "quotas": new_quotas,
        "results": predictions
    }
    return jsonify(response_body)


@app.route("/users", methods=["GET"])
@auth.login_required
def show_all_users():
    all_users = users.get_all_users_info()

    response_data = {
        "status": "ok",
        "users": all_users
    }
    return jsonify(response_data)


@app.route("/users/<string:name>", methods=["GET"])
@auth.login_required
def show_single_user(name):
    try:
        user_info = users.get_user_info(name=name)
    except errors.UserNotFoundError:
        return errors.not_found_response("No user associated with that name.")

    response_data = {
        "status": "ok",
        "user": user_info
    }
    return jsonify(response_data)


@app.route("/users", methods=["POST"])
@auth.login_required
def show_add_user_response():
    try:
        clean_info = parse_info_as_json(request.data)
        new_user_info = users.add_user_from_info(info=clean_info)
    except errors.JSONNotFoundError:
        return errors.bad_request_response("New user details are required in a proper JSON request body.")
    except errors.UserInfoError as user_info_error:
        return errors.bad_request_response(message=str(user_info_error))
    except errors.UserConflictError as user_conflict_error:
        return errors.conflict_response(message=str(user_conflict_error))

    response_data = {
        "status": "ok",
        "user": new_user_info
    }
    return make_response(jsonify(response_data), 201)


@app.route("/users/<string:name>", methods=["PUT"])
@auth.login_required
def show_update_user_response(name):
    try:
        clean_info = parse_info_as_json(request.data)
        old_user_info = users.get_user_info(name)
        updated_user_info = users.update_user_from_info(
            name=name, info=clean_info)
    except errors.UserNotFoundError:
        return errors.not_found_response("No user associated with that name.")
    except errors.JSONNotFoundError:
        return errors.bad_request_response("Updated user details are required in a proper JSON request body.")
    except errors.UserInfoError as user_info_error:
        return errors.bad_request_response(message=str(user_info_error))
    except errors.UserConflictError as user_conflict_error:
        return errors.conflict_response(message=str(user_conflict_error))

    response_data = {
        "status": "ok",
        "user": updated_user_info,
        "old_info": old_user_info
    }
    return jsonify(response_data)


@app.route("/users/<string:name>", methods=["DELETE"])
@auth.login_required
def show_delete_user_response(name):
    try:
        deleted_user_info = users.delete_user_by_name(name=name)
    except errors.UserNotFoundError:
        return errors.not_found_response("No user associated with that name.")
    except errors.UserConflictError as user_conflict_error:
        return errors.conflict_response(message=str(user_conflict_error))

    response_data = {
        "status": "ok",
        "user": deleted_user_info
    }
    return jsonify(response_data)


def get_token_from_request():
    token_detected = (request.args.get("token") or
                      request.headers.get("Authorization") or
                      request.headers.get("Token"))
    if token_detected is None:
        raise errors.UserAuthenticationError(
            "No token detected in your request.")
    elif not isinstance(token_detected, str) and not isinstance(token_detected, unicode):
        raise errors.UserAuthenticationError("Your token must be a string.")
    return token_detected.encode('utf-8')


def parse_info_as_image(raw_data):
    if raw_data == b'' or raw_data is None:
        raise errors.ImageNotFoundError(
            "Please include an image in your request body.")
    return raw_data


def parse_info_as_json(raw_data):
    try:
        raw_info_dict = json.loads(raw_data)
    #except json.JSONDecodeError:
    except ValueError:
        raise errors.JSONNotFoundError(message="Raw data not readable.")

    if not isinstance(raw_info_dict, dict):
        raise errors.JSONNotFoundError(
            message="Decoded JSON is not a Python dictionary.")

    clean_info_dict = {
        "name": raw_info_dict.get("name").encode('utf-8'),
        "token": raw_info_dict.get("token"),
        "total_quota": raw_info_dict.get("total_quota"),
        "quota_left": raw_info_dict.get("quota_left")
    }
    if clean_info_dict.get("token") is not None:
        clean_info_dict["token"] = raw_info_dict.get("token").encode('utf-8')
    return clean_info_dict


@auth.verify_password
def is_admin(username, password):
    try:
        all_admins = admins.get_password_hashes()
        password_hash = all_admins[username]
    except:
        return False
    return pwd_context.verify(password, password_hash)


@auth.error_handler
def send_error_not_admin():
    return errors.unauthorized_response(message="Admin access required.")


@app.errorhandler(400)
def send_error_bad_request(error):
    return errors.bad_request_response()


@app.errorhandler(401)
def send_error_unauthorized(error):
    return errors.unauthorized_response()


@app.errorhandler(404)
def send_error_not_found(error):
    return errors.not_found_response()


@app.errorhandler(405)
def send_error_method_not_allowed(error):
    return errors.method_not_allowed_response()


@app.errorhandler(500)
def send_error_internal_server_error(error):
    return errors.internal_server_error_response()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model prediction')

    parser.add_argument('--params','-p', type=str, default='',
                        help='path to the file which stores network parameters.')
    args = parser.parse_args()

    classifier = BoneAgePredictor(args)

    users.initialize()
    app.run(host="0.0.0.0", port=5000)
