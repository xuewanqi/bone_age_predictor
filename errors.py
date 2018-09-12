from flask import jsonify, make_response


class UserAuthenticationError(TypeError):
    def __init__(self, message):
        super(UserAuthenticationError, self).__init__(message)


class ImageNotFoundError(TypeError):
    def __init__(self, message):
        super(ImageNotFoundError, self).__init__(message)


class JSONNotFoundError(TypeError):
    def __init__(self, message):
        super(JSONNotFoundError, self).__init__(message)


class UserNotFoundError(LookupError):
    def __init__(self, message):
        super(UserNotFoundError, self).__init__(message)


class UserConflictError(ValueError):
    def __init__(self, message):
        super(UserConflictError, self).__init__(message)


class UserInfoError(ValueError):
    def __init__(self, message):
        super(UserInfoError, self).__init__(message)


class UserNameError(UserInfoError):
    def __init__(self, message):
        super(UserNameError, self).__init__(message)


class UserTokenError(UserInfoError):
    def __init__(self, message):
        super(UserTokenError, self).__init__(message)


class UserTotalQuotaError(UserInfoError):
    def __init__(self, message):
        super(UserTotalQuotaError, self).__init__(message)


class UserQuotaLeftError(UserInfoError):
    def __init__(self, message):
        super(UserQuotaLeftError, self).__init__(message)


def bad_request_response(message=None):
    return make_error_response(code=400, code_phrase="Bad Request", message=message)


def unauthorized_response(message=None, dict=None):
    body = {"status": "error", "code": "Unauthorized"}
    if message is not None:
        body["message"] = message
    if dict is not None:
        for key in dict:
            body[key] = dict[key]
    return make_response(jsonify(body), 401)


def not_found_response(message=None):
    return make_error_response(code=404, code_phrase="Not Found", message=message)


def method_not_allowed_response(message=None):
    return make_error_response(code=405, code_phrase="Method Not Allowed", message=message)


def conflict_response(message=None):
    return make_error_response(code=409, code_phrase="Conflict", message=message)


def internal_server_error_response(message=None):
    return make_error_response(code=500, code_phrase="Internal Server Error", message=message)


def make_error_response(code, code_phrase, message=None):
    body = {"status": "error", "code": code_phrase}
    if message is not None:
        body["message"] = message
    return make_response(jsonify(body), code)
