import random
import string

from passlib.apps import custom_app_context as pwd_context


def generate_token():
    """
    Generates a random 32-character token. Tokens may only include lowercase letters, uppercase letters, and digits.
    They are hence case-sensitive.
    :return: 32-character case-sensitive token as a string.
    """
    candidate_characters = string.ascii_lowercase + string.ascii_uppercase + string.digits
    new_token = ""
    for i in range(0, 32):
        new_token += random.choice(candidate_characters)
    return new_token


def get_password_hashes():
    return {
        "admin0": "$6$rounds=656000$0C7Y4JJhPeKmqYeR$7cmHGafIbTf.nQlcJoc/XdulciayVyZAivAW3tPn2EJW2R0Cp..Njwazau6WixDa7JK3TImJ96rsZhnhFNqUz."
    }


def make_password_hash(password):
    return pwd_context.encrypt(password)


if __name__ == "__main__":
    new_password = input("Enter a password to be hashed: ")
    password_hash = make_password_hash(new_password)
    print()
    print("Add this hash...")
    print()
    print("\t" + password_hash)
    print()
    print("... to the dictionary in the get_password_hashes() function, in the admins.py module.")
