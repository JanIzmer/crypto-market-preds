import json
import os

USERS_FILE = "users.json"

def load_users():
    """
    Load the list of Telegram user chat IDs from the JSON file.
    If the file does not exist, returns an empty list.

    Returns:
        list: List of chat IDs (integers or strings).
    """
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_user(chat_id):
    """
    Save a new chat ID to the users list if not already present.
    The updated list is saved back to the JSON file.

    Args:
        chat_id (int or str): Telegram chat ID to save.
    """
    users = load_users()
    if chat_id not in users:
        users.append(chat_id)
        with open(USERS_FILE, "w") as f:
            json.dump(users, f)
