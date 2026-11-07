import os


def get_system_text(key):
    if os.path.exists(os.path.dirname(key)):
        file_path = f"{key}.txt"
    else:
        filename = f"{key}.txt"
        file_path = os.path.join(os.path.dirname(__file__), "system", filename)

    # first look in prompts/system/*.txt
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    else:
        raise FileNotFoundError(f"No file found for key {key}, path={file_path}")


def get_system_text_joined(keys):
    """The files of ``keys`` read and joined as they are, then stripped once:
    the text ``get_system_text`` returns for a single file holding all of them."""
    parts = []
    for key in keys:
        file_path = os.path.join(os.path.dirname(__file__), "system", f"{key}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for key {key}, path={file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            parts.append(file.read())
    return "".join(parts).strip()
