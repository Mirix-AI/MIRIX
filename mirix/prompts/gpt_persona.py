import os

from mirix.constants import MIRIX_DIR

def get_persona_text(key):
    filename = f"{key}.txt"
    file_path = os.path.join(os.path.dirname(__file__), "personas", filename)
    # file_path = os.path.join("./mirix/prompts/personas", filename)

    # first look in prompts/system/*.txt
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    else:
        # try looking in ~/.mirix/personas/*.txt (but don't create the directory)
        user_system_prompts_dir = os.path.join(MIRIX_DIR, "personas")
        file_path = os.path.join(user_system_prompts_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        else:
            raise FileNotFoundError(f"No file found for key {key}, path={file_path}")