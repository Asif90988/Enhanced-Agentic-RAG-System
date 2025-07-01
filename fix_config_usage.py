import os

# Paths to the files you want to fix
db_manager_path = "storage/database_manager.py"
vector_store_path = "storage/vector_store.py"

# Fix database_manager.py
def fix_database_manager():
    with open(db_manager_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Replace CONFIG.database with CONFIG
        if "self.config = CONFIG.database" in line:
            new_lines.append("        self.config = CONFIG\n")
        # Replace self.config.postgres_url → self.config.database.postgres_url
        elif "self.config.postgres_url" in line:
            new_lines.append(line.replace("self.config.postgres_url", "self.config.database.postgres_url"))
        else:
            new_lines.append(line)

    with open(db_manager_path, "w") as f:
        f.writelines(new_lines)
    print(f"✅ Updated: {db_manager_path}")

# Ensure vector_store.py remains unchanged (read-only here, but we verify no extra edits needed)
def confirm_vector_store():
    with open(vector_store_path, "r") as f:
        content = f.read()

    if "self.config.vector_db_type" in content:
        print(f"✅ Confirmed: {vector_store_path} references vector_db_type correctly.")
    else:
        print(f"⚠️ vector_db_type usage not found in {vector_store_path}")

if __name__ == "__main__":
    if not os.path.exists(db_manager_path) or not os.path.exists(vector_store_path):
        print("❌ One or both target files do not exist. Please run this from the root of the project.")
    else:
        fix_database_manager()
        confirm_vector_store()
