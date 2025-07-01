from pathlib import Path

config_path = Path("config.py")

if config_path.exists():
    content = config_path.read_text()
    lines = content.splitlines()
    new_lines = []
    inside_class = False
    inserted = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("class DatabaseConfig"):
            inside_class = True
            new_lines.append(line)
            continue
        if inside_class and stripped.startswith("class "):
            inside_class = False  # Exit if another class starts
        if inside_class and not inserted and stripped == "":
            new_lines.append("    vector_db_type = os.getenv(\"VECTOR_DB_TYPE\", \"milvus\")")
            inserted = True
        new_lines.append(line)

    config_path.write_text("\n".join(new_lines))
    print("✅ vector_db_type inserted in config.py")
else:
    print("❌ config.py not found in the current directory.")
