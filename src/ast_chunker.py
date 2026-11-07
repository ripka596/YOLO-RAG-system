from pathlib import Path
import tree_sitter_cpp as tscpp
import tree_sitter_python as tspy
from tree_sitter import Language, Parser

CPP_LANG = Language(tscpp.language())
PY_LANG = Language(tspy.language())

cpp_parser = Parser(CPP_LANG)
py_parser = Parser(PY_LANG)

TARGETS = {
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier"
    },
    "python": {
        "function_definition",
        "class_definition"
    }
}

def extract_node(node, source_bytes: bytes):
    name_node = node.child_by_field_name("name")
    if name_node:
        return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")

    declarator = node.child_by_field_name("declarator")
    target = declarator if declarator else node

    for child in target.children:
        if child.type in {"identifier", "field_identifier", "qualified_identifier"}:
            return source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="ignore")
    return "anonymous"

def parse_code(file_path: Path):
    suffix = file_path.suffix.lower()

    if suffix in {".cpp", ".hpp", ".h", ".c", ".cu", ".cuh"}:
        lang = "cpp"
        parser = cpp_parser
    elif suffix == ".py":
        lang = "python"
        parser = py_parser
    else:
        return []

    with open(file_path, "rb") as f:
        source_bytes = f.read()
    if not source_bytes.strip():
        return []

    tree = parser.parse(source_bytes)
    root_node = tree.root_node
    chunks = []
    target_types = TARGETS[lang]

    def traverse(node, parent_class = None):
        if node.type in target_types:

            chunk_text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

            name = extract_node(node, source_bytes)

            is_class = node.type in {"class_definition", "class_specifier", "struct_specifier"}

            parent = parent_class if (not is_class and parent_class) else None

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "language": lang,
                    "node_type": node.type,
                    "entity_name": name,
                    "parent_class": parent if parent is not None else "",
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                }
            })

            if is_class:
                next_parent = name
            else:
                next_parent = parent_class

            for child in node.children:
                traverse(child, parent_class=next_parent)
            return
        
        for child in node.children:
            traverse(child, parent_class=parent_class)

    traverse(root_node)
    return chunks

if __name__ == "__main__":
    test = Path("data/code_py/models/yolo/model.py")
    if test.exists():
        chunks = parse_code(test)
        print(f"Извлечено {len(chunks)} чанков из {test.name}")
        for c in chunks[:5]:
            print(f"[{c['metadata']['node_type']}] {c['metadata']['entity_name']} (Parent: {c['metadata']['parent_class']})")