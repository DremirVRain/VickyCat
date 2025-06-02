import ast

def extract_class_names(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        # Remove BOM if present
        if content.startswith("\ufeff"):
            content = content[1:]
        tree = ast.parse(content)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


# 示例使用
class_names = extract_class_names("strategy/candle_pattern_strategy.py") + extract_class_names(
    "strategy/indicator_strategy.py") + extract_class_names("strategy/micro_structure.py") + extract_class_names("strategy/structure_strategy.py") + extract_class_names("strategy/volume_strategy.py")
for name in class_names:
    print(name)
