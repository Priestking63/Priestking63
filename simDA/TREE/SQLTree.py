import json

def convert_tree_to_json(tree):
    """
    Преобразует обученное дерево решений в компактный формат JSON.
    
    Args:
        tree: Обученная модель дерева решений из sklearn
        
    Returns:
        str: Представление дерева в компактном формате JSON
    """
    def build_node(node_id=0):
        if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:
            class_label = tree.tree_.value[node_id].argmax()
            return {"class": int(class_label)}
        return {
                "feature_index": int(tree.tree_.feature[node_id]),
                "threshold": round(float(tree.tree_.threshold[node_id]), 4),
                "left": build_node(tree.tree_.children_left[node_id]),
                "right": build_node(tree.tree_.children_right[node_id])
            }

    tree_dict = build_node(0)
    return json.dumps(tree_dict, separators=(',', ':'))


def generate_sql_query(tree_as_json: str, features: list) -> str:
    tree = json.loads(tree_as_json)

    def build_case(node):
        if "class" in node:
            return str(node["class"])
        else:
            feat = features[node["feature_index"]]
            thresh = node["threshold"]
            left = build_case(node["left"])
            right = build_case(node["right"])
            return f"CASE WHEN {feat} > {thresh} THEN {right} ELSE {left} END"

    case_str = build_case(tree)
    return f"SELECT {case_str} AS class_label"
