import re
import yaml
import math


def fix(x):
    """
    FIX 函數：對傳入的數值進行截斷，舍去小數部分（不四舍五入）。
    """
    return int(x)


class MacroManager:
    def __init__(self, macros=None, verbose=False):
        """
        初始化 MacroManager，可傳入一個宏變數字典。
        如果未提供，則使用預設的示例宏變數。
        """
        if macros is None or not isinstance(macros, dict):
            macros = {}
        self.macros = macros
        self.unsolved_macros = []
        self.verbose = verbose

    def replace_macros(self, gcode_line: str) -> str:
        """
        宏替換：
          1. 保留所有中括號 []。
          2. 將所有宏變數（格式為 #數字）替換為其數值。
          3. 只對方括號外的簡單數學運算進行求值。

        例如：
          輸入 "G81G99X[#14]Y[#15]Z[0.43+#930]R2.F200"
          得到 "G81G99X[100]Y[200]Z[0.43+50]R2.F200"

          輸入 "G81G99X#14Y[#15]Z0.43+#930R2.F200"
          得到 "G81G99X100Y[200]Z50.43R2.F200"
        """
        # 將方括號內的內容暫時替換為佔位符，以保護其中的運算式
        brackets = []

        def save_bracket(match):
            brackets.append(match.group(1))
            return f"[{{{len(brackets)-1}}}]"

        line = re.sub(r"\[(.*?)\]", save_bracket, gcode_line)

        # 替換方括號外的宏變數
        line = re.sub(r"#(\d+)", lambda m: str(self.get_macro("#" + m.group(1))), line)

        # 對方括號外的簡單數學運算進行求值
        def eval_arith(match: re.Match) -> str:
            expr = match.group("expr")
            try:
                return str(eval(expr))
            except Exception:
                return expr

        line = re.sub(
            r"(?P<expr>(?<![\d.])[+-]?\d+(?:\.\d+)?(?:[+\-*/]\d+(?:\.\d+)?)+)",
            eval_arith,
            line,
        )

        # 處理方括號內的內容：只替換宏變數，不進行運算
        def process_bracket_content(content):
            return re.sub(
                r"#(\d+)", lambda m: str(self.get_macro("#" + m.group(1))), content
            )

        # 還原方括號內容
        for i, content in enumerate(brackets):
            processed_content = process_bracket_content(content)
            line = line.replace(f"[{{{i}}}]", f"[{processed_content}]")

        return line

    def set_macro(self, key: str, value) -> None:
        """
        設置單個宏變數。
        參數 key 必須以 '#' 開頭（如果沒有，會自動添加）。
        """
        if not key.startswith("#"):
            key = "#" + key
        self.macros[key] = value

    def update_macros(self, new_macros: dict) -> None:
        """
        更新宏變數，將傳入字典合并更新當前宏變數。
        如果傳入的鍵不以 '#' 開頭，則自動添加。
        """
        for key, value in new_macros.items():
            if not key.startswith("#"):
                key = "#" + key
            self.macros[key] = value

    def get_macro(self, key: str):
        """
        設置單個宏變數。
        參數 key 必須以 '#' 開頭（如果沒有，會自動添加）。
        """
        if not key.startswith("#"):
            key = "#" + key
        if key not in self.macros.keys():
            if self.verbose:
                print(f"嘗試提取宏變量{key}失敗，採用默認值0")
            self.unsolved_macros.append(key)
            self.set_macro(key, 0)
        return self.macros.get(key, 0)

    def remove_macro(self, key: str) -> None:
        """
        刪除指定的宏變數。
        """
        if not key.startswith("#"):
            key = "#" + key
        if key in self.macros:
            del self.macros[key]

    def clear_macros(self) -> None:
        """
        清空所有宏變數。
        """
        self.macros.clear()

    def parse_macro_assignment(self, gcode_line):
        """
        判斷一行 gcode 是否定義宏變量（格式：#數字=值）。
        如果找到，則更新所有宏變量，並返回 True；否則返回 False。
        """
        pattern = re.compile(r"#(?P<var>\d+)\s*=\s*(?P<value>\S+)")
        found = False
        for match in pattern.finditer(gcode_line):
            var = match.group("var")
            value_str = match.group("value").strip()
            value = self.evaluate_expression(value_str)
            self.set_macro(var, value)
            if self.verbose:
                print(f"宏变量#{var}设置为{value} | {gcode_line}")
            found = True
        return found

    def evaluate_expression(self, expr: str, lookup_macro: bool = False) -> float:
        """
        評估傳入的表達式，支援格式：
          #3=#4454
          #3=#3434+3434
          #3=#[#454*121]
          #3=ABS[#454*121]
          #3=#[ABS[#454*121]+#23]
        """
        # 先替換所有宏括號表示式 #[...]
        expr = self.replace_macro_brackets(expr)
        # 轉換 G-code 風格函數呼叫，例：ABS[#454*121] 轉換為 abs(#454*121)
        expr = self.transform_gcode_expr(expr)
        # 替換所有宏變數標記，如 "#454"；若該宏已定義則使用其值，
        # 否則視為數字（例如 "#454" 轉為 454）
        expr = re.sub(r"#(\d+)", lambda m: str(self.get_macro("#" + m.group(1))), expr)
        # 將剩餘的中括號 [] 替換為小括號 () 以避免被 Python 當作 list 處理
        expr = expr.replace("[", "(").replace("]", ")")
        try:
            result = float(
                eval(
                    expr,
                    {"math": math, "abs": abs, "fix": fix, "FIX": fix},  # 新增大寫別名
                    {},
                )
            )
            if lookup_macro and result.is_integer():
                return self.get_macro("#" + str(int(result)))
            return result
        except Exception as e:
            print(f"解析宏变量算术计算错误，表达式 '{expr}': {e} | {expr}")
            return 0

    def transform_gcode_expr(self, expr: str) -> str:
        """
        將 G-code 風格的函數呼叫轉換為有效的 Python 表達式（支援嵌套）。
        例如：ABS[#454*121] 轉換為 abs(#454*121)
        """
        pattern = re.compile(r"(?P<func>[A-Za-z]+)\[(?P<inner>[^\[\]]+)\]")
        prev_expr = None
        while prev_expr != expr:
            prev_expr = expr
            expr = pattern.sub(
                lambda m: self._func_mapping(m.group("func"))
                + "("
                + self.transform_gcode_expr(m.group("inner"))
                + ")",
                expr,
            )
        return expr

    def _func_mapping(self, func: str) -> str:
        """
        將函數名稱從 G-code 風格轉換為 Python 對應的名稱
        """
        func = func.upper()
        if func == "ABS":
            return "abs"
        elif func == "SIN":
            return "math.sin"
        elif func == "COS":
            return "math.cos"
        elif func == "FIX":
            return "fix"
        elif func == "ATAN":
            return "math.atan"
        else:
            return func.lower()

    def replace_macro_brackets(self, expr: str) -> str:
        """
        將所有以 "#[ ... ]" 表示的宏括號部分替換為其評估結果。
        例如：將 "#[ABS[#454*121]+#23]" 替換為 "(評估結果)"，
        並支援混合運算，例如 "#[ABS[#454*121]+#23]+12"。
        """
        # 使用自定義邏輯來替換所有宏括號表示式 #[ ... ]，支持嵌套情況
        while True:
            start = expr.find("#[")
            if start == -1:
                break
            end = self.find_matching_macro(expr, start)
            if end == -1:
                break
            inner_expr = expr[start + 2 : end]
            value = self.evaluate_expression(inner_expr, lookup_macro=True)
            replacement = "(" + str(value) + ")"
            expr = expr[:start] + replacement + expr[end + 1 :]
        return expr

    def find_matching_macro(self, expr: str, start: int) -> int:
        """
        找到從位置 start（該處必定為 "#["）開始的宏括號對應的匹配 "]" 的索引，
        支持內部可能包含函數呼叫的中括號。
        """
        pos = start + 2
        while pos < len(expr):
            if expr[pos : pos + 2] == "#[":
                inner_end = self.find_matching_macro(expr, pos)
                if inner_end == -1:
                    return -1
                pos = inner_end + 1
            elif expr[pos] == "[":
                # 假設這是函數呼叫的左括號，直接跳過對應的 "]"
                fun_end = expr.find("]", pos)
                if fun_end == -1:
                    pos += 1
                else:
                    pos = fun_end + 1
            elif expr[pos] == "]":
                return pos
            else:
                pos += 1
        return -1

    def save_to_yaml(self, file_path: str) -> None:
        """
        將當前宏變數保存至指定的 YAML 檔案。
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(self.macros, f, allow_unicode=True)
            if self.verbose:
                print(f"宏變數已成功保存到 {file_path}")
        except Exception as e:
            print(f"保存宏變數到 YAML 檔案時發生錯誤: {e}")
