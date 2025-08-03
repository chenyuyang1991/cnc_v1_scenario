import re


def find_tools_spec(sub_program: str, content: str):
    """
    在NC代碼中找到子程序調用M98P{sub_program}的位置，並提取相關的刀具信息和規格

    Args:
        sub_program: 子程序號
        content: NC代碼內容

    Returns:
        tuple: (tool, tool_spec) 刀具號和刀具規格
    """
    # 初始化返回值
    tool = ""
    tool_spec = ""

    # 查找子程序調用的位置
    sub_program_call = f"M98P{sub_program}"
    if sub_program_call not in content:
        return tool, tool_spec

    # 分割代碼為行
    lines = content.split("\n")

    # 找到子程序調用的行號
    call_line_index = -1
    for i, line in enumerate(lines):
        if sub_program_call in line:
            call_line_index = i
            break

    if call_line_index == -1:
        return tool, tool_spec

    # 從子程序調用行向上搜索，直到找到上一個子程序調用或文件開頭
    start_search_index = call_line_index
    for i in range(call_line_index - 1, -1, -1):
        if "M98P" in lines[i]:
            start_search_index = i + 1
            break

    # 在搜索範圍內查找刀具信息
    search_range = lines[start_search_index : call_line_index + 1]
    search_text = "\n".join(search_range)

    # 使用正則表達式查找括號內的內容
    pattern = r"\(.*?T(\d+).*?\)"
    matches = re.findall(pattern, search_text)

    if matches:
        # 提取刀具號
        tool_num = matches[0]
        tool = f"T{tool_num}"

        # 提取完整的括號內容
        full_pattern = r"\((.*?T\d+.*?)\)"
        full_matches = re.findall(full_pattern, search_text)

        if full_matches:
            full_content = full_matches[0]
            # 移除T和數字部分
            tool_pattern = r"T(\d+)"
            tool_matches = re.findall(tool_pattern, full_content)
            if tool_matches:
                tool_spec_raw = full_content.replace(f"T{tool_matches[0]}", "")
                # 先用特殊字符（如*、-、空格等）分割
                parts = re.split(r"[\*\-\s]+", tool_spec_raw)
                # 過濾掉全英文且不帶數字的片段（拼音）
                filtered = [
                    p
                    for p in parts
                    if p and (re.search(r"\d", p) or not re.fullmatch(r"[A-Za-z]+", p))
                ]
                # 直接拼接剩餘片段
                tool_spec = "".join(filtered)

    return tool, tool_spec
