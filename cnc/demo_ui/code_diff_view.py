import streamlit as st
import difflib


def show_code_diff(left_code: str, right_code: str):
    """Display NC code differences with highlighted changes in right column only."""

    # Convert string to lines if needed
    left_lines = left_code.split("\n") if isinstance(left_code, str) else left_code
    right_lines = right_code.split("\n") if isinstance(right_code, str) else right_code

    # Find first non-blank line
    first_content_line = 0
    for i, line in enumerate(left_lines):
        if line.strip():
            first_content_line = i
            break

    # Process both sides with character-level diff
    highlighted_left = []
    highlighted_right = []

    for i, (left_line, right_line) in enumerate(zip(left_lines, right_lines)):
        # Create sequence matcher for this line pair
        s = difflib.SequenceMatcher(None, left_line, right_line)

        # Build highlighted versions of both lines
        left_highlighted = []
        right_highlighted = []

        # Add line number with more compact styling, only if line has content
        if i >= first_content_line:
            line_num = i - first_content_line + 1
            line_number = f'<span style="color: #6e7681; user-select: none; display: inline-block; width: 2em; font-size: 0.8em; margin-right: 0.5em; text-align: right;">{line_num}</span>'
        else:
            line_number = f'<span style="color: #6e7681; user-select: none; display: inline-block; width: 2em; font-size: 0.8em; margin-right: 0.5em; text-align: right;"></span>'

        left_highlighted.append(line_number)
        right_highlighted.append(line_number)

        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == "equal":
                # Same text in both lines
                text = left_line[i1:i2]
                left_highlighted.append(f'<span style="color: #6495ED">{text}</span>')
                right_highlighted.append(f'<span style="color: #6495ED">{text}</span>')
            elif tag == "replace":
                # Text is different
                left_highlighted.append(
                    f'<span style="color: #6495ED">{left_line[i1:i2]}</span>'
                )
                right_highlighted.append(
                    f'<span style="background-color: #8b0000; color: white">{right_line[j1:j2]}</span>'
                )
            elif tag == "delete":
                # Text only in left line
                left_highlighted.append(
                    f'<span style="color: #6495ED">{left_line[i1:i2]}</span>'
                )
            elif tag == "insert":
                # Text only in right line
                right_highlighted.append(
                    f'<span style="background-color: #8b0000; color: white">{right_line[j1:j2]}</span>'
                )

        highlighted_left.append("".join(left_highlighted))
        highlighted_right.append("".join(right_highlighted))

    # Join lines with HTML line breaks
    left_code_highlighted = "<br>".join(highlighted_left)
    right_code_highlighted = "<br>".join(highlighted_right)

    # Add CSS for code box styling
    st.markdown(
        """
        <style>
        .code-box {
            background-color: rgb(13, 17, 23);
            border-radius: 0.25rem;
            padding: 0.5em;
            font-family: 'Consolas', 'Monaco', monospace;
            white-space: pre;
            margin: 0;
            border: 1px solid rgba(255,255,255,0.1);
            font-size: 0.9em;
            line-height: 1.2;
            overflow-x: auto;
        }
        .code-box pre {
            margin: 0;
            padding: 0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 原程序")
        st.markdown(
            f'<div class="code-box"><pre>{left_code_highlighted}</pre></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### 優化後程序")
        st.markdown(
            f'<div class="code-box"><pre>{right_code_highlighted}</pre></div>',
            unsafe_allow_html=True,
        )
