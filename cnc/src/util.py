import os
import yaml
from pathlib import Path
import pandas as pd

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from openai import OpenAI
import tiktoken
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import joblib

import os
import difflib


def compare_files_unified(file1, file2, encoding="utf-8"):
    """Compare two files and return their differences in unified diff format."""
    with open(file1, "r", encoding=encoding, errors="replace") as f1, open(
        file2, "r", encoding=encoding, errors="replace"
    ) as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()

    diff = list(
        difflib.unified_diff(file1_lines, file2_lines, fromfile=file1, tofile=file2)
    )

    if diff:
        return "".join(diff)
    return None


def find_files_in_subfolders(main_folder):
    """Recursively find all files in subfolders, returning a dictionary where filenames are keys."""
    files_dict = {}

    for root, _, files in os.walk(main_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file not in files_dict:
                files_dict[file] = [file_path]
            else:
                files_dict[file].append(file_path)

    return files_dict


def create_dump_output(file1, file2, diff, output_folder, encoding):
    """Create an output folder using the last 3rd directory name from file1 and concatenate it with file2."""
    # Split the paths into components
    file1_parts = os.path.normpath(file1).split(os.sep)
    file2_parts = os.path.normpath(file2).split(os.sep)
    # Get the last 3rd directory from file1
    folder_name_file = file1_parts[-3] + "_vs_" + file2_parts[-3]
    file1_name = file1_parts[-1] + "_" + file1_parts[-3]
    file2_name = file2_parts[-1] + "_" + file2_parts[-3]
    target_filename = file1_parts[-1]

    # Create the full folder path
    output_folder = os.path.join(output_folder, folder_name_file)

    # Create the folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    with open(
        os.path.join(output_folder, f"{file1_name}.txt"), "w", encoding="utf-8"
    ) as f1_out:
        with open(file1, "r", encoding=encoding) as f1:
            f1_out.write(f1.read())

    with open(
        os.path.join(output_folder, f"{file2_name}.txt"), "w", encoding="utf-8"
    ) as f2_out:
        with open(file2, "r", encoding=encoding) as f2:
            f2_out.write(f2.read())

    with open(
        os.path.join(output_folder, f"{target_filename}_diff.txt"),
        "w",
        encoding="utf-8",
    ) as diff_out:
        diff_out.write(diff)


def get_folder_names(path):
    # List only directories in the given path
    return [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]


def get_file_names(path):
    # List only files in the given path
    return [file.name for file in Path(path).iterdir() if file.is_file()]


def read_file_as_string(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as file:
        content = file.read()
    return content


def calculate_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def llm_azure_langchain(model="gpt-4o"):
    if model == "gpt-4o":
        llm = AzureChatOpenAI(
            deployment_name=os.environ["AZURE_DEPLOYMENT"],
            model_name=os.environ["AZURE_MODEL_NAME"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0,
            model_kwargs={"seed": 1213, "top_p": 0},
        )
    elif model == "gpt-3.5":
        llm = AzureChatOpenAI(
            deployment_name=os.environ["AZURE_DEPLOYMENT_35"],
            model_name=os.environ["AZURE_MODEL_NAME_35"],
            azure_endpoint=os.environ["AZURE_ENDPOINT_35"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY_35"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_35"],
            temperature=0,
            model_kwargs={"seed": 1213, "top_p": 0},
        )
    else:
        return None
    return llm


def get_chat_answers(
    prompt, input_token_price=2.50, output_token_price=10.00, unit_token_count=1_000_000
):
    llm = llm_azure_langchain()

    # Creating the human message
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },
        ],
    )

    # Calculate input tokens
    input_tokens = calculate_tokens(prompt)
    print(f"Input tokens: {input_tokens}")

    # Get response from the model
    response = llm.invoke([message])
    # print(f"Response content: {response.content}")

    # Calculate output tokens
    output_tokens = calculate_tokens(response.content)
    print(f"Output tokens: {output_tokens}")

    # Calculate total tokens
    total_tokens = input_tokens + output_tokens
    # print(f"Total tokens: {total_tokens}")

    # Token cost calculation
    input_cost = (input_tokens / unit_token_count) * input_token_price
    output_cost = (output_tokens / unit_token_count) * output_token_price
    total_cost = input_cost + output_cost

    # Print the costs
    # print(f"Input cost: ${input_cost:.6f}")
    # print(f"Output cost: ${output_cost:.6f}")
    # print(f"Total cost: ${total_cost:.6f}")

    return response, total_cost
