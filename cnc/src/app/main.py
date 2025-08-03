import datetime
import os
from io import BytesIO
import json
import base64
import uuid
import logging
from typing import Optional
from urllib.parse import unquote
import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import ResponseValidationError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobPrefix

from model.agent.langchain.src.app.models.responses import (
    CreateCaseResponse,
    UpdateResponse,
    ChatSessionResponse,
    DataValidationResponse,
)
from model.agent.langchain.src.app.models.inputs import (
    ChatModel,
    CaseModel,
    UpdateModel,
    ReturnModel,
    ImageUploadModel,
    ConfigureModel,
    HistoricalSessionInputModel,
)
from model.agent.langchain.src.app.services.session_service import SessionService
from model.agent.langchain.src.app.services.logging_middleware import (
    LogRequestMiddleware,
)
from model.agent.langchain.src.chat_session import chat_session
from model.agent.langchain.src.utils.io import (
    load_excel_from_adls,
    write_excel_to_adls,
    write_image_to_adls,
)
# from model.agent.langchain.src.utils.utils import TIMEZONE
import pytz
TIMEZONE = pytz.timezone("Asia/Shanghai")

from model.data_upload.pipeline import (
    load_machine_info,
    load_configuration,
)
from model.data_upload.data_validate import main as data_validate
from model.mass_production.src.main_detect_issue import run_detect_issue
from model.mass_production.src.utils import (
    generate_issue_reports,
    solve_issue,
    solve_user_issue,
)
from model.historical_case_analysis.src.generate_historical_sessions import (
    get_historical_sessions,
    get_historical_details,
)
from model.agent.langchain.src.utils.exceptions import (
    LlmResponseError,
    AgentProcessError,
    ImageRelatedError,
    DataContentError,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("setpoint-log")

app = FastAPI()
version_time = datetime.datetime.now(TIMEZONE)
credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(
    account_url=f"https://{os.environ['AZURE_STORAGE_ACCOUNT']}.blob.core.windows.net",
    credential=credential,
)
container_client = blob_service_client.get_container_client(
    os.environ["AZURE_CONTAINER_NAME"]
)

origins = [
    "http://localhost:5173",
    "https://imsetpoint.foxconn.com",
    "https://fxn-genesis-app-fe-ecaqdnbqa6b6auce.japaneast-01.azurewebsites.net",
    "https://fxn-genesis-app-fe-backup-cqa7hgbcawf6eqb4.japaneast-01.azurewebsites.net",
    "http://10.30.102.237:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LogRequestMiddleware)

uploaded_files = {}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    body = await request.body()

    logger.error(f"An error occurred: {exc}")
    logger.error(f"Request details: {request.method} {request.url}")
    logger.error(f"Request body: {body.decode('utf-8') if body else 'No Body'}")

    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "details": str(exc)},
    )


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(
    request: Request, exc: ResponseValidationError
):
    logging.error(f"Response validation error in request to {request.url}: {exc}")
    return JSONResponse(
        status_code=501,
        content={
            "detail": exc.errors(),
            "message": f"Validation failed for response: {exc}",
        },
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_error_exception_handler(
    request: Request, exc: FileNotFoundError
):
    logging.error(f"ValueError in request to {request.url}: {exc}")
    return JSONResponse(
        status_code=550,
        content={
            "detail": str(exc),
            "message": "FileNotFoundError",
        },
    )


@app.exception_handler(LlmResponseError)
async def llm_response_error_exception_handler(request: Request, exc: LlmResponseError):
    logging.error(f"LlmResponseError in request to {request.url}: {exc}")
    return JSONResponse(
        status_code=551,
        content={
            "detail": str(exc),
            "message": "LlmResponseError",
        },
    )


@app.exception_handler(AgentProcessError)
async def agent_process_error_exception_handler(
    request: Request, exc: AgentProcessError
):
    logging.error(f"AgentProcessError in request to {request.url}: {exc}")
    return JSONResponse(
        status_code=552,
        content={
            "detail": str(exc),
            "message": "AgentProcessError",
        },
    )


@app.exception_handler(ImageRelatedError)
async def image_related_error_exception_handler(
    request: Request, exc: ImageRelatedError
):
    logging.error(f"ImageRelatedError in request to {request.url}: {exc}")
    return JSONResponse(
        status_code=553,
        content={
            "detail": str(exc),
            "message": "ImageRelatedError",
        },
    )


@app.exception_handler(DataContentError)
async def data_content_error_exception_handler(request: Request, exc: DataContentError):
    logging.error(f"DataContentError in request to {request.url}: {exc}")
    return JSONResponse(
        status_code=554,
        content={
            "detail": str(exc),
            "message": "DataContentError",
        },
    )


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    logging.error(f"ValueError in request to {request.url}: {exc}")
    return JSONResponse(
        status_code=555,
        content={
            "detail": str(exc),
            "message": f"ValueError",
        },
    )


@app.get(
    "/",
    summary="Root Endpoint",
    description="""
         Returns a welcome message.
         """,
)
async def root():
    """
    Root endpoint of the API.

    Returns a welcome message.
    """
    blob_name = "sample.json"
    blob_client = container_client.get_blob_client(blob_name)

    blob_data = blob_client.download_blob().readall()
    json_data = json.loads(blob_data)

    return {
        "message": f"Hello World dev! {version_time} {json.dumps(json_data, indent=2)}"
    }


@app.post(
    "/setpoint_update",
    summary="Update Setpoint",
    response_model=UpdateResponse,
    description="""
    Update the setpoint for a given session.

    {
        "session_id": "session_1",
        "setpoint": {
            "母模模溫(°C)": 61,
            "公模模溫(°C)": 60,
            "料筒溫度-第1段(°C)": 230,
            "料筒溫度-第2段(°C)": 230,
            "料筒溫度-第3段(°C)": 230,
            "料筒溫度-第4段(°C)": 220,
            "料筒溫度-第5段(°C)": 210,
            "後鬆退距離(mm)": 5,
            "計量行程(mm)": 49,
            "保壓切換位置(mm)": 16,
            "背壓(MPa)": 12,
            "冷卻時間(s)": 15,
            "射出壓力(MPa)": 189,
            "射出終點位置-第1段(mm)": 47,
            "射出終點位置-第2段(mm)": 43,
            "射出終點位置-第3段(mm)": 22,
            "射出終點位置-第4段(mm)": 15,
            "射出終點位置-第5段(mm)": 16,
            "射出速度-第1段": 90
        }
    }
    """,
)
async def update_setpoint(update_model: UpdateModel) -> UpdateResponse:
    """
    Update the setpoint for a given session.

    Args:
        update_model (UpdateModel): The session ID and the new setpoint details.

    Returns:
        Dict[str, str]: A message indicating that the setpoint was updated.
    """
    current_state = SessionService.get_session_state(update_model.session_id)
    current_state.set_setpoints(update_model.setpoint)

    return UpdateResponse()


@app.post(
    "/session",
    response_model=CreateCaseResponse,
    summary="Create Session",
    description="""
    Creates a new session with the given case name.

    {
        "case_name": "專案A"
    }
    """,
)
async def create_session(case_model: CaseModel) -> CreateCaseResponse:
    """
    Create a new session with the given case name.

    Args:
        case_model (CaseModel): The case name to create the session for.

    Returns:
        CreateCaseResponse: The ID of the created session.
    """
    logger.info("calling create_session")
    return CreateCaseResponse(
        session_id=SessionService.set_session_state(case_name=case_model.case_name)
    )


@app.post(
    "/get_chat_session",
    response_model=ChatSessionResponse,
    summary="Chat Session",
    description="""
    Diagnose the injection setpoint for a given session.

    {
        "user_input": "{dumped json}",
        "session_id": "session_1"
    }
    """,
)
async def get_chat_session(
    chat_model: ChatModel,
) -> ChatSessionResponse:
    """
    Diagnose the injection setpoint for a given session.

    Args:
        chat_model (ChatModel): The session ID and the query for diagnosis.

    Returns:
        ChatSessionResponse
    """
    current_state = SessionService.get_session_state(chat_model.session_id)
    response = chat_session(current_state, chat_model.query)
    current_serialized = SessionService.append_session_state(chat_model.session_id)
    session_bytes = current_serialized.encode("utf-8")
    # name = f"logs/{chat_model.session_id}_{current_state.case_name}_{datetime.datetime.now(TIMEZONE)}.json"
    name = f"logs/{current_state.create_date}/{current_state.case_name}_{chat_model.session_id}/{datetime.datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}.json"

    # upload
    blob_client = container_client.get_blob_client(name)
    blob_client.upload_blob(session_bytes, overwrite=True)

    return response


@app.post(
    "/return",
    response_model=ChatSessionResponse,
    summary="Return Previous Step",
    description="""
    Return previous step

    {
        "session_id": "session_1"
    }
    """,
)
async def return_previous_step(return_model: ReturnModel) -> ChatSessionResponse:
    """
    Return previous step

    Args:
        session_id (str): The session ID to return to the previous step.

    Returns:
        ChatSessionResponse
    """

    logger.info("calling return_previous_step")
    session_id = return_model.session_id
    current_new_state = SessionService.return_previous_step(session_id)

    return current_new_state.chat_session_output


@app.get(
    "/products/{category}",
    summary="Return List of Products",
    description="""
    Return list of products
    """,
)
async def return_products(category):  # npi / mp
    """
    Return list of products

    Args:
        session_id (str): The session ID to return to the previous step.

    Returns:
        List of products
    """
    logger.info("calling return_products")

    list_of_products = []
    product_df = load_excel_from_adls(f"{category}/templates/product.xlsx")
    for idx, row in product_df.iterrows():
        list_of_products.append(
            {
                "產品資訊": {
                    "模具ID": row["模具ID"],
                    "產品名稱": row["產品名稱"],
                }
            }
        )

    return list_of_products


@app.post(
    "/configuration/npi/",
    summary="Return Configuration of New Product",
    description="""
    Return configuration of selected products
    """,
)
async def return_configuration(model: ConfigureModel):
    """
    Return configuration of selected products

    Args:
        model: Model with mold ID and machine ID to generate case configuration

    Returns:
        Configuration of selected products and machine
    """
    logger.info("calling return_configuration")

    configuration = load_configuration(
        "npi",
        mold_id=model.mold_id,
        machine_id=model.machine_id,
        adls=True,
    )
    return configuration


@app.get(
    "/machines/{category}",
    summary="Return List of Machines",
    description="""
    Return list of products
    """,
)
async def return_machines(category, mold_id):
    """
    Return list of products

    Args:
        category: npi or mp
        mold_id: mold ID, to be further developed

    Returns:
        List of machines
    """
    machine_df = load_excel_from_adls(f"{category}/templates/machine.xlsx")
    machines = (
        machine_df.drop_duplicates(["廠房", "機台ID"])
        .groupby("廠房")
        .機台ID.unique()
        .to_dict()
    )
    machines = {x: list(machines[x]) for x in machines}
    return machines


@app.get(
    "/machine_areas",
    summary="Return List of Machine Areas",
    description="""
    Return list of machine areas
    """,
)
async def return_machine_areas():
    """
    Return list of products

    Args:
        category: npi or mp
        mold_id: mold ID, to be further developed

    Returns:
        List of machines
    """
    # machine_df = load_excel_from_adls(f"mp/templates/machine.xlsx")
    machine_df = pd.read_excel(
        f"../../data_upload/upload_data/mp/templates/machine.xlsx", skiprows=1
    )
    areas = ["ALL"] + list(machine_df.drop_duplicates(["廠房", "機台ID"]).廠房.unique())
    return areas


@app.get(
    "/machine_info/{category}/{machine_id}",
    summary="Return Dict of Machine info",
    description="""
    Return dict of machine info
    """,
)
async def return_machine_info(category, machine_id):
    """
    Return list of products

    Args:
        machine_id: machine ID

    Returns:
        Info of selected machine
    """
    machine_df = load_excel_from_adls(f"{category}/templates/machine.xlsx", skiprows=1)
    machine_info = load_machine_info(machine_id, machine_df)
    return machine_info


# TODO
@app.get(
    "/image/{mold_id}",
    summary="Return mold image",
    description="""
    Return base64 image
    """,
)
async def return_image(mold_id):
    logger.info("calling return_image")
    logger.info(f"mold_id = {mold_id}")

    blob_name = f"npi/images/{mold_id}.png"
    try:
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        encoded_string = base64.b64encode(blob_data).decode("utf-8")

        return {"base64_image": encoded_string}

    except Exception as e:
        return {"base64_image": None}


@app.get(
    "/session_list",
    summary="Return session list",
    description="""
    Return list of sessions
    """,
)
async def return_session_list():
    logger.info("calling return_session_list")

    blobs = container_client.list_blobs(name_starts_with="logs/")
    list_of_blobs = list(
        set([os.path.dirname(x.name) for x in blobs if x.name.endswith(".json")])
    )
    list_of_sessions = []
    for each in list_of_blobs:
        try:
            create_date = datetime.datetime.strptime(each.split("/")[1], "%Y%m%d")
            today = datetime.datetime.today()
            if (today - create_date).days < 3:
                list_of_sessions.append(
                    {
                        "case_name": each.replace("logs/", "").split("_")[0],
                        "session_id": each.split("_")[1],
                    }
                )
        except:
            pass
    return list_of_sessions
    # return SessionService.return_session_list()


@app.post(
    "/session_load",
    summary="Load session from storage",
    description="""
    Load session from storage
    """,
)
async def return_session_load(return_model: ReturnModel):

    # 找到Session_id的最後一輪
    session_id = return_model.session_id
    blobs = container_client.list_blobs(name_starts_with=f"logs/")
    snapshots = [
        x.name for x in blobs if x.name.startswith("logs/") and session_id in x.name
    ]
    snapshots_path = max(snapshots)
    name = snapshots_path.split("/")[2].split("_")[0]

    SessionService.set_session_state(
        case_name=name, session_id=session_id, load_from=snapshots_path
    )

    return json.loads(SessionService.get_session_state(session_id).serialize())


@app.post(
    "/session_restore",
    summary="Return previous session",
    description="""
    Return to previous session
    """,
)
async def return_previous_session(return_model: ReturnModel):
    logger.info("calling return_previous_session")
    previous_state = SessionService.get_session_state(return_model.session_id)
    return json.loads(previous_state.serialize())


@app.get(
    "/upload_info/{category}",
    summary="Return relevant information for upload",
    description="""
    Return relevant information for upload
    """,
)
async def return_upload_info(category):
    logger.info("calling return_upload_info")
    logger.info(f"category = {category}")

    return [
        {
            "tab_name": "機台",
            "api_name": f"{category}/machine",
            "dropdown_name": "機台ID",
            "file_info": [
                "機台型號",
                "機台廠牌",
                "機台ID",
                "機台類型",
                "螺桿直徑(mm)",
                "最大射出計量(mm)",
                "HMI面板壓力單位",
                "最大射出壓力",
                "HMI面板速度單位",
                "最大射出速度單位",
                "最大射出速度",
                "螺桿增強比",
                "HMI射出段數",
                "車間",
                "機台號",
            ],
        },
        {
            "tab_name": "材料",
            "api_name": f"{category}/material",
            "dropdown_name": "材料型号",
            "file_info": [
                "材料類型",
                "材料廠牌",
                "材料型號",
                "建議最低模溫",
                "建議最高模溫",
                "建議最低料溫",
                "建議最高料溫",
                "建議最低背壓(mpa)",
                "建議最高背壓(mpa)",
                "熔融態密度",
                "固態密度",
                "熱擴散係數",
                "熱變形溫度",
            ],
        },
        {
            "tab_name": "產品",
            "api_name": f"{category}/product",
            "dropdown_name": "模具ID",
            "file_info": [
                "模具號",
                "模具ID",
                "產品名稱",
                "外觀要求",
                "顔色",
                "模穴數",
                "流道體積(mm3)",
                "產品體積(mm3)",
                "流道重量(g)",
                "產品重量(g)",
                "產品平均厚度(mm)",
                "產品最大厚度(mm)",
                "所用機台型號",
                "所用機台ID",
                "材料類型",
                "材料廠牌",
                "材料型號",
                "模溫配置_滑塊",
                "模溫配置_公模",
                "模溫配置_母模",
                "是否時序進膠",
                "是否用模流分析參數",
                "主澆口號",
                "模流-填充時間",
                "模流-澆口封閉時間",
                "模流-料溫",
                "模流-模溫",
                "澆口_1_距主澆口距離",
            ],
        },
        {
            "tab_name": "參數",
            "api_name": f"{category}/mp_setpoint",
            "dropdown_name": "模具號",
            "file_info": [
                "機台ID",
                "模具號",
                "模具ID",
                "參數記錄日期",
                "噴嘴N1",
                "截流閥",
                "第1段料溫T1",
                "第2段料溫T2",
                "第3段料溫T3",
                "第4段料溫T4",
                "滑塊模溫",
                "公模溫",
                "母模溫",
                "螺桿轉速",
                "背壓",
                "計量",
            ],
        },
    ]


@app.get(
    "/template/{category}/{template_name}",
    summary="Return template",
    description="""
    Return template
    """,
)
async def return_template(template_name, category):
    blob_name = f"{category}/templates/{template_name}.xlsx"
    blob_client = container_client.get_blob_client(blob_name)

    stream = BytesIO()
    download_stream = blob_client.download_blob()
    download_stream.readinto(stream)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={blob_name}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@app.post("/check")
async def check_xlsx(name: Optional[str] = Form(None), file: UploadFile = File(...)):
    """
    return DataValidationResponse
    """

    logger.info("calling upload_xlsx")
    logger.info(f"name = {name}")

    file_id = str(uuid.uuid4())
    _, file_extension = os.path.splitext(file.filename)

    if file_extension != ".xlsx":
        return JSONResponse(
            status_code=400,
            content={"message": f"File {name} must be an Excel file (.xlsx)"},
        )

    category, name = name.split("/")
    contents = await file.read()
    df = pd.read_excel(BytesIO(contents), engine="openpyxl")
    blob_path = f"{category}/templates/{name}.xlsx"
    archiving_path = f"archived_data/{category}/templates/{name}_{datetime.datetime.now(TIMEZONE).strftime('%Y%m%d %H:%M:%S')}.xlsx"
    pre_upload_path = f"archived_data/pre_upload_data/{name}_{datetime.datetime.now(TIMEZONE).strftime('%Y%m%d %H:%M:%S')}.xlsx"
    uploaded_files[file_id] = {
        "blob_path": blob_path,
        "contents": contents,
        "archiving_path": archiving_path,
        "pre_upload_path": pre_upload_path,
    }

    write_excel_to_adls(df, uploaded_files[file_id]["pre_upload_path"])
    df = load_excel_from_adls(uploaded_files[file_id]["pre_upload_path"], skiprows=1)

    # TODO：data validation here
    issues = data_validate(df, category, name)

    is_valid = True
    if len(issues["errors"]):
        is_valid = False
        errors_text = "### 上傳文件數據驗證未通過，請檢查以下錯誤，並在修改後重試\n"
        for idx, each in enumerate(issues["errors"]):
            errors_text += f"{idx+1}. {each}\n"
    else:
        errors_text = "### 上傳文件數據驗證通過，請點擊「確認更新」按鈕上傳\n"
    if len(issues["warnings"]):
        warnings_text = (
            "### 上傳文件有以下情況不符合規範，您可以更新數據，但系統仍建議您檢查：\n"
        )
        for idx, each in enumerate(issues["warnings"]):
            warnings_text += f"{idx+1}. {each}\n"
    else:
        warnings_text = None

    return DataValidationResponse(
        is_valid=is_valid,
        file_id=file_id,
        errors=issues["errors"],
        warnings=issues["warnings"],
        errors_text=errors_text,
        warnings_text=warnings_text,
    )


@app.post("/upload")
async def upload_xlsx(file_id: str):
    # upload
    blob_client = container_client.get_blob_client(uploaded_files[file_id]["blob_path"])
    blob_client.upload_blob(uploaded_files[file_id]["contents"], overwrite=True)

    # upload backup
    blob_client = container_client.get_blob_client(
        uploaded_files[file_id]["archiving_path"]
    )
    blob_client.upload_blob(uploaded_files[file_id]["contents"], overwrite=True)

    # no longer trigger etl
    # category, name, _ = uploaded_files[file_id]["blob_path"].split("/")
    # remove existing files
    # blobs = container_client.list_blobs(name_starts_with=f"{category}/products/")
    # for blob in blobs:
    #     logger.info(f"Deleting blob: {blob.name}")
    #     container_client.delete_blob(blob.name)
    #
    # result = load_configurations_batch(category)

    return JSONResponse(
        status_code=200,
        content={"message": f"Success!"},
    )


@app.get(
    "/issue_reports",
    summary="Return json of reports on detected issues",
    description="""
    Return detected issues report
    """,
)
async def return_issue_reports(area="ALL"):
    """
    Return json of reports on detected issues

    Returns:
        Json of reports on detected issues
    """

    # 判斷最近ETL刷新時間
    if area == "ALL":
        blob_list = container_client.list_blobs()
        file_list = []
        for blob in blob_list:
            if blob.name.startswith(
                f"mp_realtime_data/etl_output"
            ) and blob.name.endswith(".xlsx"):
                if blob.name not in [
                    "mp_realtime_data/etl_output/anomaly.xlsx",
                    "mp_realtime_data/etl_output/machine_up_to_date.xlsx",
                ]:
                    file_list.append(blob.name)
        if len(file_list):
            last_time = [x.split(".")[0].split("_")[-1] for x in file_list]
            logger.info(f"上次刷新數據時間，{last_time}")
            last_time = datetime.datetime.strptime(max(last_time), "%Y-%m-%d %H:%M:%S")
            last_time = last_time.replace(tzinfo=TIMEZONE)
            if (datetime.datetime.now(TIMEZONE) - last_time).total_seconds() > 7200:
                logger.info("距離上次刷新數據已經經過2小時，run ETL")
                _ = run_detect_issue("./model/mass_production/data/", adls=True)
        else:
            _ = run_detect_issue("./model/mass_production/data/", adls=True)

    issues = generate_issue_reports(area=area)

    return issues


@app.get(
    "/issue_solver/{machine_id}",
    summary="Return suggestions on issue solving",
    description="""
    Return suggestion json
    """,
)
async def return_issue_solver(machine_id):
    """
    Return suggestions on issue solving

    Args:
        session_id (str): The session ID to return to the previous step.

    Returns:
        Json of reports on detected issues
    """
    if "&" not in machine_id:
        response = solve_issue(machine_id)
    else:
        machine_id = unquote(machine_id)
        machine_id, defect_name = machine_id.split("&")
        response = solve_user_issue(machine_id, defect_name)

    return response


@app.post(
    "/historical_case",
    summary="Return list of historical cases",
    description="""
    Return list of historical cases
    """,
)
async def return_historical_cases(model: HistoricalSessionInputModel):
    """
    Return historical cases

    Returns:
        Json of historical cases
    """
    machine_id = model.machine_id
    mold_id = model.mold_id
    session_start_date = model.session_start_date
    session_end_date = model.session_end_date
    historical_cases = get_historical_sessions(
        machine_id=machine_id,
        mold_id=mold_id,
        session_start_date=session_start_date,
        session_end_date=session_end_date,
    )
    return historical_cases


@app.get(
    "/historical_cases/{session_id}",
    summary="Return historical case by session_id",
    description="""
    Return historical case by session_id
    """,
)
async def return_case_detail(session_id):
    """
    Return historical case by session_id

    Args:
        session_id (str): The session ID to return to the previous step.

    Returns:
        Historical case details by session_id
    """
    case_detail = get_historical_details(session_id)
    return case_detail


@app.get(
    "/image_lists",
    summary="Return 2 list of images stored in ADLS, products and products with images",
    description="""
    Return 2 list of images stored in ADLS, products and products with images
    """,
)
async def return_image_lists():
    """
    Return 2 list of images stored in ADLS, products and products with images

    Returns:
        2 list of images stored in ADLS, products and products with images
    """
    products_list = (
        load_excel_from_adls("npi/templates/product.xlsx")["模具號"]
        .astype(str)
        .to_list()
        + load_excel_from_adls("mp/templates/product.xlsx")["模具號"]
        .astype(str)
        .to_list()
    )

    npi_images = container_client.list_blobs(
        name_starts_with=f"npi/images/moldflow_images"
    )
    mp_images = container_client.list_blobs(
        name_starts_with=f"mp/images/moldflow_images"
    )
    images = list(set([x.name for x in npi_images] + [x.name for x in mp_images]))
    images = [".".join(x.split("/")[-1].split(".")[:-1]) for x in images]
    print(images)
    products_with_image = [x for x in images if x in products_list]

    return {"products_list": products_list, "products_with_image": products_with_image}


@app.post(
    "/image_upload",
    summary="Upload and save image in ADLS",
    description="""
    Upload and save image in ADLS
    """,
)
async def image_upload(image_upload_model: ImageUploadModel):
    try:
        image_name = image_upload_model.image_name
        if not image_name.endswith(".png"):
            image_name = image_name + ".png"
        image_data = image_upload_model.image.split(",")[-1].strip()
        write_image_to_adls(image_data, f"mp/images/moldflow_images/{image_name}")
        write_image_to_adls(image_data, f"npi/images/moldflow_images/{image_name}")
        return JSONResponse(
            status_code=200,
            content={"message": f"Success!"},
        )
    except:
        return JSONResponse(
            status_code=380,
            content={"message": f"Image Upload failed!"},
        )


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run(
        "model.agent.langchain.src.app.main:app", host="0.0.0.0", port=8000, reload=True
    )
