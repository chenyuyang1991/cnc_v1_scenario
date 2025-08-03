from collections import (
    OrderedDict,
    defaultdict,
)
from fastapi import HTTPException
from model.agent.langchain.src.session import Session
from model.agent.langchain.src.utils.io import load_json_from_adls
import uuid
import json


session_cache = OrderedDict()  # load session
session_history = defaultdict(list)  # list of session json


class SessionService:
    @staticmethod
    def get_session_state(session_id: str):
        current_state = session_cache.get(session_id)
        if not current_state:
            raise HTTPException(status_code=404, detail="Session not found")
        return current_state

    @staticmethod
    def set_session_state(case_name: str, session_id=None, load_from=None):
        if session_id is None:
            session_id = str(uuid.uuid4())
        session = Session(
            case_name=case_name,
        )
        if load_from is not None:
            session.load_from_json(
                json.dumps(load_json_from_adls(load_from), indent=4, ensure_ascii=False)
            )
        session_cache[session_id] = session
        session_history[session_id].append(session.serialize())

        # 限定session數量，防止爆內存
        while len(session_cache) > 100:
            session_id_to_delete = list(session_cache)[0]
            del session_cache[session_id_to_delete]
            del session_history[session_id_to_delete]

        return session_id

    @staticmethod
    def return_previous_step(session_id: str):
        session_list = session_history.get(session_id)
        session_list.pop()
        session_cache[session_id].load_from_json(session_list[-1])
        return session_cache[session_id]

    @staticmethod
    def return_previous_session(session_id: str):
        return [
            {"case_name": v.case_name, "session_id": k}
            for k, v in session_cache.items()
        ]

    @staticmethod
    def append_session_state(session_id: str):
        current_state = session_cache.get(session_id)
        current_serialized = current_state.serialize()
        session_history[session_id].append(current_serialized)
        return current_serialized

    @staticmethod
    def return_session_list():
        return [
            {"case_name": v.case_name, "session_id": k}
            for k, v in session_cache.items()
        ]
