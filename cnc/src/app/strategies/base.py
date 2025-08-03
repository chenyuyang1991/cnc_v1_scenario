from abc import ABC, abstractmethod
from typing import Generator, Union
from model.agent.langchain.src.app.models.responses import (
    ChatResponse,
    DataFrameResponse,
    UserDataFrameResponse,
    EndResponse,
)
from model.agent.langchain.src.app.services.session_service import SessionService
import pandas as pd


class SessionDataStrategy(ABC):

    def get_data(self, session_id: str):
        current_state = SessionService.get_session_state(session_id)
        output = self.initialize_data(current_state)
        for each in output:
            if isinstance(each, str):
                yield ChatResponse(event="response_string", data=each)
            elif isinstance(each, pd.DataFrame):
                if "有效黏度" in each.columns:
                    yield UserDataFrameResponse(
                        event="response_user_input_dataframe",
                        data=each.to_dict(orient="records"),
                    )
                else:
                    yield DataFrameResponse(
                        event="response_dataframe", data=each.to_dict(orient="records")
                    )
        yield EndResponse(event="response_end", data="Stream end")

    @abstractmethod
    def initialize_data(self, current_state):
        pass


class InjectionSetpointStrategy(SessionDataStrategy):
    def initialize_data(self, current_state):
        return current_state.injection_setpoint_init()


class MeltingSetpointStrategy(SessionDataStrategy):
    def initialize_data(self, current_state):
        return current_state.melting_setpoint_init()
