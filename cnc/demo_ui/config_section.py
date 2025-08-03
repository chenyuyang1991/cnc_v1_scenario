import pandas as pd
import streamlit as st
import yaml
from cnc_genai.demo_ui.config_tabs import (
    render_parameter_settings_tab,
    render_sub_program_settings_tab,
    render_block_disable_tab,
    render_advanced_settings_tab,
)
from cnc_genai.demo_ui import conf_init


@st.cache_data
def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_cached_config():
    try:
        default_config = load_config("cnc_genai/conf/v1_config.yaml")
        return default_config["hyper_params"]
    except:
        return {
            "percentile_threshold": 0.95,
            "multiplier_max": 1.5,
            "multiplier_min": 1.0,
            "multiplier_air": 2.0,
            "apply_finishing": 1,
            "multiplier_finishing": 1.0,
        }


@st.cache_data
def load_cached_sub_programs(selected_department, selected_clamping):
    return conf_init.load_sub_program_init(selected_department, selected_clamping)


def render_config_section():
    """Render the complete configuration section"""
    with st.container():
        st.header("設定優化配置")
        st.markdown(
            f"##### <span style='color:white'>{st.session_state.selected_clamping} - 基於{st.session_state.baseline_display}代碼提升</span>",
            unsafe_allow_html=True,
        )

        # 添加使用切削知識庫推薦倍率按鈕
        if "use_cnc_knowledge_base" not in st.session_state:
            st.session_state.use_cnc_knowledge_base = False
            st.session_state.use_cnc_knowledge_base_changed = False

        use_kb = st.toggle(
            "使用切削知識庫推薦倍率",
            key="kb_recommendation_toggle",
            value=st.session_state.use_cnc_knowledge_base,
        )

        if use_kb != st.session_state.use_cnc_knowledge_base:
            st.session_state.use_cnc_knowledge_base = use_kb
            st.session_state.use_cnc_knowledge_base_changed = True
        else:
            st.session_state.use_cnc_knowledge_base_changed = False

        # Create tabs
        tab_names = ["參數設定", "子程式設定", "關閉特定程序段", "高階默認配置設定"]
        param_tab, prog_tab, ban_n_tab, adv_tab = st.tabs(tab_names)

        # Initialize variables
        hyper_params_config = load_cached_config()

        # Initialize session state for storing tab results if not exists
        tab_results = {
            "hyper_params_dict": None,
            "sub_programs_df": None,
            "ban_n_df": None,
            "advanced_params_dict": None,
        }

        # Only compute the active tab
        with param_tab:
            if param_tab.selected:
                tab_results["hyper_params_dict"] = render_parameter_settings_tab(
                    hyper_params_config
                )
                st.session_state.temp_hyper_params_dict = tab_results[
                    "hyper_params_dict"
                ]

        with prog_tab:
            # todo 關聯刀具
            if prog_tab.selected:
                tab_results["sub_programs_df"] = render_sub_program_settings_tab(
                    df=load_cached_sub_programs(
                        st.session_state.selected_department,
                        st.session_state.selected_clamping,
                    )
                )

        with ban_n_tab:
            # todo 關聯刀具
            if ban_n_tab.selected:
                tab_results["ban_n_df"] = render_block_disable_tab()

        with adv_tab:
            if adv_tab.selected:
                tab_results["advanced_params_dict"] = render_advanced_settings_tab(
                    hyper_params_config
                )
                if (
                    tab_results["advanced_params_dict"]
                    and tab_results["hyper_params_dict"]
                ):
                    tab_results["hyper_params_dict"].update(
                        tab_results["advanced_params_dict"]
                    )

        # Save Configuration Button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回CNC360 V1首頁", use_container_width=True):
                st.session_state.current_page = "landing"
                st.rerun()
        with col2:
            if st.button("儲存設定", use_container_width=True, type="primary"):

                # Force compute any missing tab results
                if tab_results["hyper_params_dict"] is None:
                    tab_results["hyper_params_dict"] = render_parameter_settings_tab(
                        hyper_params_config
                    )
                if tab_results["sub_programs_df"] is None:
                    tab_results["sub_programs_df"] = render_sub_program_settings_tab(
                        df=load_cached_sub_programs(
                            st.session_state.selected_department,
                            st.session_state.selected_clamping,
                        )
                    )
                if tab_results["ban_n_df"] is None:
                    tab_results["ban_n_df"] = render_block_disable_tab()
                if tab_results["advanced_params_dict"] is None:
                    tab_results["advanced_params_dict"] = render_advanced_settings_tab(
                        hyper_params_config
                    )
                    if (
                        tab_results["advanced_params_dict"]
                        and tab_results["hyper_params_dict"]
                    ):
                        tab_results["hyper_params_dict"].update(
                            tab_results["advanced_params_dict"]
                        )

                st.session_state["hyper_params_dict"] = tab_results["hyper_params_dict"]
                st.session_state["sub_programs_df"] = tab_results["sub_programs_df"]
                st.session_state["ban_n_df"] = tab_results["ban_n_df"]
                st.session_state["ban_row_df"] = pd.DataFrame(
                    columns=[
                        "sub_program",
                        "function",
                        "tool",
                        "tool_spec",
                        "N",
                        "ban_row",
                        "src",
                    ]
                )  # initial run we don't allow manual control of ban row, only rerun

                # Set flag to indicate configuration is saved and ready to move to next page
                st.session_state.config_saved = True

            else:
                # Reset the flag if the button is not clicked
                st.session_state["hyper_params_dict"] = {}
                st.session_state["sub_programs_df"] = pd.DataFrame(
                    columns=[
                        "sub_program",
                        "function",
                        "tool",
                        "tool_spec",
                        "finishing",
                        "apply_afc",
                        "apply_air",
                        "apply_turing",
                        "multiplier_max",
                    ]
                )
                st.session_state["ban_n_df"] = pd.DataFrame(
                    columns=[
                        "sub_program",
                        "function",
                        "tool",
                        "tool_spec",
                        "ban_n",
                    ]
                )
                st.session_state["ban_row_df"] = pd.DataFrame(
                    columns=[
                        "sub_program",
                        "function",
                        "tool",
                        "tool_spec",
                        "N",
                        "ban_row",
                        "src",
                    ]
                )
                st.session_state.config_saved = False
