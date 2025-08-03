import streamlit as st


def apply_custom_style():
    st.markdown(
        """
        <style>
        /* Overall theme */
        .stApp {
            background-color: #1B2838;
            color: #FFFFFF;
        }

        /* Title styling */
        h1 {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            margin-left: 2rem !important;
            margin-right: 2rem !important;
        }

        /* Make the page wider */
        .block-container {
            max-width: 95% !important;
            padding-top: 1rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Adjust title size */
        h1 {
            font-size: 1.8rem !important;
            padding-bottom: 2rem !important;
        }

        /* Navigation button styling */
        .stButton button {
            height: auto !important;
            padding: 15px 15px !important;
            background-color: #2A3F5F;
            border: none;
            font-size: 0.95rem !important;
            line-height: 1.4;
            margin: 0.5rem 0;
            white-space: normal !important;
            text-align: left !important;
        }

        /* Active navigation button */
        .stButton button[kind="primary"] {
            background-color: #4A90E2 !important;
            border-color: #4A90E2 !important;
        }

        /* Hover effect */
        .stButton button:hover {
            background-color: #3A4F6F !important;
            border-color: #3A4F6F !important;
        }

        .stButton button[kind="primary"]:hover {
            background-color: #357ABD !important;
            border-color: #357ABD !important;
        }

        /* Vertical line container */
        .vertical-nav {
            padding: 1rem 0.5rem;
            position: relative;
            margin-right: 1rem;
        }

        /* Vertical line */
        .vertical-line {
            position: absolute;
            left: calc(50% - 1px);
            top: 0;
            bottom: 0;
            width: 2px;
            background: #2A3F5F;
            z-index: 1;
        }

        /* Content area spacing */
        .stTab {
            padding-top: 1rem;
        }

        /* Red underline for active button */
        .stButton button[kind="primary"]::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 100%;
            height: 2px;
            background-color: #FF4B4B;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
