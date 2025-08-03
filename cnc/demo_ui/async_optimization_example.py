"""
Example usage of async optimization functions

This file demonstrates how to integrate the async optimization 
into your Streamlit UI components.
"""

import streamlit as st
from processing import (
    process_optimization_async, 
    render_optimization_status,
    is_optimization_running,
    is_optimization_complete
)


def example_optimization_ui():
    """
    Example UI component that uses async optimization
    
    Replace the original process_optimization() call with this pattern
    """
    
    st.header("優化設定")
    
    # Your existing UI components for configuration
    # ... (hyperparameters, ban settings, etc.)
    
    # Optimization buttons and status
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Async optimization button
        if st.button("開始優化 (背景運行)", key="start_async_optimization", type="primary"):
            # Call the async version instead of the original
            new_codes, old_codes, out_df = process_optimization_async()
            
            # If results are immediately available (already completed), store them
            if new_codes is not None:
                st.session_state["new_codes"] = new_codes
                st.session_state["old_codes"] = old_codes
                st.session_state["out_df"] = out_df
                st.success("優化結果已載入！")
    
    with col2:
        # Synchronous optimization button (original behavior)
        if st.button("開始優化 (同步等待)", key="start_sync_optimization"):
            from processing import process_optimization
            
            with st.spinner("正在優化中，請等待..."):
                new_codes, old_codes, out_df = process_optimization()
                st.session_state["new_codes"] = new_codes
                st.session_state["old_codes"] = old_codes
                st.session_state["out_df"] = out_df
                st.success("優化完成！")
    
    # Render status for async optimization
    render_optimization_status()
    
    # Show current status
    if is_optimization_running():
        st.warning("⏳ 優化任務正在後台運行中...")
    elif is_optimization_complete():
        st.info("✅ 優化任務已完成，結果可用")
    
    # Display results if available
    if "new_codes" in st.session_state and st.session_state["new_codes"]:
        st.success("📋 優化結果已準備就緒")
        
        with st.expander("查看優化統計"):
            out_df = st.session_state.get("out_df")
            if out_df is not None:
                st.dataframe(out_df)
        
        # Your existing code to display/download results
        # render_nc_code() or other display functions


def integration_example():
    """
    Example of how to integrate into existing config workflow
    """
    
    # Your existing configuration code...
    # render_config_section()
    
    # Replace this pattern:
    # OLD:
    # if st.button("完成設定"):
    #     new_codes, old_codes, out_df = process_optimization()
    #     st.session_state["new_codes"] = new_codes
    #     ...
    
    # NEW:
    if st.button("完成設定 (背景優化)"):
        new_codes, old_codes, out_df = process_optimization_async()
        if new_codes is not None:
            st.session_state["new_codes"] = new_codes
            st.session_state["old_codes"] = old_codes
            st.session_state["out_df"] = out_df
    
    # Always render status to handle polling
    render_optimization_status()


if __name__ == "__main__":
    st.title("Async Optimization Example")
    example_optimization_ui() 