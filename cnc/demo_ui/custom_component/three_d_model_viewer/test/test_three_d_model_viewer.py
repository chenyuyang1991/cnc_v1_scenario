import os
import sys
from pathlib import Path
import streamlit as st

# 添加專案路徑到 Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # 根據您的專案結構調整
sys.path.insert(0, str(project_root))
print(f"project_root: {project_root}")

# 導入 3D 模型檢視器組件
try:
    from three_d_model_viewer import render_from_file, render_from_text
except ImportError as e:
    st.error(f"無法導入 3D 模型檢視器組件: {e}")
    st.stop()

def main():
    st.set_page_config(
        page_title="3D 模型檢視器測試",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 3D 模型檢視器測試工具")
    st.markdown("---")
    
    # 側邊欄控制面板
    with st.sidebar:
        st.header("⚙️ 檢視器設定")
        
        # 檔案上傳區域
        st.subheader("📁 檔案上傳")
        uploaded_file = st.file_uploader(
            "選擇 STL 檔案",
            type=['stl'],
            help="支援二進制和 ASCII 格式的 STL 檔案"
        )
        
        # 基本設定
        st.subheader("🎨 外觀設定")
        color = st.color_picker("物件顏色", value="#696969")
        
        material = st.selectbox(
            "材質類型",
            options=['material', 'flat', 'wireframe'],
            index=0,
            help="material: 具反光效果的材質, flat: 平面材質, wireframe: 線框模式"
        )
        
        opacity = st.slider(
            "透明度",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="1.0 = 完全不透明, 0.1 = 幾乎透明"
        )
        
        shininess = st.slider(
            "光澤度",
            min_value=0,
            max_value=200,
            value=100,
            step=10,
            help="僅在 material 模式下有效果"
        )
        
        # 相機設定
        st.subheader("📷 相機設定")
        auto_rotate = st.checkbox("自動旋轉", value=False)
        
        cam_v_angle = st.slider(
            "垂直角度 (度)",
            min_value=-180,
            max_value=180,
            value=60,
            step=5
        )
        
        cam_h_angle = st.slider(
            "水平角度 (度)",
            min_value=-180,
            max_value=180,
            value=0,
            step=5
        )
        
        cam_distance = st.slider(
            "相機距離",
            min_value=0,
            max_value=1000,
            value=0,
            step=10,
            help="0 = 自動計算最佳距離"
        )
        
        # 檢視器設定
        st.subheader("🖥️ 檢視器設定")
        viewer_height = st.slider(
            "檢視器高度 (像素)",
            min_value=300,
            max_value=800,
            value=390,
            step=50
        )
        
        max_view_distance = st.slider(
            "最大檢視距離",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )
    
    # 主要內容區域
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🎯 3D 檢視器")
        
        if uploaded_file is not None:
            try:
                # 顯示檔案資訊
                file_details = {
                    "檔案名稱": uploaded_file.name,
                    "檔案大小": f"{uploaded_file.size / 1024:.2f} KB",
                    "檔案類型": uploaded_file.type
                }
                
                with st.expander("📋 檔案資訊", expanded=False):
                    for key, value in file_details.items():
                        st.write(f"**{key}**: {value}")
                
                # 讀取檔案內容
                file_content = uploaded_file.read()
                
                # 使用 3D 模型檢視器顯示模型
                success = render_from_text(
                    text=file_content,
                    color=color,
                    material=material,
                    auto_rotate=auto_rotate,
                    opacity=opacity,
                    shininess=shininess,
                    cam_v_angle=cam_v_angle,
                    cam_h_angle=cam_h_angle,
                    cam_distance=cam_distance,
                    height=viewer_height,
                    max_view_distance=max_view_distance,
                    show_performance=True
                )
                
                if not success:
                    st.error("❌ 3D 模型檔案載入失敗")
                    
            except Exception as e:
                st.error(f"❌ 處理檔案時發生錯誤: {str(e)}")
                st.exception(e)
        else:
            # 顯示上傳提示
            st.info("📤 請在左側面板上傳 STL 檔案以開始檢視")
            
            # 顯示使用說明
            with st.expander("📖 使用說明", expanded=True):
                st.markdown("""
                **基本操作:**
                - 🖱️ **滑鼠左鍵拖曳**: 旋轉模型
                - 🖱️ **滑鼠右鍵拖曳**: 平移視角
                - 🖱️ **滑鼠滾輪**: 縮放檢視
                
                **觸控操作:**
                - 👆 **單指拖曳**: 旋轉模型
                - ✌️ **雙指拖曳**: 平移視角
                - 🤏 **雙指縮放**: 縮放檢視
                
                **參數說明:**
                - **材質類型**: 選擇不同的渲染模式
                - **透明度**: 調整物件的透明程度
                - **光澤度**: 在 material 模式下調整反光效果
                - **相機角度**: 設定初始檢視角度
                - **自動旋轉**: 啟用後模型會持續旋轉
                """)
    
    with col2:
        st.subheader("🎛️ 控制面板")
        
        # 重置按鈕
        if st.button("🔄 重置設定", use_container_width=True):
            st.rerun()
        
        # 範例檔案下載
        st.subheader("📦 範例檔案")
        st.markdown("""
        如果你沒有 STL 檔案，可以：
        1. 從網路下載免費的 STL 檔案
        2. 使用 CAD 軟體建立簡單的 3D 模型
        3. 搜尋 "free STL files" 尋找測試檔案
        """)
        
        # 當前設定摘要
        with st.expander("📊 當前設定", expanded=True):
            settings_summary = {
                "顏色": color,
                "材質": material,
                "透明度": f"{opacity:.1f}",
                "自動旋轉": "是" if auto_rotate else "否",
                "檢視器高度": f"{viewer_height}px"
            }
            
            for setting, value in settings_summary.items():
                st.write(f"**{setting}**: {value}")

if __name__ == "__main__":
    main()