import os
import numpy as np
import zstandard as zstd
import streamlit as st
from skimage import measure
from stl import mesh
import scipy.ndimage
from pathlib import Path
import time

from cnc_genai.demo_ui.custom_component.three_d_model_viewer import render_from_file


def load_from_zst(input_path):
    """Load and decompress ZST file containing 3D array data."""
    # Read compressed file
    with open(input_path, "rb") as f:
        compressed = f.read()

    # Decompress
    decompressor = zstd.ZstdDecompressor()
    decompressed = decompressor.decompress(compressed)

    # Parse shape and origin from filename
    # Example filename: 6628_shape=1348_1954_63_3_origin=0_0_-1030.zst
    shape = input_path.split("=")[1].replace("_origin", "")
    origin = input_path.split("=")[-1].replace(".zst", "")
    matrix_shape = [int(float(x)) for x in shape.split("_")]
    matrix_origin = [int(float(x)) for x in origin.split("_")]
    
    # Convert bytes to numpy array
    decompressed_matrix = np.frombuffer(decompressed, dtype=np.uint8).reshape(matrix_shape)
    return decompressed_matrix, matrix_origin


def array_to_stl(voxel_array, filename='temp.stl', downsample_factor=None, step_size=2,
                 scale_factor=0.25, target_voxel_count=7_000_000):
    """Convert voxel array to STL file with detailed performance metrics and dynamic downsampling."""
    
    start_time = time.time()
    
    original_voxel_count = np.prod(voxel_array.shape, dtype=np.uint64)
    print(f"🔢 原始體素陣列: {voxel_array.shape} = {original_voxel_count:,} 體素")
    
    # Step 1: Dynamic downsample calculation
    if downsample_factor is None:
        if original_voxel_count > target_voxel_count:
            # 計算需要的降採樣係數，使體素數量不超過目標值
            downsample_factor = (original_voxel_count / target_voxel_count) ** (1/3)
            downsample_factor = max(1.0, downsample_factor)  # 確保不小於1
            print(f"🎯 動態計算降採樣係數: {downsample_factor:.2f} (目標: {target_voxel_count:,} 體素)")
        else:
            downsample_factor = 1.0
            print(f"✅ 體素數量已在目標範圍內，無需降採樣")
    else:
        print(f"📌 使用指定降採樣係數: {downsample_factor}")
    
    # Apply downsampling
    if downsample_factor > 1.0:
        voxel_array = scipy.ndimage.zoom(voxel_array, zoom=1/downsample_factor, order=0)
        new_voxel_count = np.prod(voxel_array.shape)
        reduction_percent = (1 - new_voxel_count/original_voxel_count) * 100
        print(f"📉 降採樣後: {voxel_array.shape} = {new_voxel_count:,} 體素 ({reduction_percent:.2f}% 縮減)")
    
    # Step 2: Generate mesh
    mesh_start = time.time()
    verts, faces, normals, values = measure.marching_cubes(voxel_array, level=0.5, step_size=step_size)
    mesh_time = time.time() - mesh_start
    
    print(f"🔺 生成網格: {len(verts):,} 頂點, {len(faces):,} 面片 (耗時: {mesh_time:.2f}s)")
    
    # Step 3: Scale vertices
    if scale_factor != 1.0:
        verts = verts * scale_factor
        print(f"📏 尺寸縮放: {scale_factor}x")
    
    # Step 4: Create STL
    stl_start = time.time()
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[f[j], :]
    
    stl_mesh.save(filename)
    stl_time = time.time() - stl_start
    
    # 分析檔案大小
    file_size = Path(filename).stat().st_size
    total_time = time.time() - start_time
    
    print(f"💾 STL 檔案: {filename} ({file_size/1024:.2f} KB)")
    print(f"⏱️  總處理時間: {total_time:.2f}s (網格生成: {mesh_time:.2f}s, STL建立: {stl_time:.2f}s)")
    
    return filename


def display_stl(file_path: str, height: int = 390,
                downsample_factor: int = None, step_size: int = 2,
                scale_factor: float = 0.25,
                target_voxel_count: int = 7_000_000,
                debug_mode: bool = False) -> None:
    """
    Display an STL file with enhanced debugging capabilities and dynamic downsampling.
    """
    
    if debug_mode:
        st.subheader("🔧 3D 檢視器診斷面板")
        
        # 創建診斷欄位
        debug_col1, debug_col2 = st.columns(2)
        
        with debug_col1:
            st.write("**檔案診斷:**")
            st.write(f"- 輸入路徑: `{file_path}`")
            st.write(f"- 檔案存在: {'✅' if Path(file_path).exists() else '❌'}")
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                st.write(f"- 檔案大小: {file_size / 1024:.2f} KB")
            
        with debug_col2:
            st.write("**組件參數:**")
            st.write(f"- 檢視器高度: {height}px")
            st.write(f"- 降採樣係數: {'自動計算' if downsample_factor is None else downsample_factor}")
            st.write(f"- 目標體素數: {target_voxel_count:,}")
            st.write(f"- Step Size: {step_size}")
            st.write(f"- Scale Factor: {scale_factor}")
    
    try:
        # 處理 ZST 轉 STL 邏輯
        if file_path.endswith('.zst'):
            zst_dir = os.path.dirname(file_path)
            stl_path = os.path.join(zst_dir, 'final_simulated.stl')
            
            if not Path(stl_path).exists():
                st.info("🔄 正在轉換 ZST 檔案...")
                
                voxel_array, _ = load_from_zst(file_path)
                
                if len(voxel_array.shape) == 4 and voxel_array.shape[-1] == 3:
                    voxel_array = np.all(voxel_array == [0, 255, 0], axis=-1).astype(np.uint8)
                
                array_to_stl(
                    voxel_array, 
                    filename=stl_path,
                    downsample_factor=downsample_factor,
                    step_size=step_size,
                    scale_factor=scale_factor,
                    target_voxel_count=target_voxel_count
                )
                
                st.success("✅ STL 轉換完成")
        else:
            stl_path = file_path

        if debug_mode:
            st.write(f"**最終 STL 路徑:** `{stl_path}`")
            st.write(f"**STL 檔案存在:** {'✅' if Path(stl_path).exists() else '❌'}")
            
            if Path(stl_path).exists():
                stl_size = Path(stl_path).stat().st_size
                st.write(f"**STL 檔案大小:** {stl_size / 1024:.2f} KB")

        # 顯示 3D 模型
        if Path(stl_path).exists():
            if debug_mode:
                st.write("🎯 **正在載入 3D 檢視器...**")
            
            # 添加容器邊框便於識別
            container_style = """
            <div style="border: 2px solid #ff6b6b; border-radius: 8px; padding: 10px; margin: 10px 0;">
                <p style="color: #ff6b6b; margin: 0; font-weight: bold;">3D 檢視器容器 (紅色邊框)</p>
            </div>
            """ if debug_mode else ""
            
            if debug_mode:
                st.markdown(container_style, unsafe_allow_html=True)
            
            # 呼叫 3D 檢視器
            success = render_from_file(
                file_path=stl_path,
                color='#696969',
                material='material',
                auto_rotate=False,
                opacity=1,
                shininess=100,
                cam_v_angle=60,
                cam_h_angle=0,
                cam_distance=0,
                height=height,
                max_view_distance=1000,
                show_performance=True,
            )
            
            if debug_mode:
                st.write(f"**組件回傳狀態:** {'✅ 成功' if success else '❌ 失敗'}")
        else:
            st.error(f"STL 檔案不存在: {stl_path}")
                
    except Exception as e:
        st.error(f"Error displaying STL: {str(e)}")
        if debug_mode:
            import traceback
            st.code(traceback.format_exc())