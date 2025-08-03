import os
import shutil
import atexit
import tempfile
from typing import Literal
import streamlit.components.v1 as components


parent_dir = os.path.dirname(os.path.abspath(__file__))

class ThreeDModelViewer:
    def __init__(self):
        # 3D 模型檢視器初始化
        self._initialized = False
        self._working_directory = None
        self._temporary_files_registry = []
        self._prepare_environment()

    def _prepare_environment(self):
        # 準備執行環境，確保只執行一次初始化
        if self._initialized:
            return
            
        # 清理舊有的工作目錄
        if self._working_directory and os.path.exists(self._working_directory):
            shutil.rmtree(self._working_directory)
            
        # 建立新的臨時工作目錄
        self._working_directory = tempfile.mkdtemp(suffix='_3d_viewer')
        
        # 複製元件所需的資源檔案
        for item in os.listdir(parent_dir):
            source_path = os.path.join(parent_dir, item)
            target_path = os.path.join(self._working_directory, item)
            if os.path.isdir(source_path):
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy(source_path, target_path)

        # 標記初始化完成
        self._initialized = True  

    def render_from_text(self, 
                        text: str,
                        color: str = '#696969', 
                        material: Literal['material', 'flat', 'wireframe'] = 'material',
                        auto_rotate: bool = False,
                        opacity: int = 1,
                        shininess: int = 100,
                        cam_v_angle: int = 60,
                        cam_h_angle: int = -90,
                        cam_distance: int = 0,
                        height: int = 500,
                        max_view_distance: int = 1000,
                        show_performance: bool = False,
                        **kwargs):
        # 使用文字內容渲染 3D STL 模型
        # text: STL 檔案的文字內容
        # color: 物件顏色，必須是十六進制格式（如 '#696969'）
        # material: 材質風格 ('material'|'flat'|'wireframe')
        # auto_rotate: 是否自動旋轉
        # opacity: 透明度 (0-1)
        # shininess: 光澤度
        # cam_v_angle: 攝影機垂直角度
        # cam_h_angle: 攝影機水平角度
        # cam_distance: 攝影機距離
        # height: 檢視器高度（像素）
        # max_view_distance: 最大檢視距離
        # show_performance: 是否啟用效能監視器
        # 回傳 True 表示成功建立元件
        
        self._prepare_environment()
        stored_file_path = ""
        
        # 驗證材質參數
        valid_materials = ('material', 'flat', 'wireframe')
        if material not in valid_materials:
            raise ValueError(f'材質必須是 {valid_materials} 之一，但收到 {material}')
            
        # 驗證顏色格式
        if not color.startswith('#'):
            raise ValueError(f"顏色必須是以 '#' 開頭的十六進制值，但收到 {color}")
            
        if text is not None:
            # 在工作目錄中建立暫存檔案
            try:
                temp_file_handle = tempfile.NamedTemporaryFile(
                    dir=self._working_directory, 
                    suffix='.stl', 
                    delete=False
                )
                
                with temp_file_handle as file_handle:
                    if isinstance(text, bytes):
                        file_handle.write(text)
                    elif isinstance(text, str):
                        file_handle.write(text.encode("utf-8"))
                    else:
                        raise ValueError("STL 檔案內容必須是字串或位元組格式")
                    
                    file_handle.flush()
                    stored_file_path = os.path.basename(temp_file_handle.name)
                    self._temporary_files_registry.append(temp_file_handle.name)

            except Exception as error:
                print(f"處理 STL 檔案時發生錯誤: {error}")
                _component_func(files_text='', height=height, **kwargs)
                return False

        # 呼叫 Streamlit 元件
        _component_func(
            file_path=stored_file_path, 
            color=color, 
            material=material, 
            auto_rotate=bool(auto_rotate), 
            opacity=opacity, 
            shininess=shininess,
            cam_v_angle=cam_v_angle,
            cam_h_angle=cam_h_angle,
            cam_distance=cam_distance,
            height=height, 
            max_view_distance=max_view_distance,
            show_performance=show_performance,
            **kwargs
        )
        return True

    def render_from_file(self, 
                      file_path: str, 
                      color: str = '#696969',
                      material: Literal['material', 'flat', 'wireframe'] = 'material',
                      auto_rotate: bool = False,
                      opacity: int = 1, 
                      shininess: int = 100,
                      cam_v_angle: int = 60,
                      cam_h_angle: int = -90,
                      cam_distance: int = 0,
                      height: int = 500,
                      max_view_distance: int = 1000,
                      show_performance: bool = False,
                      **kwargs):
        # 從檔案路徑渲染 3D STL 模型
        # file_path: STL 檔案路徑
        # 其他參數說明同 render_from_text 方法
        # 回傳 True 表示成功建立元件

        file_content = None

        # 讀取檔案內容並轉換為適當格式
        if file_path is not None:
            with open(file_path, "rb") as file_handle:
                file_content = file_handle.read()
        
        # 委託給文字渲染方法處理
        return self.render_from_text(
            text=file_content, 
            color=color, 
            material=material, 
            auto_rotate=auto_rotate, 
            opacity=opacity,
            shininess=shininess,
            height=height, 
            cam_v_angle=cam_v_angle,
            cam_h_angle=cam_h_angle,
            cam_distance=cam_distance,
            max_view_distance=max_view_distance,
            show_performance=show_performance,
            **kwargs
        )

    def dispose_resources(self):
        # 清理暫存檔案和目錄資源
        # 移除整個工作目錄
        try:
            if os.path.exists(self._working_directory):
                shutil.rmtree(self._working_directory)

        except Exception as error:
            print(f"刪除暫存目錄 {self._working_directory} 時發生錯誤: {error}")
            # 如果無法刪除目錄，嘗試逐一刪除檔案
            for temp_file_path in self._temporary_files_registry:
                try:
                    os.unlink(temp_file_path)
                except Exception as file_error:
                    print(f"刪除暫存檔案 {temp_file_path} 時發生錯誤: {file_error}")

# 建立 3D 模型檢視器實例並設定資源管理
model_viewer = ThreeDModelViewer()
# 註冊程式結束時的清理函數
atexit.register(model_viewer.dispose_resources)


# 提供對外的介面函數 - 使用新的命名規範
render_from_text = model_viewer.render_from_text
render_from_file = model_viewer.render_from_file


# 宣告 Streamlit 元件並連結至工作目錄
_component_func = components.declare_component(
    "three_d_model_viewer",
    path=model_viewer._working_directory,
)