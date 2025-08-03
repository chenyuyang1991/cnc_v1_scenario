## 概述
`simulate.py` 是一個用於模擬 CNC（計算機數控）加工過程的 Python 腳本。它通過讀取包含 CNC 程式的 DataFrame (`df`) 和工具信息 (`tools_df`)，模擬刀具在毛坯上的切割路徑，並生成相應的圖像和數據記錄。

## 主要功能
1. **篩選子程序程式碼行**：根據 `code_id` 篩選出特定的 CNC 程式行。
2. **建立遮罩**：建立與原始圖像大小相同的遮罩，用於加速計算。
3. **逐行處理 CNC 程式**：根據每行程式的指令類型（如 G01、G02、G03），進行相應的切割模擬。
4. **計算切割參數**：計算切割區域、切深、切寬等參數。
5. **更新圖像**：根據切割結果更新毛坯圖像。
6. **保存中間結果**：將每一步子程序的模擬結果保存為 Excel 文件和圖像文件。

## 函數說明
### `run`
#### 參數
- `df`: 解析後的 CNC 程式的 DataFrame。
- `code_id`: 子程序的唯一標識符。
- `tools_df`: 包含刀具信息的 DataFrame。
- `image`: 毛坯圖像（numpy 陣列）。
- `center`: 座標系原點。
- `precision`: 精度，默认为 4。
- `verbose`: 是否輸出詳細日誌，默认为 False。
- `r_slack`: 刀具半徑的鬆弛量，默认为 0。
- `z_slack`: Z 軸方向的鬆弛量，默认为 0。
- `timestamp`: 時間戳，默认为當前時間。

#### 返回值
- `image`: 更新後的毛坯圖像。

#### 處理步驟
1. **篩選子程序程式碼行**：
   - 根據 `code_id` 篩選出特定的 CNC 程式行。
   - 初始化刀具位置，填充缺失的 X、Y、Z 座標值。

2. **建立遮罩**：
   - 建立與原始圖像大小相同的遮罩，用於加速計算。

3. **逐行處理 CNC 程式**：
   - 遍歷每一行 CNC 程式，獲取當前刀具信息。
   - 將物理座標轉換為像素座標。
   - 根據指令類型（G01、G02、G03）調用相應的繪圖函數（`draw_G01_cv`, `draw_G02_cv`, `draw_G03_cv`）生成切割遮罩。
   - 計算切割區域、切深、切寬等參數。
   - 更新毛坯圖像。

4. **保存中間結果**：
   - 將每一步的模擬結果保存為 Excel 文件。
   - 保存更新後的毛坯圖像。

## 關鍵函數調用
- `physical_to_pixel`: 將物理座標轉換為像素座標。
- `draw_G01_cv`, `draw_G02_cv`, `draw_G03_cv`: 繪製直線、順時針圓弧、逆時針圓弧的切割遮罩。
- `identify_is_valid`: 判斷切割是否有效。
- `calculate_ap`: 計算切深。
- `calculate_ae`: 計算切寬。
- `update_image`: 更新毛坯圖像。
- `display_recent_cutting`: 顯示最近的切割結果。
- `save_to_zst`, `load_from_zst`: 保存和加載壓縮圖像文件。

## 注意事項
- 該腳本依賴於多個外部模組和自定義函數，確保這些模組和函數已正確導入和實現。
- 模擬過程中會生成大量的中間文件，建議定期清理或管理這些文件以節省存儲空間。
- 模擬過程中可能會遇到性能瓶頸，特別是在處理大規模圖像或複雜路徑時，可以考慮優化算法或使用更高效的計算資源。

## 示例用法
```python
import os
import pandas as pd
import numpy as np
from datetime import datetime
from cnc_genai.src.simulation.simulate import run
from cnc_genai.src.simulation.colors import *

# 加载数据
funcs = pd.read_excel('../app/simulation_master/X2867_CNC2/product_master.xlsx')
funcs['sub_program'] = funcs['sub_program'].astype(int).astype(str)
funcs['sub_program_last'] = funcs['sub_program'].shift(1)

# 獲取解析後的GCODE代碼
df = pd.read_excel('./cnc_genai/parsed_code/command_extract.xlsx')
df = df.drop_duplicates(['row_id','src', 'code_id'], keep='last').reset_index()

# 獲取刀具信息
tools_df = pd.read_excel('./cnc_genai/data/X2867刀具.xlsx').drop_duplicates(['刀號','規格型號'])

# 定義仿真精度
precision = 4

# 創建保存結果的文件夾
timestamp = datetime.today().strftime("%y%m%d_%H%M%S")
os.makedirs(f"../cnc_intermediate/simulation/simulation_npys_{timestamp}", exist_ok=True)

# 對每個子程序進行模擬 
for idx, row in funcs.iterrows():
    if pd.isna(row['sub_program_last']):
        THICKNESS = 2.3 
        EDGE_W = 8.2 
        EDGE_H = 10.2
        SIZE_X = 548.63 # mm
        SIZE_Y = 376.32
        SIZE_Z = 12.46
    
        # 注意xy
        size = np.array([SIZE_X, SIZE_Y, SIZE_Z])
        pixel_size = np.round(size * 10 ** (precision-3)).astype(int)
    
        thickness = int(THICKNESS * 10 ** (precision-3))
        edge_w = int(EDGE_W * 10 **(precision-3))
        edge_h = int(EDGE_H * 10 ** (precision-3))
    
        image = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2], 3), np.uint8)
        image[:] = MATERIAL_COLOR
        image[thickness:-thickness,thickness:-thickness,thickness:-thickness] = EMPTY_COLOR
        image[edge_w:-edge_w, edge_h:-edge_h, thickness:] = EMPTY_COLOR
        
        np.save(f'../cnc_intermediate/simulation/simulation_npys_{timestamp}/{row["sub_program"]}_input_precision={precision}.npy', image)
    else:
        print(f'Loading from {row["sub_program_last"]}')
        image = np.load(f'../cnc_intermediate/simulation/simulation_npys_{timestamp}/{row["sub_program_last"]}_output_precision={precision}.npy')
        pixel_size = np.array([image.shape[1], image.shape[0], image.shape[2]]).astype(int)
        
    center = np.array([
        pixel_size[0] // 2,
        pixel_size[1] // 2,
        0,
    ]).astype(int) # G54
    print(f'----毛坯尺寸: {pixel_size}')
    print(f'----坐標原點位置: {center}')

    # verbose=False 不渲染图片能节约20%时间
    out_image = run(df, row["sub_program"], tools_df, image, center, precision=4, verbose=False, r_slack=1, z_slack=1, timestamp=timestamp)
    np.save(f'../cnc_intermediate/simulation/simulation_npys_{timestamp}/{row["sub_program"]}_output_precision={precision}.npy', out_image)
 ```