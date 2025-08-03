import numpy as np

# import cupynumeric as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy
import shutil
from datetime import datetime
from functools import wraps
import zstandard as zstd
import matplotlib.pyplot as plt
import trimesh
import json
from multiprocessing import Pool
from functools import partial
import threading
import os
import platform
import subprocess
import time
from viztracer import log_sparse

from cnc_genai.src.simulation.colors import (
    MATERIAL_MASK_COLOR,
    EMPTY_MASK_COLOR,
    CUTTING_MASK_COLOR,
    PATH_MASK_COLOR,
    MATERIAL_COLOR,
    EMPTY_COLOR,
    CUTTING_COLOR,
)


# 全域 tracer 實例和執行緒鎖
_global_tracer = None
_tracer_lock = threading.Lock()


# 創建一個空的tracer類來防止NoneType錯誤
class DummyTracer:
    def log_event(self, event_name):
        class DummyContextManager:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return DummyContextManager()

    def start(self):
        pass

    def stop(self):
        pass

    def save(self, output_file=None):
        pass


def get_dummy_tracer():
    return DummyTracer()


def get_smart_tracer():
    """
    提供全域唯一的 VizTracer 實例（單例模式）
    """
    global _global_tracer

    # 如果已經有 tracer 實例，直接返回
    if _global_tracer is not None:
        return _global_tracer

    # 使用鎖確保執行緒安全
    with _tracer_lock:
        # 雙重檢查鎖定模式
        if _global_tracer is not None:
            return _global_tracer

        # 檢查是否啟用 VizTracer
        enable_viztracer = os.getenv("ENABLE_VIZTRACER", "0").lower() in ("1", "true")

        if not enable_viztracer:
            print("VizTracer disabled, using DummyTracer")
            _global_tracer = get_dummy_tracer()
            return _global_tracer

        try:
            from viztracer import VizTracer

            print("Creating global VizTracer instance")

            _global_tracer = VizTracer(
                log_sparse=True,
                max_stack_depth=6,
                tracer_entries=50000000,  # 增加記憶體使用量，因為是全域共享
            )

            return _global_tracer

        except ImportError:
            print("Warning: VizTracer not installed, using DummyTracer")
            _global_tracer = get_dummy_tracer()
            return _global_tracer
        except Exception as e:
            print(f"Warning: VizTracer initialization failed: {e}, using DummyTracer")
            _global_tracer = get_dummy_tracer()
            return _global_tracer


def reset_global_tracer():
    """
    重置全域 tracer 實例（主要用於測試）
    """
    global _global_tracer

    with _tracer_lock:
        if _global_tracer is not None:
            try:
                # 嘗試正確關閉 tracer
                if hasattr(_global_tracer, "stop"):
                    _global_tracer.stop()
            except Exception as e:
                print(f"Warning: Error stopping tracer during reset: {e}")
            finally:
                _global_tracer = None
                print("Global tracer reset")


def get_thread_tracer_info():
    """
    獲取全域 tracer 資訊（用於調試）
    """
    global _global_tracer

    thread_name = threading.current_thread().name
    thread_id = threading.get_ident()

    has_tracer = _global_tracer is not None
    tracer_type = type(_global_tracer).__name__ if has_tracer else "None"

    return {
        "thread_name": thread_name,
        "thread_id": thread_id,
        "has_global_tracer": has_tracer,
        "tracer_type": tracer_type,
        "tracer_instance": _global_tracer,
    }


class SafeFileHandler:
    """安全的檔案處理器，支援跨平台檔案鎖定"""

    # 類級別的 fcntl 模組快取
    _fcntl = None
    _fcntl_checked = False

    @classmethod
    def _get_fcntl(cls):
        """獲取 fcntl 模組（僅在 Unix/Linux 系統上可用）"""
        if not cls._fcntl_checked:
            try:
                import fcntl

                cls._fcntl = fcntl
            except ImportError:
                cls._fcntl = None
            cls._fcntl_checked = True
        return cls._fcntl

    @staticmethod
    def safe_read_json(file_path, max_retries=3, retry_delay=0.1):
        """安全讀取 JSON 檔案"""
        file_path = Path(file_path)

        for attempt in range(max_retries):
            try:
                if not file_path.exists():
                    return None

                if platform.system() == "Windows":
                    # Windows: 使用共享讀取模式
                    with open(file_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                else:  # Linux/macOS
                    # Unix: 使用檔案鎖
                    fcntl = SafeFileHandler._get_fcntl()
                    with open(file_path, "r", encoding="utf-8") as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # 共享鎖
                        try:
                            return json.load(f)
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 解鎖

            except (PermissionError, OSError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    print(
                        f"[SafeFileHandler] 讀取失敗 (嘗試 {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[SafeFileHandler] 最終讀取失敗: {e}")
                    return None

        return None

    @staticmethod
    def safe_write_json(file_path, data, max_retries=3, retry_delay=0.1):
        """安全寫入 JSON 檔案"""
        file_path = Path(file_path)

        for attempt in range(max_retries):
            try:
                if platform.system() == "Windows":
                    # Windows: 使用臨時檔案 + 原子性替換
                    temp_file = file_path.with_suffix(".tmp")
                    with open(temp_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    # 原子性替換
                    if file_path.exists():
                        file_path.unlink()
                    temp_file.rename(file_path)

                else:  # Linux/macOS
                    # Unix: 使用檔案鎖
                    fcntl = SafeFileHandler._get_fcntl()
                    with open(file_path, "w", encoding="utf-8") as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 獨佔鎖
                        try:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 解鎖

                return True

            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    print(
                        f"[SafeFileHandler] 寫入失敗 (嘗試 {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[SafeFileHandler] 最終寫入失敗: {e}")
                    return False

        return False


class SimulationDataAppender:
    """
    高效能資料追加器，支援 Parquet 和 Excel 格式
    使用單一追蹤檔案簡化進度監控
    """

    def __init__(self, output_path, sub_program, file_format="parquet"):
        """
        初始化 ParquetAppender

        Args:
            output_path: 輸出路徑
            sub_program: 子程式名稱
            file_format: 檔案格式 ("parquet" 或 "excel")
            sheet: Excel sheet 名稱 (僅當 file_format="excel" 時使用)
        """
        self.output_path = Path(output_path)
        self.sub_program = sub_program
        self.file_format = file_format.lower()
        self.sheet = "Sheet1"
        self.buffer = []
        self.lock = threading.Lock()
        self.start_time = datetime.now()

        # 設定檔案路徑
        if self.file_format == "excel":
            self.final_path = self.output_path / f"{sub_program}.xlsx"
        else:
            self.final_path = self.output_path / f"{sub_program}.parquet"

        # 使用單一追蹤檔案
        self.tracking_file = self.output_path / f"{sub_program}.tracking"

        # 確保目錄存在
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 移除舊檔案
        if self.final_path.exists():
            self.final_path.unlink()
            print(f"Removed existing file: {self.final_path}")

        # 移除舊有追蹤檔案
        if self.tracking_file.exists():
            self.tracking_file.unlink()
            print(f"Removed existing tracking file: {self.tracking_file}")

        # 初始化追蹤檔案
        self._init_tracking()

        print(f"Will create new {self.file_format.upper()} file: {self.final_path}")

    def _init_tracking(self):
        """初始化追蹤檔案"""
        tracking_data = {
            "sub_program": self.sub_program,
            "status": "running",
            "start_time": self.start_time.isoformat(),
            "last_update": datetime.now().isoformat(),
            "rows_count": 0,
            "file_format": self.file_format,
            "final_path": str(self.final_path),
            "is_complete": False,
        }

        self._write_tracking(tracking_data)
        print(f"Tracking initialized for {self.sub_program}")

    def _write_tracking(self, data):
        """寫入追蹤檔案"""
        success = SafeFileHandler.safe_write_json(self.tracking_file, data)
        if not success:
            print(f"Warning: Failed to write tracking file: {self.tracking_file}")

    def _update_tracking(self, status="running", **kwargs):
        """更新追蹤檔案"""
        try:
            # 讀取現有資料
            tracking_data = SafeFileHandler.safe_read_json(self.tracking_file)
            if tracking_data is None:
                tracking_data = {}

            # 更新資料
            tracking_data.update(
                {
                    "status": status,
                    "last_update": datetime.now().isoformat(),
                    "rows_count": len(self.buffer),
                    **kwargs,
                }
            )

            self._write_tracking(tracking_data)

        except Exception as e:
            print(f"Warning: Failed to update tracking: {e}")

    @log_sparse(stack_depth=3)
    def append_row(self, series_row: pd.Series):
        """添加一行到記憶體緩衝區"""
        with self.lock:
            self.buffer.append(series_row.to_dict())

            # 每1000行更新一次追蹤
            if len(self.buffer) % 1000 == 0:
                # self._update_tracking("running")
                pass

    def _write_parquet_file(self, df):
        """寫入 Parquet 檔案"""
        try:
            df.to_parquet(
                self.final_path, engine="pyarrow", compression="snappy", index=False
            )
            return True
        except Exception as e:
            print(f"Error writing Parquet file: {e}")
            return False

    def _write_excel_file(self, df):
        """寫入 Excel 檔案"""
        try:
            df.to_excel(
                self.final_path, sheet_name=self.sheet, index=False, engine="openpyxl"
            )
            return True
        except Exception as e:
            print(f"Error writing Excel file: {e}")
            return False

    @log_sparse(stack_depth=3)
    def close(self):
        """關閉並寫入最終檔案"""
        start_time = datetime.now()

        try:
            with self.lock:
                if not self.buffer:
                    print("No data to write")
                    self._update_tracking(
                        "completed",
                        is_complete=True,
                        end_time=datetime.now().isoformat(),
                    )
                    return

                # 更新追蹤為正在寫入
                self._update_tracking("writing")

                # 轉換為 DataFrame
                df = pd.DataFrame(self.buffer)
                data_count = len(df)

                print(
                    f"Writing {data_count} rows to {self.file_format.upper()} file..."
                )

                # 根據格式寫入檔案
                if self.file_format == "excel":
                    success = self._write_excel_file(df)
                else:
                    success = self._write_parquet_file(df)

                execution_time = datetime.now() - start_time

                if success:
                    file_size = (
                        self.final_path.stat().st_size
                        if self.final_path.exists()
                        else 0
                    )

                    print(
                        f"Successfully wrote {data_count} rows to {self.final_path} in {execution_time.total_seconds():.3f} seconds"
                    )

                    # 更新追蹤為完成
                    self._update_tracking(
                        "completed",
                        is_complete=True,
                        end_time=datetime.now().isoformat(),
                        execution_time_seconds=execution_time.total_seconds(),
                        file_size_bytes=file_size,
                        total_rows=data_count,
                    )

                    print(f"Tracking updated for completed {self.sub_program}")
                else:
                    self._update_tracking(
                        "failed",
                        is_complete=False,
                        error="Failed to write file",
                        end_time=datetime.now().isoformat(),
                    )

        except Exception as e:
            print(f"Error during close: {e}")
            self._update_tracking(
                "failed",
                is_complete=False,
                error=str(e),
                end_time=datetime.now().isoformat(),
            )
            raise

    def get_tracking_info(self):
        """獲取追蹤資訊"""
        try:
            return SafeFileHandler.safe_read_json(self.tracking_file)
        except Exception as e:
            print(f"Error reading tracking file: {e}")
            return None

    def is_complete(self):
        """檢查是否完成"""
        tracking_info = self.get_tracking_info()
        return tracking_info and tracking_info.get("is_complete", False)

    @staticmethod
    def merge_all_simulations(product_master_path, excel_out_path):
        """
        合併所有仿真結果為 all_simulated.parquet 和 all_simulated.xlsx

        Args:
            product_master_path: product_master.xlsx 的路徑
            excel_out_path: 仿真結果輸出目錄路徑

        Returns:
            dict: 合併結果的統計資訊
        """
        from pathlib import Path
        import pandas as pd
        import glob

        # 初始化 all_simulated 的 tracking 機制
        excel_out_path = Path(excel_out_path)
        all_simulated_tracking = excel_out_path / "all_simulated.tracking"
        start_time = datetime.now()

        # 創建初始 tracking 檔案
        tracking_data = {
            "sub_program": "all_simulated",
            "status": "running",
            "start_time": start_time.isoformat(),
            "last_update": datetime.now().isoformat(),
            "rows_count": 0,
            "file_format": "parquet_and_excel",
            "final_path": str(excel_out_path / "all_simulated"),
            "is_complete": False,
        }

        # 確保目錄存在
        excel_out_path.mkdir(parents=True, exist_ok=True)

        # 移除舊的 tracking 檔案
        if all_simulated_tracking.exists():
            all_simulated_tracking.unlink()
            print(
                f"Removed existing all_simulated tracking file: {all_simulated_tracking}"
            )

        # 寫入初始 tracking 檔案
        SafeFileHandler.safe_write_json(all_simulated_tracking, tracking_data)
        print(f"all_simulated tracking initialized")

        try:
            # 讀取 product_master.xlsx 獲取 sub_program 列表
            product_master_path = Path(product_master_path)
            excel_out_path = Path(excel_out_path)

            if not product_master_path.exists():
                raise FileNotFoundError(
                    f"Product master file not found: {product_master_path}"
                )

            # 讀取 product_master.xlsx
            try:
                funcs = pd.read_excel(product_master_path)
                if "sub_program" not in funcs.columns:
                    # 嘗試第二行作為 header
                    funcs = pd.read_excel(product_master_path, header=1)
            except Exception as e:
                raise ValueError(f"Failed to read product_master.xlsx: {e}")

            if "sub_program" not in funcs.columns:
                raise ValueError("sub_program column not found in product_master.xlsx")

            # 清理和準備 sub_program 資料
            funcs = funcs.drop_duplicates(["sub_program"], keep="first")
            funcs["sub_program"] = (
                funcs["sub_program"].astype(int).astype(str).str.zfill(4)
            )

            print(f"Found {len(funcs)} sub_programs in product_master.xlsx")

            # 合併所有仿真結果
            simulated_dfs = []
            merge_stats = {
                "total_sub_programs": len(funcs),
                "successful_merges": 0,
                "failed_merges": 0,
                "file_format_used": {},
                "missing_files": [],
                "errors": [],
            }

            for idx, row in funcs.iterrows():
                sub_program = str(row["sub_program"])

                try:
                    # 優先查找 parquet 檔案
                    parquet_path = excel_out_path / f"{sub_program}.parquet"
                    excel_path = excel_out_path / f"{sub_program}.xlsx"

                    simulated_sub_df = None
                    file_format_used = None

                    if parquet_path.exists():
                        # 使用 parquet 檔案
                        simulated_sub_df = pd.read_parquet(parquet_path)
                        file_format_used = "parquet"
                        print(f"Loaded {sub_program} from parquet file")
                    elif excel_path.exists():
                        # 使用 excel 檔案
                        simulated_sub_df = pd.read_excel(excel_path)
                        file_format_used = "excel"
                        print(f"Loaded {sub_program} from excel file")
                    else:
                        # 檔案不存在
                        merge_stats["missing_files"].append(sub_program)
                        merge_stats["failed_merges"] += 1
                        print(
                            f"Warning: No data file found for sub_program {sub_program}"
                        )
                        continue

                    # 添加額外欄位
                    simulated_sub_df["sub_program"] = sub_program
                    simulated_sub_df["sub_program_key"] = (
                        str(idx + 1).zfill(2) + "-" + sub_program
                    )
                    simulated_sub_df["sub_program_seq"] = idx + 1

                    simulated_dfs.append(simulated_sub_df)
                    merge_stats["successful_merges"] += 1

                    # 統計使用的檔案格式
                    if file_format_used in merge_stats["file_format_used"]:
                        merge_stats["file_format_used"][file_format_used] += 1
                    else:
                        merge_stats["file_format_used"][file_format_used] = 1

                except Exception as e:
                    error_msg = f"Error processing sub_program {sub_program}: {str(e)}"
                    merge_stats["errors"].append(error_msg)
                    merge_stats["failed_merges"] += 1
                    print(f"Error: {error_msg}")
                    continue

            if not simulated_dfs:
                raise ValueError("No simulation data found to merge")

            # 合併所有 DataFrame
            print(f"Merging {len(simulated_dfs)} simulation results...")
            simulated_df = pd.concat(simulated_dfs, axis=0, ignore_index=True)

            # 確保輸出目錄存在
            excel_out_path.mkdir(parents=True, exist_ok=True)

            # 同時儲存為 parquet 和 excel 格式
            parquet_output_path = excel_out_path / "all_simulated.parquet"
            excel_output_path = excel_out_path / "all_simulated.xlsx"

            # 儲存為 parquet 檔案
            try:
                simulated_df.to_parquet(
                    parquet_output_path,
                    engine="pyarrow",
                    compression="snappy",
                    index=False,
                )
                print(
                    f"Successfully saved all_simulated.parquet ({len(simulated_df)} rows)"
                )
                merge_stats["parquet_saved"] = True
                merge_stats["parquet_path"] = str(parquet_output_path)
            except Exception as e:
                merge_stats["parquet_saved"] = False
                merge_stats["parquet_error"] = str(e)
                print(f"Error saving parquet file: {e}")

            # 儲存為 excel 檔案
            try:
                simulated_df.to_excel(excel_output_path, index=False)
                print(
                    f"Successfully saved all_simulated.xlsx ({len(simulated_df)} rows)"
                )
                merge_stats["excel_saved"] = True
                merge_stats["excel_path"] = str(excel_output_path)
            except Exception as e:
                merge_stats["excel_saved"] = False
                merge_stats["excel_error"] = str(e)
                print(f"Error saving excel file: {e}")

            # 更新統計資訊
            merge_stats["total_rows"] = len(simulated_df)
            merge_stats["total_columns"] = len(simulated_df.columns)

            # 計算執行時間
            execution_time = datetime.now() - start_time

            # 更新最終 tracking 狀態
            tracking_data.update(
                {
                    "status": "completed",
                    "is_complete": True,
                    "last_update": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "execution_time_seconds": execution_time.total_seconds(),
                    "merge_stats": merge_stats,
                }
            )
            SafeFileHandler.safe_write_json(all_simulated_tracking, tracking_data)

            print(f"Merge completed successfully:")
            print(f"  - Total sub_programs: {merge_stats['total_sub_programs']}")
            print(f"  - Successful merges: {merge_stats['successful_merges']}")
            print(f"  - Failed merges: {merge_stats['failed_merges']}")
            print(f"  - Total rows: {merge_stats['total_rows']}")
            print(f"  - File formats used: {merge_stats['file_format_used']}")

            return merge_stats

        except Exception as e:
            print(f"Error in merge_all_simulations: {e}")
            raise


def get_process_info(sim_latest_path):
    try:
        process_info_file = os.path.join(sim_latest_path, f"process_info.json")
        with open(process_info_file, "r", encoding="utf-8") as f:
            process_info = json.load(f)
        return process_info
    except Exception as e:
        return None


def check_process_status(process_id):
    """
    檢查進程是否還在運行

    Args:
        process_id: 進程ID

    Returns:
        str: 進程狀態 ("運行中", "已終止", "未知")
    """
    if not process_id or process_id == "未知":
        return "未知"

    try:
        pid = str(process_id).strip()

        if platform.system() == "Windows":
            # 驗證 PID 格式
            if not str(pid).isdigit():
                return "未知"

            # 設定完整的 PowerShell 命令，確保 UTF-8 輸出
            powershell_cmd = f"""
            $OutputEncoding = [System.Text.Encoding]::UTF8;
            [Console]::OutputEncoding = [System.Text.Encoding]::UTF8;
            if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ 'EXISTS' }} else {{ 'NOT_EXISTS' }}
            """

            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", powershell_cmd],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return "運行中" if "EXISTS" == result.stdout.strip() else "已終止"
        else:
            # Linux/macOS 使用 ps -p 直接查詢特定 PID
            result = subprocess.run(
                ["ps", "-p", pid],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return "運行中" if result.returncode == 0 else "已終止"

    except Exception as e:
        return "未知"


class SimulationStatusReader:
    """
    仿真任務狀態讀取器，統一處理仿真任務的狀態資訊
    與 SimulationDataAppender 配合使用，提供完整的仿真任務監控
    """

    @staticmethod
    def upgrade_legacy_simulation_formats(simulation_paths=None):
        """
        檢測並升級舊版本仿真模擬格式為新版本格式

        Args:
            simulation_paths: list 或 None
                要處理的仿真路徑列表。如果為 None，則自動掃描所有仿真目錄

        Returns:
            dict: 升級結果統計
        """
        import glob
        from pathlib import Path
        import json
        from datetime import datetime
        import pandas as pd

        # 如果沒有指定路徑，則自動掃描
        if simulation_paths is None:
            simulation_paths = glob.glob("../app/*/simulation_master/*")

        print(f"[Legacy Upgrade] 開始檢測 {len(simulation_paths)} 個仿真目錄")

        upgrade_stats = {
            "total_checked": 0,
            "legacy_detected": 0,
            "upgrade_success": 0,
            "upgrade_failed": 0,
            "already_upgraded": 0,
            "upgrade_details": [],
        }

        for sim_path in simulation_paths:
            try:
                upgrade_stats["total_checked"] += 1
                sim_path = Path(sim_path)

                print(f"[Legacy Upgrade] 檢查目錄: {sim_path}")

                # 檢查是否為舊版本格式
                is_legacy, legacy_info = SimulationStatusReader._detect_legacy_format(
                    sim_path
                )

                if not is_legacy:
                    upgrade_stats["already_upgraded"] += 1
                    print(f"[Legacy Upgrade] 已是新版本格式，跳過: {sim_path}")
                    continue

                upgrade_stats["legacy_detected"] += 1
                print(f"[Legacy Upgrade] 檢測到舊版本格式: {sim_path}")
                print(f"[Legacy Upgrade] 舊版本資訊: {legacy_info}")

                # 執行升級
                success, upgrade_detail = (
                    SimulationStatusReader._upgrade_single_simulation(
                        sim_path, legacy_info
                    )
                )

                if success:
                    upgrade_stats["upgrade_success"] += 1
                    print(f"[Legacy Upgrade] 升級成功: {sim_path}")
                else:
                    upgrade_stats["upgrade_failed"] += 1
                    print(f"[Legacy Upgrade] 升級失敗: {sim_path}")

                upgrade_stats["upgrade_details"].append(upgrade_detail)

            except Exception as e:
                upgrade_stats["upgrade_failed"] += 1
                error_detail = {
                    "path": str(sim_path),
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                upgrade_stats["upgrade_details"].append(error_detail)
                print(f"[Legacy Upgrade] 處理 {sim_path} 時發生錯誤: {e}")

        # 輸出統計結果
        print(f"\n[Legacy Upgrade] 升級完成統計:")
        print(f"  總檢查數: {upgrade_stats['total_checked']}")
        print(f"  舊版本數: {upgrade_stats['legacy_detected']}")
        print(f"  升級成功: {upgrade_stats['upgrade_success']}")
        print(f"  升級失敗: {upgrade_stats['upgrade_failed']}")
        print(f"  已升級數: {upgrade_stats['already_upgraded']}")

        return upgrade_stats

    @staticmethod
    def _detect_legacy_format(sim_path):
        """
        檢測是否為舊版本格式
        舊版本特徵：有符合條件的 Excel 檔案 + *_temp 目錄
        新版本特徵：有 Parquet 檔案 + *.tracking 檔案

        只處理以下 Excel 檔案：
        1. product_master.xlsx 中記錄的 {sub_program}.xlsx
        2. all_simulated.xlsx

        Args:
            sim_path: Path 仿真目錄路徑

        Returns:
            tuple: (is_legacy: bool, legacy_info: dict)
        """
        sim_latest = sim_path / "simulation" / "latest"

        if not sim_latest.exists():
            return False, {"reason": "simulation/latest 目錄不存在"}

        # 檢查是否有 process_info.json
        process_info_file = sim_latest / "process_info.json"
        if not process_info_file.exists():
            return False, {"reason": "process_info.json 不存在"}

        try:
            # 載入 process_info.json
            process_info = SafeFileHandler.safe_read_json(process_info_file)
            if process_info is None:
                return False, {"reason": "無法讀取 process_info.json"}

            # 載入 product_master.xlsx 獲取有效的子程式列表
            product_master_path = sim_path / "product_master.xlsx"
            if not product_master_path.exists():
                return False, {"reason": "product_master.xlsx 不存在"}

            try:
                df = pd.read_excel(product_master_path, engine="openpyxl")
                valid_sub_programs = set(
                    df["sub_program"].astype(str).str.zfill(4).unique()
                )
            except Exception as e:
                return False, {"reason": f"無法讀取 product_master.xlsx: {e}"}

            # 檢查檔案類型
            all_excel_files = list(sim_latest.glob("*.xlsx"))
            parquet_files = list(sim_latest.glob("*.parquet"))
            tracking_files = list(sim_latest.glob("*.tracking"))
            temp_dirs = list(sim_latest.glob("*_temp"))

            # 篩選出需要處理的 Excel 檔案
            target_excel_files = []
            for excel_file in all_excel_files:
                file_stem = excel_file.stem
                # 只處理 product_master.xlsx 中記錄的子程式和 all_simulated.xlsx
                if file_stem in valid_sub_programs or file_stem == "all_simulated":
                    target_excel_files.append(excel_file)

            # 篩選出需要處理的 temp 目錄
            target_temp_dirs = []
            for temp_dir in temp_dirs:
                temp_stem = temp_dir.stem.replace("_temp", "")
                # 只處理 product_master.xlsx 中記錄的子程式
                if temp_stem in valid_sub_programs:
                    target_temp_dirs.append(temp_dir)

            legacy_info = {
                "process_info": process_info,
                "valid_sub_programs": list(valid_sub_programs),
                "all_excel_files_count": len(all_excel_files),
                "target_excel_files_count": len(target_excel_files),
                "parquet_files_count": len(parquet_files),
                "tracking_files_count": len(tracking_files),
                "all_temp_dirs_count": len(temp_dirs),
                "target_temp_dirs_count": len(target_temp_dirs),
                "target_excel_files": [f.stem for f in target_excel_files],
                "parquet_files": [f.stem for f in parquet_files],
                "tracking_files": [f.stem for f in tracking_files],
                "target_temp_dirs": [
                    d.stem.replace("_temp", "") for d in target_temp_dirs
                ],
                "ignored_excel_files": [
                    f.stem for f in all_excel_files if f not in target_excel_files
                ],
                "ignored_temp_dirs": [
                    d.stem.replace("_temp", "")
                    for d in temp_dirs
                    if d not in target_temp_dirs
                ],
            }

            # 判斷是否為舊版本
            # 舊版本：有目標 Excel 檔案，且沒有對應的 tracking 檔案
            if target_excel_files:
                excel_stems = set([f.stem for f in target_excel_files])
                tracking_stems = set([f.stem for f in tracking_files])

                # 如果有目標 Excel 檔案但沒有對應的 tracking 檔案，判定為舊版本
                missing_tracking = excel_stems - tracking_stems
                if missing_tracking:
                    legacy_info["missing_tracking_for_excel"] = list(missing_tracking)
                    return True, legacy_info

            # 如果有目標 temp 目錄但沒有對應的 tracking 檔案，也判定為舊版本
            if target_temp_dirs:
                temp_stems = set(
                    [d.stem.replace("_temp", "") for d in target_temp_dirs]
                )
                tracking_stems = set([f.stem for f in tracking_files])

                missing_tracking = temp_stems - tracking_stems
                if missing_tracking:
                    legacy_info["missing_tracking_for_temp"] = list(missing_tracking)
                    return True, legacy_info

            return False, legacy_info

        except Exception as e:
            return False, {"reason": f"檢測過程發生錯誤: {e}"}

    @staticmethod
    def _upgrade_single_simulation(sim_path, legacy_info):
        """
        升級單個仿真目錄
        升級步驟：
        1. 為每個符合條件的 Excel 檔案轉換出對應的 Parquet 檔案
        2. 為每個 Parquet 檔案建立對應的 *.tracking 檔案
        3. 找出 current sub_program 並補上 *.tracking 檔案

        只處理以下 Excel 檔案：
        1. product_master.xlsx 中記錄的 {sub_program}.xlsx
        2. all_simulated.xlsx

        Args:
            sim_path: Path 仿真目錄路徑
            legacy_info: dict 舊版本資訊

        Returns:
            tuple: (success: bool, upgrade_detail: dict)
        """
        from datetime import datetime
        import pandas as pd

        sim_latest = sim_path / "simulation" / "latest"
        upgrade_detail = {
            "path": str(sim_path),
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "excel_to_parquet": [],
            "created_tracking_files": [],
            "current_subprogram": None,
            "valid_sub_programs": legacy_info.get("valid_sub_programs", []),
            "ignored_files": {
                "excel": legacy_info.get("ignored_excel_files", []),
                "temp_dirs": legacy_info.get("ignored_temp_dirs", []),
            },
            "errors": [],
        }

        try:
            # 獲取有效的子程式列表
            valid_sub_programs = set(legacy_info.get("valid_sub_programs", []))

            upgrade_detail["total_sub_programs"] = len(valid_sub_programs)

            # Step 1: 為每個符合條件的 Excel 檔案轉換出對應的 Parquet 檔案
            all_excel_files = list(sim_latest.glob("*.xlsx"))
            processed_excel_count = 0

            for excel_file in all_excel_files:
                file_stem = excel_file.stem

                # 只處理 product_master.xlsx 中記錄的子程式和 all_simulated.xlsx
                if file_stem not in valid_sub_programs and file_stem != "all_simulated":
                    print(
                        f"[Upgrade] 跳過不在 product_master.xlsx 中的檔案: {excel_file.name}"
                    )
                    continue

                try:
                    parquet_file = sim_latest / f"{file_stem}.parquet"

                    # 如果 Parquet 檔案已存在，跳過轉換
                    if parquet_file.exists():
                        print(f"[Upgrade] Parquet 檔案已存在，跳過轉換: {parquet_file}")
                        continue

                    # 讀取 Excel 並轉換為 Parquet
                    print(
                        f"[Upgrade] 轉換 Excel 到 Parquet: {excel_file} -> {parquet_file}"
                    )
                    excel_df = pd.read_excel(excel_file, engine="openpyxl")
                    excel_df.to_parquet(
                        parquet_file,
                        engine="pyarrow",
                        compression="snappy",
                        index=False,
                    )

                    upgrade_detail["excel_to_parquet"].append(
                        {
                            "sub_program": file_stem,
                            "excel_file": str(excel_file),
                            "parquet_file": str(parquet_file),
                            "rows_count": len(excel_df),
                        }
                    )
                    processed_excel_count += 1

                except Exception as e:
                    upgrade_detail["errors"].append(
                        f"轉換 {excel_file} 時發生錯誤: {e}"
                    )

            print(
                f"[Upgrade] 已處理 {processed_excel_count} 個 Excel 檔案，跳過 {len(all_excel_files) - processed_excel_count} 個不相關檔案"
            )

            # Step 2: 為每個 Parquet 檔案建立對應的 *.tracking 檔案
            parquet_files = list(sim_latest.glob("*.parquet"))
            created_tracking_count = 0

            for parquet_file in parquet_files:
                file_stem = parquet_file.stem

                # 只處理符合條件的 Parquet 檔案
                if file_stem not in valid_sub_programs and file_stem != "all_simulated":
                    print(f"[Upgrade] 跳過不相關的 Parquet 檔案: {parquet_file.name}")
                    continue

                try:
                    tracking_file = sim_latest / f"{file_stem}.tracking"

                    # 如果 tracking 檔案已存在，跳過
                    if tracking_file.exists():
                        print(f"[Upgrade] Tracking 檔案已存在，跳過: {tracking_file}")
                        continue

                    # 獲取 Parquet 檔案資訊
                    file_stats = parquet_file.stat()
                    file_size = file_stats.st_size
                    file_mtime = datetime.fromtimestamp(file_stats.st_mtime)

                    # 讀取 Parquet 檔案獲取行數
                    try:
                        parquet_df = pd.read_parquet(parquet_file)
                        rows_count = len(parquet_df)
                    except Exception as e:
                        rows_count = 0
                        upgrade_detail["errors"].append(
                            f"無法讀取 {parquet_file.name}: {e}"
                        )

                    # 創建 tracking 檔案（已完成狀態）
                    tracking_data = {
                        "sub_program": file_stem,
                        "status": "completed",
                        "start_time": file_mtime.isoformat(),
                        "last_update": file_mtime.isoformat(),
                        "end_time": file_mtime.isoformat(),
                        "rows_count": rows_count,
                        "total_rows": rows_count,
                        "file_format": "parquet",
                        "final_path": str(parquet_file),
                        "is_complete": True,
                        "file_size_bytes": file_size,
                        "upgrade_note": "由舊版本格式自動升級生成（已完成）",
                        "upgrade_timestamp": datetime.now().isoformat(),
                    }

                    # 寫入 tracking 檔案
                    success = SafeFileHandler.safe_write_json(
                        tracking_file, tracking_data
                    )
                    if success:
                        upgrade_detail["created_tracking_files"].append(
                            f"{file_stem} (completed)"
                        )
                        created_tracking_count += 1
                        print(f"[Upgrade] 創建 tracking 檔案: {tracking_file}")
                    else:
                        upgrade_detail["errors"].append(
                            f"創建 tracking 檔案失敗: {tracking_file}"
                        )

                except Exception as e:
                    upgrade_detail["errors"].append(
                        f"為 {parquet_file} 創建 tracking 檔案時發生錯誤: {e}"
                    )

            print(
                f"[Upgrade] 已為 {created_tracking_count} 個 Parquet 檔案創建 tracking 檔案"
            )

            # Step 3: 找出 current sub_program 並補上 *.tracking 檔案
            # 使用現有的方法來獲取當前子程式，這會自動處理完成狀態檢查
            # 創建一個臨時的 SimulationStatusReader 實例來重用邏輯
            temp_reader = SimulationStatusReader(sim_path)
            current_subprogram = temp_reader._get_current_subprogram_from_temp_dirs()

            upgrade_detail["current_subprogram"] = current_subprogram

            if current_subprogram is None:
                print(
                    f"[Upgrade] 無當前子程式（可能所有子程式都已完成或無相關 temp 目錄）"
                )
            else:
                print(f"[Upgrade] 找到當前子程式: {current_subprogram}")

                tracking_file = sim_latest / f"{current_subprogram}.tracking"
                if not tracking_file.exists():
                    # 為當前正在處理的子程式創建 tracking 檔案
                    # 尋找對應的 temp 目錄來獲取時間戳
                    temp_dir = sim_latest / f"{current_subprogram}_temp"
                    if temp_dir.exists():
                        temp_mtime = datetime.fromtimestamp(temp_dir.stat().st_mtime)
                    else:
                        temp_mtime = datetime.now()

                    tracking_data = {
                        "sub_program": current_subprogram,
                        "status": "running",
                        "start_time": temp_mtime.isoformat(),
                        "last_update": datetime.now().isoformat(),
                        "rows_count": 0,
                        "file_format": "parquet",
                        "final_path": str(sim_latest / f"{current_subprogram}.parquet"),
                        "is_complete": False,
                        "upgrade_note": "由舊版本格式自動升級生成",
                        "upgrade_timestamp": datetime.now().isoformat(),
                        "temp_dir": str(temp_dir) if temp_dir.exists() else None,
                    }

                    SafeFileHandler.safe_write_json(tracking_file, tracking_data)

                    upgrade_detail["created_tracking_files"].append(
                        f"{current_subprogram} (running)"
                    )
                    print(f"[Upgrade] 創建當前子程式 tracking 檔案: {tracking_file}")
                else:
                    print(f"[Upgrade] 當前子程式 tracking 檔案已存在: {tracking_file}")

            upgrade_detail["status"] = "success"
            upgrade_detail["created_count"] = len(
                upgrade_detail["created_tracking_files"]
            )
            upgrade_detail["converted_count"] = len(upgrade_detail["excel_to_parquet"])

            return True, upgrade_detail

        except Exception as e:
            upgrade_detail["status"] = "failed"
            upgrade_detail["errors"].append(f"升級過程發生錯誤: {e}")
            return False, upgrade_detail

    def __init__(self, simulation_path):
        """
        初始化狀態讀取器

        Args:
            simulation_path: 仿真路徑，格式如 "../app/department/simulation_master/clamping"
        """
        self.simulation_path = Path(simulation_path)
        self.department = self.simulation_path.parts[-3]
        self.clamping = self.simulation_path.parts[-1]
        self.sim_latest = self.simulation_path / "simulation" / "latest"
        self.product_master_path = self.simulation_path / "product_master.xlsx"

        # 狀態資訊
        self._process_info = None
        self._sub_programs = None
        self._status_info = None

    def _load_process_info(self):
        """載入進程資訊"""
        if self._process_info is not None:
            return self._process_info

        self._process_info = get_process_info(self.sim_latest)

        return self._process_info

    def _load_sub_programs(self):
        """載入子程式清單"""
        if self._sub_programs is not None:
            return self._sub_programs

        try:
            if not self.product_master_path.exists():
                return []

            df = pd.read_excel(self.product_master_path, engine="openpyxl")
            self._sub_programs = (
                df["sub_program"].astype(str).str.zfill(4).unique().tolist()
            )
            return self._sub_programs
        except Exception as e:
            print(f"[SimulationStatusReader] 載入子程式清單失敗: {e}")
            return []

    def _check_process_status(self, process_id):
        return check_process_status(process_id)

    def _get_finished_count(self):
        """獲取已完成的子程式數量"""
        try:
            if not self.sim_latest.exists():
                return 0

            # 使用 SimulationDataAppender 的追蹤檔案來獲取更準確的進度
            sub_programs = self._load_sub_programs()
            finished_count = 0

            for sub_program in sub_programs:
                # 檢查追蹤檔案
                tracking_file = self.sim_latest / f"{sub_program}.tracking"
                if tracking_file.exists():
                    try:
                        tracking_data = SafeFileHandler.safe_read_json(tracking_file)
                        if tracking_data and tracking_data.get("is_complete", False):
                            finished_count += 1
                    except:
                        pass
                else:
                    # 回退到檢查 Excel 檔案
                    excel_file = self.sim_latest / f"{sub_program}.xlsx"
                    if excel_file.exists():
                        finished_count += 1

            return finished_count

        except Exception as e:
            print(f"[SimulationStatusReader] 獲取完成數量失敗: {e}")
            return 0

    def _get_current_subprogram(self):
        """獲取當前正在處理的子程式 - 相容舊版本和新版本格式"""
        try:
            if not self.sim_latest.exists():
                return None

            # 新版本：優先檢查 tracking 檔案
            sub_programs = self._load_sub_programs()
            if sub_programs:
                current_subprogram = self._get_current_subprogram_from_tracking(
                    sub_programs
                )
                if current_subprogram:
                    return current_subprogram

            # 檢查是否正在進行 all_simulated 合併
            all_simulated_tracking = self.sim_latest / "all_simulated.tracking"
            if all_simulated_tracking.exists():
                try:
                    tracking_data = SafeFileHandler.safe_read_json(
                        all_simulated_tracking
                    )
                    if tracking_data and not tracking_data.get("is_complete", False):
                        # 如果 all_simulated 的 tracking 檔案存在且狀態未完成，說明正在合併
                        return "all_simulated"
                except Exception as e:
                    print(
                        f"[SimulationStatusReader] 讀取 all_simulated tracking 檔案失敗: {e}"
                    )

            # 舊版本：回退到檢查 temp 目錄
            return self._get_current_subprogram_from_temp_dirs()

        except Exception as e:
            print(f"[SimulationStatusReader] 獲取當前子程式失敗: {e}")
            return None

    def _get_current_subprogram_from_tracking(self, sub_programs):
        """從 tracking 檔案獲取當前子程式（新版本）"""
        try:
            # 找到狀態為 "running" 的子程式
            for sub_program in sub_programs:
                tracking_file = self.sim_latest / f"{sub_program}.tracking"
                if tracking_file.exists():
                    try:
                        tracking_data = SafeFileHandler.safe_read_json(tracking_file)
                        if tracking_data is None:
                            continue

                        status = tracking_data.get("status", "")
                        if status == "running":
                            return sub_program

                    except Exception as e:
                        print(
                            f"[SimulationStatusReader] 讀取 tracking 檔案失敗 {tracking_file}: {e}"
                        )
                        continue

            # 如果沒有找到 running 狀態的，返回 None
            return None

        except Exception as e:
            print(f"[SimulationStatusReader] 從 tracking 檔案獲取當前子程式失敗: {e}")
            return None

    def _get_current_subprogram_from_temp_dirs(self):
        """從 temp 目錄獲取當前子程式（舊版本）"""
        try:
            # 先檢查是否所有子程式都已完成
            sub_programs = self._load_sub_programs()
            if sub_programs:
                # 檢查所有子程式是否都有 tracking 檔案且狀態為 completed
                tracking_files = list(self.sim_latest.glob("*.tracking"))
                tracking_stems = set([f.stem for f in tracking_files])
                valid_sub_programs = set(sub_programs)

                if valid_sub_programs.issubset(tracking_stems):
                    # 進一步檢查 tracking 檔案的狀態
                    all_completed = True
                    for sub_program in valid_sub_programs:
                        tracking_file = self.sim_latest / f"{sub_program}.tracking"
                        try:
                            tracking_data = SafeFileHandler.safe_read_json(
                                tracking_file
                            )
                            if tracking_data is None:
                                all_completed = False
                                break
                            # 如果狀態不是 completed，表示還有未完成的
                            if tracking_data.get("status") != "completed":
                                all_completed = False
                                break
                        except:
                            all_completed = False
                            break

                    if all_completed:
                        # [SimulationStatusReader] 所有子程式都已完成，無當前子程式
                        return None

            # 如果還有未完成的子程式，則檢查 temp 目錄
            subprogram_temp_dirs = list(self.sim_latest.glob("*_temp"))
            if not subprogram_temp_dirs:
                return None

            # 只處理有效的 temp 目錄（在 product_master.xlsx 中記錄的子程式）
            valid_temp_dirs = []
            if sub_programs:
                valid_sub_programs = set(sub_programs)
                for temp_dir in subprogram_temp_dirs:
                    temp_stem = temp_dir.stem.replace("_temp", "")
                    if temp_stem in valid_sub_programs:
                        valid_temp_dirs.append(temp_dir)
            else:
                valid_temp_dirs = subprogram_temp_dirs

            if not valid_temp_dirs:
                return None

            # 找到最新修改的目錄
            latest_temp_dir = max(valid_temp_dirs, key=lambda x: x.stat().st_mtime)
            current_subprogram = latest_temp_dir.stem.replace("_temp", "")

            return current_subprogram

        except Exception as e:
            print(f"[SimulationStatusReader] 從 temp 目錄獲取當前子程式失敗: {e}")
            return None

    def get_status_info(self):
        """
        獲取完整的狀態資訊

        Returns:
            dict: 包含所有狀態資訊的字典
        """
        if self._status_info is not None:
            return self._status_info

        # 載入基本資訊
        process_info = self._load_process_info()
        if process_info is None:
            return None

        sub_programs = self._load_sub_programs()
        total_count = len(sub_programs)
        finished_count = self._get_finished_count()

        # 基本資訊
        precision = (
            f'{10 ** (3 - process_info.get("precision"))}mm'
            if process_info.get("precision")
            else "未知"
        )
        username = process_info.get("username", "未知用戶")
        start_time = process_info.get("start_time", "未知")
        start_timestamp = process_info.get("start_timestamp")
        finish_time = process_info.get("finish_time", "未完成")
        finish_timestamp = process_info.get("finish_timestamp")
        finish_flag = process_info.get("finish_flag", False)
        process_id = str(process_info.get("process_id", "未知"))
        cmd = process_info.get("cmd", "未知")

        # 計算耗時
        if finish_flag and finish_timestamp:
            time_diff = finish_timestamp - start_timestamp
            elapsed_time = round(time_diff / 3600, 1)
        elif start_timestamp:
            time_diff = datetime.now().timestamp() - start_timestamp
            elapsed_time = round(time_diff / 3600, 1)
        else:
            elapsed_time = "未知"

        # 檢查進程狀態
        process_status = self._check_process_status(process_id)

        # 判斷任務狀態和進度
        current_subprogram = self._get_current_subprogram()
        if not self.sim_latest.exists():
            status = "未啟動"
            progress = 0.0
        elif finish_flag and finished_count == total_count and total_count > 0:
            status = "已完成"
            progress = 1.0
        else:
            if current_subprogram:
                if process_status != "已終止":
                    status = f"{process_status}，當前運行到{current_subprogram}"
                else:
                    status = f"⚠️異常終止，運行到{current_subprogram}"
                progress = finished_count / total_count if total_count > 0 else 0
            else:
                if process_status != "已終止":
                    status = f"{process_status}，當前運行到準備毛坯矩陣"
                else:
                    status = f"⚠️異常終止，運行到準備毛坯矩陣階段"
                progress = 0.0

        # 組裝狀態資訊
        self._status_info = {
            "department": self.department,
            "username": username,
            "clamping": self.clamping,
            "precision": precision,
            "start_time": start_time,
            "finish_time": finish_time,
            "elapsed_time": elapsed_time,
            "total_count": total_count,
            "finished_count": finished_count,
            "task_progress": f"{finished_count}/{total_count}",
            "progress_percentage": progress * 100,
            "task_status": status,
            "process_id": process_id,
            "process_status": process_status,
            "cmd": cmd,
            "finish_flag": finish_flag,
            "current_subprogram": current_subprogram,
            "simulation_path": str(self.simulation_path),
            "sim_latest_path": str(self.sim_latest),
        }

        return self._status_info

    def is_task_valid(self):
        """檢查任務是否有效（是否有 process_info.json）"""
        return self._load_process_info() is not None

    def is_completed(self):
        """檢查任務是否已完成"""
        status_info = self.get_status_info()
        return status_info and status_info.get("finish_flag", False)

    def is_running(self):
        """檢查任務是否正在運行"""
        status_info = self.get_status_info()
        return status_info and status_info.get("process_status") == "運行中"

    def is_terminated(self):
        """檢查任務是否異常終止"""
        status_info = self.get_status_info()
        return status_info and "異常終止" in status_info.get("task_status", "")

    def get_resume_checkpoint(self):
        """獲取可恢復的檢查點"""
        if not self.is_terminated():
            return None

        status_info = self.get_status_info()
        if status_info and status_info.get("current_subprogram"):
            return status_info["current_subprogram"]
        return None

    def refresh(self):
        """刷新狀態資訊（清除快取）"""
        self._process_info = None
        self._sub_programs = None
        self._status_info = None


def post_process(img, target_value):
    """
    后处理，如果某个点为target_value，那么将该xy位置的所有Z轴位置大于z的点都设为target_value。

    参数:
    img: numpy.ndarray
        三维图像数据。
    target_value: int or float
        目标像素值。

    返回:
    numpy.ndarray
        更新后的图像数据。
    """
    zero_positions = np.argwhere(img == target_value)

    # 标记每个 (x, y) 的最小 z 值
    min_z_per_xy = {}

    for x_, y_, z_ in zero_positions:
        if (x_, y_) not in min_z_per_xy:
            min_z_per_xy[(x_, y_)] = z_

    # 根据最小 z 值更新图像
    for (x_, y_), min_z in min_z_per_xy.items():
        img[x_, y_, min_z + 1 :] = target_value

    return img


def physical_to_pixel(physical, origin, size, precision=4):
    """
    將物理座標轉換為像素座標。

    參數:
    physical: list
        包含物理座標 (x, y, z) 的列表
    origin: tuple
        圖像中心的像素座標。
    size: tuple
        圖像的大小 (length, width, height)
    precision: int, optional
        轉換精度，預設值為 4

    返回:
    list
        轉換後的像素座標 (x, y, z)
    """
    pixel_x = physical[0] * 10 ** (precision - 3) + origin[0]
    pixel_y = size[1] - (physical[1] * 10 ** (precision - 3) + origin[1])
    if len(physical) > 2:
        pixel_z = physical[2] * 10 ** (precision - 3) + origin[2]
    return np.array([int(x) for x in [pixel_x, pixel_y, pixel_z]]).astype(int)


def pixel_to_physical(pixels, origin, size, precision=4):
    """
    將像素座標轉換為物理座標。

    參數:
    pixels: list
        包含像素座標 (x, y, z) 的列表
    origin: tuple
        圖像中心的物理座標。
    size: tuple
        圖像的大小 (length, width, height)，假設對應整個物理範圍為 -size/2 至 size/2
    precision: int, optional
        轉換精度，預設值為 4

    返回:
    list
        轉換後的物理座標 (x, y, z)
    """
    scale = 10 ** (precision - 3)
    physical = []
    for i in range(len(pixels)):
        if i == 1:  # 對於 y 軸需要反轉
            physical_value = round((size[1] - pixels[i] - origin[i]) / scale)
        else:
            physical_value = round((pixels[i] - origin[i]) / scale)
        physical.append(physical_value)
    return physical


def display_recent_cutting(img, deepest_layer, row):
    """
    显示给定图像在指定深度层的切割效果。

    参数:
    img: numpy.ndarray
        包含三维图像数据的数组。
    deepest_layer: int
        要显示的图像深度层。
    row: dict
        包含切割路径信息的字典，包括 'sub_program', 'row_id', 和 'src'。

    显示:
    matplotlib.figure.Figure
        在指定深度层显示图像并在标题中显示切割信息。
    """
    out_img = img[:, :, deepest_layer]
    plt.imshow(out_img)
    plt.title(
        f"{row['sub_program']} - Cutting No.{row['row_id']}, {row['src']}, z={deepest_layer}"
    )
    plt.show()


@log_sparse(stack_depth=4)
def update_image(img, mask, mask_range, step_color, binary=False, **kwargs):
    """
    更新圖片，將刀具路徑設為 path_color，將命中的材料設為 cut_color。

    參數:
    img: numpy.ndarray
        四維圖像數據 [H,W,D,C]，C=1(binary) 或 C=3(RGB)。
    mask: numpy.ndarray
        切割掩碼，用於標識需要更新的像素區域 [H,W,D]。
    mask_range: tuple
        掩碼在圖像中的範圍，包含六個整數 (x_start, x_end, y_start, y_end, z_start, z_end)。
    cut_color: tuple
        命中的材料需要設置的顏色值。
    path_color: tuple
        刀具路徑需要設置的顏色值。
    binary: bool
        是否為二進制圖像模式。預設為 False。

    返回:
    tuple
        更新後的圖像數據和最深的切割層。
    """
    # 添加圖像格式斷言
    if binary:
        assert len(img.shape) == 4, "img should be 4D ([H,W,D,1] for binary)"
        assert img.shape[3] == 1, "img should be 4D ([H,W,D,1] for binary)"
    else:
        assert len(img.shape) == 4, "img should be 4D ([H,W,D,3] for RGB)"
        assert img.shape[3] == 3, "img should be 4D ([H,W,D,3] for RGB)"
    assert len(mask.shape) == 3, "mask should be 3D ([H,W,D])"

    use_gpu = kwargs.get("use_gpu", False)

    if use_gpu:
        import cupy

        img_patch = img[
            mask_range[0] : mask_range[1],
            mask_range[2] : mask_range[3],
            mask_range[4] : mask_range[5],
        ].copy()

        cut_mask = mask == CUTTING_MASK_COLOR  # == 1

        if binary:
            condition = (img_patch[..., 0] == cupy.array(EMPTY_MASK_COLOR)) & cut_mask
            img_patch[condition] = cupy.array(CUTTING_MASK_COLOR)
            condition = (
                img_patch[..., 0] == cupy.array(MATERIAL_MASK_COLOR)
            ) & cut_mask
            img_patch[condition] = cupy.array(PATH_MASK_COLOR)
        else:
            condition = (
                cupy.all(img_patch == cupy.array(EMPTY_COLOR), axis=-1)
            ) & cut_mask
            img_patch[condition] = cupy.array(CUTTING_COLOR)
            condition = (
                cupy.all(img_patch == cupy.array(MATERIAL_COLOR), axis=-1)
            ) & cut_mask
            img_patch[condition] = cupy.array(step_color)

        pixels = cupy.where(condition)
        try:
            deepest_layer = pixels[2].min() + mask_range[4]
        except:
            deepest_layer = img.shape[2] - 1
        img[
            mask_range[0] : mask_range[1],
            mask_range[2] : mask_range[3],
            mask_range[4] : mask_range[5],
        ] = img_patch
        return img, deepest_layer

    # CPU version
    with get_smart_tracer().log_event("copy image_out"):
        # 只在必要時創建副本
        if img.flags.writeable:
            image_out = img  # 如果原始陣列可寫，直接使用
        else:
            image_out = img.copy()  # 否則創建副本

    with get_smart_tracer().log_event("copy image by mask"):
        img_patch = img[
            mask_range[0] : mask_range[1],
            mask_range[2] : mask_range[3],
            mask_range[4] : mask_range[5],
        ].copy()

    with get_smart_tracer().log_event("cal condition"):
        if binary:
            # Binary 模式：使用單通道比較
            condition = (img_patch[..., 0] == EMPTY_MASK_COLOR) & (
                mask == PATH_MASK_COLOR
            )
            img_patch[condition] = CUTTING_MASK_COLOR
            condition = (img_patch[..., 0] == MATERIAL_MASK_COLOR) & (
                mask == CUTTING_MASK_COLOR
            )
        else:
            # RGB 模式：使用多通道比較
            condition = (np.all(img_patch == EMPTY_COLOR, axis=-1)) & (
                mask == CUTTING_MASK_COLOR
            )
            img_patch[condition] = CUTTING_COLOR
            condition = (np.all(img_patch == MATERIAL_COLOR, axis=-1)) & (
                mask == CUTTING_MASK_COLOR
            )

    with get_smart_tracer().log_event("assign cut_color"):
        if binary:
            img_patch[condition] = PATH_MASK_COLOR
        else:
            img_patch[condition] = step_color
        pixels = np.where(condition)

    with get_smart_tracer().log_event("assign image by mask"):
        try:
            deepest_layer = pixels[2].min() + mask_range[4]
        except:
            deepest_layer = img.shape[2] - 1

        image_out[
            mask_range[0] : mask_range[1],
            mask_range[2] : mask_range[3],
            mask_range[4] : mask_range[5],
        ] = img_patch

    return image_out, deepest_layer


def save_to_zst(array, output_path, origin=[0, 0, 0]):
    # 压缩原始矩阵
    print(f"[DEBUG] saving_to_zst: {output_path}")
    compressor = zstd.ZstdCompressor()
    print(f"[DEBUG] step 1")
    compressed = compressor.compress(array.tobytes())
    print(f"[DEBUG] step 2")
    output_path = output_path.replace(
        ".zst",
        f"_shape={'_'.join([str(x) for x in array.shape])}_origin={'_'.join([str(int(x)) for x in origin])}.zst",
    )

    # 将压缩后的数据保存到文件
    with open(output_path, "wb") as f:
        f.write(compressed)
    print(f"[DEBUG] step 3")

    return output_path


def load_from_zst(input_path):
    # 读取压缩文件
    with open(input_path, "rb") as f:
        compressed = f.read()

    # 解压缩
    decompressor = zstd.ZstdDecompressor()
    decompressed = decompressor.decompress(compressed)

    # 将解压后的字节数据重新转换为 NumPy 数组
    # sample: 6628_shape=1348_1954_63_3_origin=0_0_-1030.zst
    shape = input_path.split("=")[1].replace("_origin", "")
    origin = input_path.split("=")[-1].replace(".zst", "")
    matrix_shape = [int(float(x)) for x in shape.split("_")]
    matrix_origin = [int(float(x)) for x in origin.split("_")]
    decompressed_matrix = np.frombuffer(decompressed, dtype=np.uint8).reshape(
        matrix_shape
    )

    return decompressed_matrix, matrix_origin


def validate_rotation(axis):
    if axis not in ["X", "Z"]:  # 第四軸標準配置檢查
        raise ValueError("不支援的旋轉軸，第四軸應為X或Z軸")


def validate_virtual_axis(axis):
    virtual_axis_mapping = {0.5: ("Y", "Z"), 1.5: ("A", "C")}  # 虛擬軸對應關係
    if axis in virtual_axis_mapping:
        print(f"虛擬軸{axis}對應物理軸：{virtual_axis_mapping[axis]}")


def process_chunk(chunk, scale, order=1):
    return scipy.ndimage.zoom(chunk.astype(np.float32), zoom=scale, order=order)


def analyze_stl_complexity(stl_path):
    """
    分析STL文件的複雜度，幫助用戶了解轉換速度的原因

    返回:
    dict: 包含三角形數量、文件大小等信息的字典
    """
    import gc

    try:
        mesh = trimesh.load_mesh(stl_path)
        file_size = os.path.getsize(stl_path) / (1024 * 1024)  # MB

        info = {
            "file_size_mb": round(file_size, 2),
            "triangle_count": len(mesh.faces),
            "vertex_count": len(mesh.vertices),
            "bounds": mesh.bounds,
            "volume": mesh.volume if mesh.is_volume else "N/A",
        }

        # 立即釋放網格內存
        del mesh
        gc.collect()

        # 預估轉換時間複雜度
        complexity = "低"
        if info["triangle_count"] > 100000:
            complexity = "極高"
        elif info["triangle_count"] > 50000:
            complexity = "高"
        elif info["triangle_count"] > 10000:
            complexity = "中"

        info["complexity"] = complexity

        print(f"[STL Analysis] 文件: {stl_path}")
        print(f"[STL Analysis] 大小: {info['file_size_mb']} MB")
        print(f"[STL Analysis] 三角形數量: {info['triangle_count']:,}")
        print(f"[STL Analysis] 頂點數量: {info['vertex_count']:,}")
        print(f"[STL Analysis] 複雜度: {complexity}")

        if complexity in ["高", "極高"]:
            print(f"[STL Analysis] 建議: 考慮增加 resolution 參數至 2.0-3.0 以加速轉換")

        return info

    except Exception as e:
        print(f"[STL Analysis] 分析失敗: {e}")
        return None


def convert_stl_to_numpy(stl_path, resolution=1.0, precision=4, binary=True):
    """
    @Jue, convert_stl_to_numpy

    將STL文件轉換成numpy矩陣 - 使用分塊策略避免記憶體爆炸

    參數:
    stl_path: str
        STL文件的路徑
    resolution: float
        轉換解析度，預設值為1.0。增加此值(例如2.0, 3.0)可顯著加速轉換，但會降低精度
    precision: int
        數值精度，預設值為4。降低此值可減少內存使用和處理時間
    binary: bool
        是否返回二進制體素矩陣，預設值為True
        - True: 返回3維布林陣列，節省記憶體
        - False: 返回4維RGB彩色陣列，用於視覺化

    返回:
    numpy.ndarray
        轉換後的體素矩陣（binary=True時為布林體素，binary=False時為RGB體素）

    性能優化建議：
    - 如果轉換太慢，建議先檢查STL文件的三角形數量
    - 使用分塊策略避免一次性分配大量記憶體
    - 使用binary=True可節省約75%的記憶體使用量
    """
    import gc  # 添加垃圾回收模塊

    print(f"[STL Conversion] 開始轉換STL文件: {stl_path}")
    print(f"[STL Conversion] 解析度: {resolution}")
    print(f"[STL Conversion] 數值精度: {precision}")

    start_time = datetime.now()
    scale = 10 ** (precision - 3) * resolution

    # 分析STL文件複雜度
    stl_info = analyze_stl_complexity(stl_path)

    # Step 1: 加載網格並立即開始體素化
    my_mesh = trimesh.load_mesh(stl_path)
    print(f"[STL Conversion] Step 1 Load to Trimesh: {datetime.now() - start_time}s")

    start_time = datetime.now()
    my_voxelized = my_mesh.voxelized(
        pitch=resolution, method="subdivide", max_iter=None
    )

    # 立即釋放原始網格內存
    del my_mesh
    gc.collect()

    my_voxelized = my_voxelized.fill()
    print(
        f"[STL Conversion] Step 2 Transform to Voxels: {datetime.now() - start_time}s"
    )

    # Step 2: 轉換體素矩陣並立即釋放體素化對象
    voxels_matrix = np.array(my_voxelized.matrix)

    # 立即釋放體素化對象
    del my_voxelized
    gc.collect()

    voxels_matrix = np.transpose(voxels_matrix, (1, 0, 2))
    print(f"[STL Conversion] Voxels Matrix Shape: {voxels_matrix.shape}")

    # 如果解析度為1，則不進行分塊
    if resolution == 10 ** (3 - precision):
        if binary:
            return voxels_matrix
        else:
            voxels_matrix = np.where(voxels_matrix >= 0.5, MATERIAL_COLOR, EMPTY_COLOR)
            return voxels_matrix

    # STEP 3 & 4 合併處理
    start_time = datetime.now()

    # 三維分塊策略，避免內存溢出
    max_chunk_size = int(os.getenv("STL_MAX_CHUNK_SIZE", "32"))  # 減小塊大小避免OOM
    x_chunks = max(1, voxels_matrix.shape[0] // max_chunk_size)
    y_chunks = max(1, voxels_matrix.shape[1] // max_chunk_size)
    z_chunks = max(1, voxels_matrix.shape[2] // max_chunk_size)

    print(
        f"[STL Conversion] 使用三維分塊: {x_chunks}×{y_chunks}×{z_chunks} = {x_chunks*y_chunks*z_chunks}個塊"
    )

    # 計算每個塊在各維度的大小
    x_chunk_size = (voxels_matrix.shape[0] + x_chunks - 1) // x_chunks
    y_chunk_size = (voxels_matrix.shape[1] + y_chunks - 1) // y_chunks
    z_chunk_size = (voxels_matrix.shape[2] + z_chunks - 1) // z_chunks

    print(f"[STL Conversion] 每個塊尺寸: {x_chunk_size}×{y_chunk_size}×{z_chunk_size}")

    # 計算最終矩陣大小並初始化
    if binary:
        final_shape = (
            int(voxels_matrix.shape[0] * scale),
            int(voxels_matrix.shape[1] * scale),
            int(voxels_matrix.shape[2] * scale),
        )
    else:
        final_shape = (
            int(voxels_matrix.shape[0] * scale),
            int(voxels_matrix.shape[1] * scale),
            int(voxels_matrix.shape[2] * scale),
            3,  # RGB 通道
        )

    # 檢查內存需求
    estimated_memory_gb = (final_shape[0] * final_shape[1] * final_shape[2]) / (1024**3)
    print(
        f"[STL Conversion] 預估最終矩陣內存需求: {estimated_memory_gb*3:.2f} GB for RGB图像 and {estimated_memory_gb:.2f} GB for binary图像"
    )

    if estimated_memory_gb > 50.0:  # 如果超過50GB，給出警告
        print(
            f"[STL Conversion] 警告: 內存需求過大，建議增加resolution或減少precision參數"
        )

    print(f"[STL Conversion] 最終矩陣創建完成: {final_shape}")

    # 初始化最終體素矩陣
    if binary:
        final_voxels = np.full(final_shape, 0, dtype=np.uint8)
    else:
        final_voxels = np.full(final_shape, EMPTY_COLOR, dtype=np.uint8)

    # 處理每個三維塊
    total_chunks = x_chunks * y_chunks * z_chunks
    processed_count = 0

    with tqdm(total=total_chunks, desc="Processing 3D chunks") as pbar:
        for xi in range(x_chunks):
            for yi in range(y_chunks):
                for zi in range(z_chunks):
                    # 計算當前塊的邊界
                    x_start = xi * x_chunk_size
                    x_end = min((xi + 1) * x_chunk_size, voxels_matrix.shape[0])
                    y_start = yi * y_chunk_size
                    y_end = min((yi + 1) * y_chunk_size, voxels_matrix.shape[1])
                    z_start = zi * z_chunk_size
                    z_end = min((zi + 1) * z_chunk_size, voxels_matrix.shape[2])

                    # 提取當前塊
                    chunk = voxels_matrix[x_start:x_end, y_start:y_end, z_start:z_end]

                    # 處理當前塊
                    high_precision_chunk = scipy.ndimage.zoom(
                        chunk.astype(np.float32), zoom=scale, order=0
                    )

                    # 立即釋放塊內存
                    del chunk

                    # 計算在最終數組中的位置
                    final_x_start = int(x_start * scale)
                    final_x_end = int(x_end * scale)
                    final_y_start = int(y_start * scale)
                    final_y_end = int(y_end * scale)
                    final_z_start = int(z_start * scale)
                    final_z_end = int(z_end * scale)

                    # RGB模式：返回彩色矩陣
                    if not binary:
                        rgb_chunk = np.where(
                            high_precision_chunk[..., None] >= 0.5,
                            np.array(MATERIAL_COLOR, dtype=np.uint8),
                            np.array(EMPTY_COLOR, dtype=np.uint8),
                        )
                    else:
                        rgb_chunk = high_precision_chunk

                    # 立即釋放high_precision_chunk內存
                    del high_precision_chunk

                    final_voxels[
                        final_x_start:final_x_end,
                        final_y_start:final_y_end,
                        final_z_start:final_z_end,
                    ] = rgb_chunk

                    # 立即釋放rgb_chunk內存
                    del rgb_chunk

                    processed_count += 1
                    pbar.update(1)

                    # 每處理5個塊進行一次垃圾回收
                    if processed_count % 5 == 0:
                        gc.collect()

    # 最終釋放體素矩陣內存
    del voxels_matrix
    gc.collect()

    print(
        f"[STL Conversion] Step 3&4 Combined Processing: {datetime.now() - start_time}s"
    )

    return final_voxels


def convert_stp_to_numpy(
    stp_path,
    resolution=1.0,
    precision=4,
    binary=False,
    linear_deflection=0.5,
    angular_deflection=0.2,
):
    """
    將STP/STEP文件轉換成numpy矩陣

    參數:
    stp_path: str
        STP文件的路徑
    resolution: float
        轉換精度，預設值為1.0
    precision: int
        數值精度，預設值為4
    binary: bool
        是否返回二進制體素矩陣，預設值為False
        - True: 返回3維布林陣列
        - False: 返回4維RGB彩色陣列

    返回:
    numpy.ndarray
        轉換後的體素矩陣（binary=True時為布林體素，binary=False時為RGB體素）
    """
    import tempfile
    import os
    from pathlib import Path

    start_time = datetime.now()
    stp_path = Path(stp_path)

    if not stp_path.exists():
        raise FileNotFoundError(f"STP文件不存在: {stp_path}")

    # 創建臨時STL文件
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp_stl:
        temp_stl_path = tmp_stl.name
        print(f"[STP Conversion] 臨時STL文件: {temp_stl_path}")

    # 将stp_path转换为stl_path
    stl_path = stp_path.with_suffix(".stl")

    # 使用FreeCAD轉換
    try:
        print(f"[STP Conversion FreeCAD] 開始嘗試FreeCAD轉換")
        success = _convert_stp_to_stl_freecad(str(stp_path), temp_stl_path)
        if success:
            print(
                f"[STP Conversion FreeCAD] 使用FreeCAD轉換成功，另存為stl_path: {stl_path}"
            )
            shutil.copy2(temp_stl_path, stl_path)
            return convert_stl_to_numpy(stl_path, resolution, precision, binary)
        else:
            print(f"[STP Conversion FreeCAD] 使用FreeCAD轉換失敗，嘗試cadquery轉換")
    except Exception as e:
        print(f"[STP Conversion FreeCAD] FreeCAD轉換失敗: {e}，嘗試cadquery轉換")

    # 嘗試使用OpenCASCADE cadquery轉換
    try:
        print(f"[STP Conversion cadquery] 開始嘗試cadquery轉換")
        success = _convert_stp_to_stl_cadquery(str(stp_path), temp_stl_path)
        if success:
            print(
                f"[STP Conversion cadquery] 使用cadquery轉換成功，另存為stl_path: {stl_path}"
            )
            shutil.copy2(temp_stl_path, stl_path)
            return convert_stl_to_numpy(stl_path, resolution, precision, binary)
        else:
            print(f"[STP Conversion cadquery] 使用cadquery轉換失敗，嘗試OCC轉換")
    except Exception as e:
        print(f"[STP Conversion cadquery] cadquery轉換失敗: {e}，嘗試OCC轉換")

    # 嘗試使用OCC轉換
    try:
        print(f"[STP Conversion OCC] 開始嘗試OCC轉換")
        success = _convert_stp_to_stl_occ(
            str(stp_path), temp_stl_path, linear_deflection=1.0, angular_deflection=0.8
        )
        if success:
            print(f"[STP Conversion OCC] 使用OCC轉換成功，另存為stl_path: {stl_path}")
            shutil.copy2(temp_stl_path, stl_path)
            return convert_stl_to_numpy(stl_path, resolution, precision, binary)
        else:
            print(f"[STP Conversion OCC] 使用OCC轉換失敗")
    except Exception as e:
        print(f"[STP Conversion OCC] OCC轉換失敗: {e}")

    # 如果都失敗了，拋出異常
    raise RuntimeError("無法轉換STP文件，請確保安裝了FreeCAD或相關的CAD處理庫")


def _convert_stp_to_stl_freecad(stp_path, stl_path):
    """
    使用FreeCAD將STP文件轉換為STL文件
    """
    try:
        import FreeCAD
        import Part
        import Mesh

        # 讀取STP文件
        doc = FreeCAD.newDocument("temp_doc")
        try:
            # 導入STEP文件
            import Import

            Import.insert(stp_path, doc.Name)

            # 獲取所有對象並合併
            objects = doc.Objects
            if not objects:
                raise ValueError(
                    "[STP Conversion FreeCAD] STP文件中沒有找到任何幾何對象"
                )

            # 合併所有形狀
            shapes = []
            for obj in objects:
                if hasattr(obj, "Shape") and obj.Shape:
                    shapes.append(obj.Shape)

            if not shapes:
                raise ValueError("[STP Conversion FreeCAD] STP文件中沒有找到有效的形狀")

            # 如果有多個形狀，進行布爾並集操作
            if len(shapes) == 1:
                combined_shape = shapes[0]
            else:
                combined_shape = shapes[0]
                for shape in shapes[1:]:
                    combined_shape = combined_shape.fuse(shape)

            # 創建網格 - 調整tessellate參數以平衡質量和速度
            # tessellate參數控制網格精細程度，值越大網格越粗糙但速度越快
            tessellate_precision = float(
                os.getenv("FREECAD_TESSELLATE_PRECISION", "0.5")
            )
            print(
                f"[STP Conversion FreeCAD] 使用tessellate精度: {tessellate_precision}"
            )

            mesh = doc.addObject("Mesh::Feature", "Mesh")
            mesh.Mesh = Mesh.Mesh(combined_shape.tessellate(tessellate_precision))

            # 導出STL
            mesh.Mesh.write(stl_path)

            return True

        finally:
            FreeCAD.closeDocument(doc.Name)

    except ImportError:
        print("[STP Conversion FreeCAD] FreeCAD未安裝或無法導入")
        return False

    except Exception as e:
        print(f"[STP Conversion FreeCAD] FreeCAD轉換失敗: {e}")
        return False


def _convert_stp_to_stl_cadquery(stp_path, stl_path):
    """
    使用cadquery將STP文件轉換為STL文件
    """
    try:
        import cadquery as cq

        # 讀取STEP文件
        result = cq.importers.importStep(stp_path)

        # 導出為STL
        if hasattr(result, "objects") and result.objects:
            # 如果有多個對象，合併它們
            combined = result.objects[0]
            for obj in result.objects[1:]:
                combined = combined.union(obj)
            combined.exportStl(stl_path)
        else:
            # 單個對象
            result.exportStl(stl_path)

        return True

    except ImportError:
        print("[STP Conversion cadquery] cadquery未安裝，嘗試直接使用OCC")
        return False

    except Exception as e:
        print(f"[STP Conversion cadquery] cadquery轉換失敗: {e}")
        return False


def _convert_stp_to_stl_occ(
    stp_path, stl_path, linear_deflection=0.5, angular_deflection=0.2
):
    """
    使用OCC將STP文件轉換為STL文件
    """
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.StlAPI import StlAPI_Writer
        from OCP.BRepMesh import BRepMesh_IncrementalMesh

        # 讀取STEP文件
        reader = STEPControl_Reader()
        status = reader.ReadFile(stp_path)

        if status != IFSelect_RetDone:
            raise ValueError("[STP Conversion OCC] 無法讀取STEP文件")

        # 轉換所有實體
        reader.TransferRoots()
        shape = reader.OneShape()

        # 創建網格 - 調整參數以平衡質量和速度
        # 參數說明：
        # 1. shape: 3D形狀對象
        # 2. 0.5: 線性偏差(毫米) - 增加此值可減少三角形數量，加速轉換
        # 3. False: 相對偏差標誌
        # 4. 0.2: 角度偏差(弧度) - 增加此值可減少三角形數量
        # 5. True: 並行執行
        print(
            f"[STP Conversion OCC] 使用網格參數 - 線性偏差: {linear_deflection}, 角度偏差: {angular_deflection}"
        )
        BRepMesh_IncrementalMesh(
            shape, linear_deflection, False, angular_deflection, True
        )

        # 寫入STL
        writer = StlAPI_Writer()
        writer.Write(shape, stl_path)

        return True

    except ImportError as e:
        print(f"[STP Conversion OCC] OpenCASCADE未安裝: {e}")
        return False

    except Exception as e:
        print(f"[STP Conversion OCC] OpenCASCADE轉換失敗: {e}")
        return False


def test_stp_to_numpy_size(stp_path, resolution=1.0, precision=4):
    """
    直接使用OpenCASCADE將STP文件轉換為numpy矩陣
    採用兩階段轉換策略：先粗糙轉換，再精細化upsample

    參數:
    stp_path: str
        STP文件的路徑
    resolution: float
        初始體素化解析度，預設值為1.0 (每個體素代表1mm)
    precision: int
        最終精度，預設值為4 (最終體素邊長為0.1mm)

    返回:
    numpy.ndarray
        轉換後的RGB體素矩陣

    轉換策略:
    1. 階段一：使用resolution進行粗糙體素化 (快速)
    2. 階段二：使用precision進行精細化upsample (精確)

    例如: resolution=1.0, precision=4
    - 初始: 每個體素 = 1mm
    - 最終: 每個體素 = 0.1mm
    - 放大倍數: 10倍 (體積放大1000倍)
    """
    from pathlib import Path

    start_time = datetime.now()
    stp_path = Path(stp_path)

    if not stp_path.exists():
        raise FileNotFoundError(f"STP文件不存在: {stp_path}")

    # 計算最終體素大小
    final_voxel_size = 10 ** (3 - precision)  # precision=4 → 0.1mm
    upsample_factor = resolution / final_voxel_size

    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane

    # ================================
    # 階段一：粗糙體素化 (使用resolution)
    # ================================

    # Step 1: 讀取STEP文件
    step_start = datetime.now()
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(stp_path))

    if status != IFSelect_RetDone:
        raise ValueError("[STP Direct Conversion] 無法讀取STEP文件")

    # 轉換所有實體
    reader.TransferRoots()
    shape = reader.OneShape()

    # Step 2: 計算邊界盒
    step_start = datetime.now()
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)

    x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()

    print(
        f"[STP Direct Conversion] 邊界盒: x({x_min:.2f}, {x_max:.2f}), y({y_min:.2f}, {y_max:.2f}), z({z_min:.2f}, {z_max:.2f})"
    )

    test_image = np.zeros(
        (
            int(x_max - x_min) * 10 ** (precision - 3),
            int(y_max - y_min) * 10 ** (precision - 3),
            int(z_max - z_min) * 10 ** (precision - 3),
            3,
        ),
        dtype=np.uint8,
    )
    test_image_size = test_image.nbytes
    print(
        f"[STP Direct Conversion] RGB三通道（现方案）: {test_image_size} bytes,  即 {test_image_size / 1024**3:.2f} GB"
    )
    del test_image

    test_image = np.zeros(
        (
            int(x_max - x_min) * 10 ** (precision - 3),
            int(y_max - y_min) * 10 ** (precision - 3),
            int(z_max - z_min) * 10 ** (precision - 3),
        ),
        dtype=np.bool_,
    )
    test_image_size = test_image.nbytes
    print(
        f"[STP Direct Conversion] BOOL測試圖像內存估計: {test_image_size} bytes,  即 {test_image_size / 1024**3:.2f} GB"
    )
    del test_image

    test_image = np.zeros(
        (
            int(x_max - x_min) * 10 ** (precision - 3),
            int(y_max - y_min) * 10 ** (precision - 3),
            int(z_max - z_min) * 10 ** (precision - 3),
        ),
        dtype=np.uint8,
    )
    test_image_size = test_image.nbytes
    print(
        f"[STP Direct Conversion] UINT8測試圖像內存估計: {test_image_size} bytes,  即 {test_image_size / 1024**3:.2f} GB"
    )
    del test_image
    print("-" * 30)


def post_fill_image(image, seed_points):
    """
    从stl转化为的numpy array有可能是只有轮廓，没有填充的图形，请你根据图形的闭合属性，帮我填充

    Args:
        image: 待填充的图像，numpy array, shape (H, W, 3) 用於2D圖像
        seed_points: 种子点列表,与种子点相连通的空白像素将被填充为材料颜色

    Returns:
        numpy array, 與輸入相同形狀，填充后的图像
    """
    from collections import deque
    from scipy import ndimage

    # 創建圖像的副本以避免修改原始圖像
    filled_image = image.copy()

    # 將材料顏色和空白顏色轉換為numpy數組以便比較
    material_color = np.array(MATERIAL_COLOR, dtype=np.uint8)
    empty_color = np.array(EMPTY_COLOR, dtype=np.uint8)

    # 創建二進制遮罩（材料為True，空白為False）
    material_mask = np.all(filled_image == material_color, axis=-1)

    # 先通过膨胀1像素，再腐蚀1像素，实现区域联通，再进行洪水填充
    # 對二進制遮罩進行形態學操作
    structure_2d = np.ones((3, 3), dtype=bool)
    dilated_mask = ndimage.binary_dilation(material_mask, structure=structure_2d)
    processed_mask = ndimage.binary_erosion(dilated_mask, structure=structure_2d)

    # 將處理後的遮罩應用回RGB圖像
    filled_image[processed_mask] = material_color
    filled_image[~processed_mask] = empty_color

    # 檢查種子點是否為空
    if not seed_points:
        print("[Fill Warning] 沒有提供種子點，返回原始圖像")
        return filled_image

    height, width = image.shape[:2]
    print(f"[Fill Info] 檢測到2D圖像，尺寸: {height}x{width}")

    # 2D洪水填充算法
    def flood_fill_2d_from_seed(start_y, start_x):
        """從指定的2D種子點開始執行洪水填充"""
        # 檢查起始點是否在圖像範圍內
        if start_y < 0 or start_y >= height or start_x < 0 or start_x >= width:
            return 0

        # 檢查起始點是否已經是材料顏色
        if np.array_equal(filled_image[start_y, start_x], material_color):
            return 0

        # 檢查起始點是否為空白顏色
        if not np.array_equal(filled_image[start_y, start_x], empty_color):
            return 0

        # 使用隊列來避免遞歸深度問題
        queue = deque([(start_y, start_x)])
        filled_pixels = 0
        visited = set()

        # 四連通性的偏移量（上、下、左、右）
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            y, x = queue.popleft()

            # 避免重複處理
            if (y, x) in visited:
                continue

            # 檢查當前點是否在範圍內且為空白顏色
            if (
                y < 0
                or y >= height
                or x < 0
                or x >= width
                or not np.array_equal(filled_image[y, x], empty_color)
            ):
                continue

            # 標記為已訪問
            visited.add((y, x))

            # 填充當前像素
            filled_image[y, x] = material_color
            filled_pixels += 1

            # 將鄰接的像素加入隊列
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (
                    (ny, nx) not in visited
                    and 0 <= ny < height
                    and 0 <= nx < width
                    and np.array_equal(filled_image[ny, nx], empty_color)
                ):
                    queue.append((ny, nx))

        return filled_pixels

    # 對每個2D種子點執行填充
    total_filled = 0
    for i, seed_point in enumerate(seed_points):
        if len(seed_point) != 2:
            print(
                f"[Fill Warning] 2D圖像需要2D種子點 (y, x)，跳過種子點 {i+1}: {seed_point}"
            )
            continue

        seed_y, seed_x = seed_point
        print(
            f"[Fill Process] 處理2D種子點 {i+1}/{len(seed_points)}: ({seed_y}, {seed_x})"
        )
        filled_count = flood_fill_2d_from_seed(seed_y, seed_x)
        total_filled += filled_count
        print(f"[Fill Process] 2D種子點 {i+1} 填充了 {filled_count} 個像素")

    print(f"[Fill Complete] 總共填充了 {total_filled} 個像素")

    return filled_image


def post_fill_image_morphological(image, seed_points, slack=1, max_iterations=10):
    """
    使用形態學操作的高級填充方法，解決像素誤差導致的不封閉問題

    算法流程：
    1. 將圖形膨脹slack像素，再腐蝕相同的像素，以封閉間隙
    2. 使用連通區域標記分割圖形
    3. 從種子點開始標記實心區域
    4. 迭代標記：與實心相接觸的區域為空心，只與空心接觸的區域為實心
    5. 將實心區域腐蝕slack個像素，恢復原始尺寸

    Args:
        image: 待填充的圖像，numpy array, shape (H, W, 3)
        seed_points: 種子點列表，格式同post_fill_image
        slack: 膨脹/腐蝕的像素數，預設為2。增加此值可以彌補更大的間隙，但會影響精度
        max_iterations: 最大迭代次數，預設為10，避免無限循環

    Returns:
        numpy array, 與輸入相同形狀，填充後的圖像
    """
    from scipy import ndimage
    from scipy.ndimage import label, binary_dilation, binary_erosion

    # 創建圖像的副本以避免修改原始圖像
    filled_image = image.copy()

    # 將材料顏色和空白顏色轉換為numpy數組以便比較
    material_color = np.array(MATERIAL_COLOR, dtype=np.uint8)
    empty_color = np.array(EMPTY_COLOR, dtype=np.uint8)

    # 檢查種子點是否為空
    if not seed_points:
        print("[Morphological Fill Warning] 沒有提供種子點，返回原始圖像")
        return filled_image

    # 判斷是2D還是3D圖像
    is_3d = len(image.shape) == 4  # (H, W, D, 3)

    if is_3d:
        height, width, depth = image.shape[:3]
        print(f"[Morphological Fill Info] 檢測到3D圖像，尺寸: {height}x{width}x{depth}")
        print(
            f"[Morphological Fill Info] 使用膨脹/腐蝕像素數: {slack}, 最大迭代次數: {max_iterations}"
        )

        # Step 1: 創建二值遮罩（材料為True，空白為False）
        material_mask = np.all(image == material_color, axis=-1)

        # Step 2: 膨脹操作封閉間隙 - 3D結構元素
        structure_3d = ndimage.generate_binary_structure(3, 1)  # 6連通性
        dilated_mask = binary_dilation(
            material_mask, structure=structure_3d, iterations=slack
        )

        print(f"[Morphological Fill] 3D膨脹操作完成，膨脹 {slack} 個像素")

        # Step 3: 使用連通區域標記
        labeled_array, num_features = label(dilated_mask, structure=structure_3d)
        print(f"[Morphological Fill] 發現 {num_features} 個連通區域")

        # Step 4: 從種子點標記實心區域
        solid_regions = set()
        hollow_regions = set()

        for i, seed_point in enumerate(seed_points):
            if len(seed_point) != 3:
                print(
                    f"[Morphological Fill Warning] 3D圖像需要3D種子點 (y, x, z)，跳過種子點 {i+1}: {seed_point}"
                )
                continue

            seed_y, seed_x, seed_z = seed_point

            # 檢查種子點是否在圖像範圍內
            if (
                seed_y < 0
                or seed_y >= height
                or seed_x < 0
                or seed_x >= width
                or seed_z < 0
                or seed_z >= depth
            ):
                print(f"[Morphological Fill Warning] 種子點 {seed_point} 超出圖像範圍")
                continue

            # 找到種子點所在的連通區域
            region_label = labeled_array[seed_y, seed_x, seed_z]
            if region_label > 0:
                solid_regions.add(region_label)
                print(
                    f"[Morphological Fill] 種子點 {i+1} 標記區域 {region_label} 為實心"
                )

        # Step 5: 迭代標記空心和實心區域
        for iteration in range(max_iterations):
            new_hollow = set()
            new_solid = set()

            # 檢查每個未標記的區域
            for region_id in range(1, num_features + 1):
                if region_id in solid_regions or region_id in hollow_regions:
                    continue

                # 獲取當前區域的遮罩
                region_mask = labeled_array == region_id

                # 檢查與已標記區域的鄰接性
                touches_solid = False
                touches_hollow = False

                # 膨脹當前區域以檢查鄰接性
                dilated_region = binary_dilation(
                    region_mask, structure=structure_3d, iterations=1
                )

                for solid_id in solid_regions:
                    solid_mask = labeled_array == solid_id
                    if np.any(dilated_region & solid_mask):
                        touches_solid = True
                        break

                for hollow_id in hollow_regions:
                    hollow_mask = labeled_array == hollow_id
                    if np.any(dilated_region & hollow_mask):
                        touches_hollow = True
                        break

                # 標記規則：與實心相接觸的為空心，只與空心接觸的為實心
                if touches_solid and not touches_hollow:
                    new_hollow.add(region_id)
                elif touches_hollow and not touches_solid:
                    new_solid.add(region_id)

            # 更新區域集合
            hollow_regions.update(new_hollow)
            solid_regions.update(new_solid)

            print(
                f"[Morphological Fill] 迭代 {iteration + 1}: 新增 {len(new_solid)} 個實心區域, {len(new_hollow)} 個空心區域"
            )

            # 如果沒有新的標記，提前終止
            if not new_hollow and not new_solid:
                print(f"[Morphological Fill] 在第 {iteration + 1} 次迭代後收斂")
                break

        # Step 6: 創建最終的實心遮罩
        final_solid_mask = np.zeros_like(material_mask, dtype=bool)
        for solid_id in solid_regions:
            final_solid_mask |= labeled_array == solid_id

        # Step 7: 腐蝕操作恢復原始尺寸
        final_solid_mask = binary_erosion(
            final_solid_mask, structure=structure_3d, iterations=slack
        )

        # Step 8: 應用到最終圖像
        filled_image[final_solid_mask] = material_color

        filled_pixels = np.sum(final_solid_mask) - np.sum(material_mask)
        print(f"[Morphological Fill Complete] 總共填充了 {filled_pixels} 個3D像素")

    else:
        height, width = image.shape[:2]
        print(f"[Morphological Fill Info] 檢測到2D圖像，尺寸: {height}x{width}")
        print(
            f"[Morphological Fill Info] 使用膨脹/腐蝕像素數: {slack}, 最大迭代次數: {max_iterations}"
        )

        # Step 1: 創建二值遮罩（材料為True，空白為False）
        material_mask = np.all(image == material_color, axis=-1)

        # Step 2: 膨脹操作封閉間隙 - 2D結構元素
        structure_2d = ndimage.generate_binary_structure(2, 1)  # 4連通性
        dilated_mask = binary_dilation(
            material_mask, structure=structure_2d, iterations=slack
        )

        print(f"[Morphological Fill] 2D膨脹操作完成，膨脹 {slack} 個像素")

        # Step 3: 使用連通區域標記
        labeled_array, num_features = label(dilated_mask, structure=structure_2d)
        print(f"[Morphological Fill] 發現 {num_features} 個連通區域")

        # Step 4: 從種子點標記實心區域
        solid_regions = set()
        hollow_regions = set()

        for i, seed_point in enumerate(seed_points):
            if len(seed_point) != 2:
                print(
                    f"[Morphological Fill Warning] 2D圖像需要2D種子點 (y, x)，跳過種子點 {i+1}: {seed_point}"
                )
                continue

            seed_y, seed_x = seed_point

            # 檢查種子點是否在圖像範圍內
            if seed_y < 0 or seed_y >= height or seed_x < 0 or seed_x >= width:
                print(f"[Morphological Fill Warning] 種子點 {seed_point} 超出圖像範圍")
                continue

            # 找到種子點所在的連通區域
            region_label = labeled_array[seed_y, seed_x]
            if region_label > 0:
                solid_regions.add(region_label)
                print(
                    f"[Morphological Fill] 種子點 {i+1} 標記區域 {region_label} 為實心"
                )

        # Step 5: 迭代標記空心和實心區域
        for iteration in range(max_iterations):
            new_hollow = set()
            new_solid = set()

            # 檢查每個未標記的區域
            for region_id in range(1, num_features + 1):
                if region_id in solid_regions or region_id in hollow_regions:
                    continue

                # 獲取當前區域的遮罩
                region_mask = labeled_array == region_id

                # 檢查與已標記區域的鄰接性
                touches_solid = False
                touches_hollow = False

                # 膨脹當前區域以檢查鄰接性
                dilated_region = binary_dilation(
                    region_mask, structure=structure_2d, iterations=1
                )

                for solid_id in solid_regions:
                    solid_mask = labeled_array == solid_id
                    if np.any(dilated_region & solid_mask):
                        touches_solid = True
                        break

                for hollow_id in hollow_regions:
                    hollow_mask = labeled_array == hollow_id
                    if np.any(dilated_region & hollow_mask):
                        touches_hollow = True
                        break

                # 標記規則：與實心相接觸的為空心，只與空心接觸的為實心
                if touches_solid and not touches_hollow:
                    new_hollow.add(region_id)
                elif touches_hollow and not touches_solid:
                    new_solid.add(region_id)

            # 更新區域集合
            hollow_regions.update(new_hollow)
            solid_regions.update(new_solid)

            print(
                f"[Morphological Fill] 迭代 {iteration + 1}: 新增 {len(new_solid)} 個實心區域, {len(new_hollow)} 個空心區域"
            )

            # 如果沒有新的標記，提前終止
            if not new_hollow and not new_solid:
                print(f"[Morphological Fill] 在第 {iteration + 1} 次迭代後收斂")
                break

        # Step 6: 創建最終的實心遮罩
        final_solid_mask = np.zeros_like(material_mask, dtype=bool)
        for solid_id in solid_regions:
            final_solid_mask |= labeled_array == solid_id

        # Step 7: 腐蝕操作恢復原始尺寸
        final_solid_mask = binary_erosion(
            final_solid_mask, structure=structure_2d, iterations=slack
        )

        # Step 8: 應用到最終圖像
        filled_image[final_solid_mask] = material_color

        filled_pixels = np.sum(final_solid_mask) - np.sum(material_mask)
        print(f"[Morphological Fill Complete] 總共填充了 {filled_pixels} 個2D像素")

    return filled_image


def _flood_fill_3d_voxel(voxel_matrix, start_z, start_y, start_x):
    """
    3D體素矩陣的洪水填充算法

    Args:
        voxel_matrix: 3D布林矩陣，True表示材料，False表示空白
        start_z, start_y, start_x: 起始種子點座標

    Returns:
        填充後的3D布林矩陣
    """
    from collections import deque

    # 創建副本以避免修改原始矩陣
    filled_matrix = voxel_matrix.copy()

    # 檢查起始點是否有效
    depth, height, width = filled_matrix.shape
    if (
        start_z < 0
        or start_z >= depth
        or start_y < 0
        or start_y >= height
        or start_x < 0
        or start_x >= width
    ):
        print(f"[3D Fill] 起始點 ({start_z}, {start_y}, {start_x}) 超出範圍")
        return filled_matrix

    # 如果起始點已經是材料，直接返回
    if filled_matrix[start_z, start_y, start_x]:
        print(f"[3D Fill] 起始點已經是材料，無需填充")
        return filled_matrix

    # 使用隊列來避免遞歸深度問題
    queue = deque([(start_z, start_y, start_x)])
    visited = set()
    filled_count = 0

    # 6連通性的偏移量（上、下、前、後、左、右）
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    while queue:
        z, y, x = queue.popleft()

        # 避免重複處理
        if (z, y, x) in visited:
            continue

        # 檢查當前點是否在範圍內且為空白
        if (
            z < 0
            or z >= depth
            or y < 0
            or y >= height
            or x < 0
            or x >= width
            or filled_matrix[z, y, x]
        ):  # 已經是材料
            continue

        # 標記為已訪問
        visited.add((z, y, x))

        # 填充當前體素
        filled_matrix[z, y, x] = True
        filled_count += 1

        # 將鄰接的體素加入隊列
        for dz, dy, dx in directions:
            nz, ny, nx = z + dz, y + dy, x + dx
            if (
                (nz, ny, nx) not in visited
                and 0 <= nz < depth
                and 0 <= ny < height
                and 0 <= nx < width
                and not filled_matrix[nz, ny, nx]
            ):  # 不是材料
                queue.append((nz, ny, nx))

    print(f"[3D Fill] 3D洪水填充完成，填充了 {filled_count} 個體素")
    return filled_matrix
