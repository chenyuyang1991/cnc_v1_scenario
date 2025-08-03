#!/usr/bin/env python3
"""
STL 到 NumPy 轉換工具

這個命令行工具用於將 STL 檔案轉換為 NumPy 陣列格式。
支援自定義解析度和精度參數。

使用方法:
    python stl2numpy.py input.stl -o output.npy -r 1.0 -p 4

參數說明:
    - input.stl: 輸入的 STL 檔案路徑
    - -o/--output: 輸出的 NumPy 檔案路徑 (預設: input_converted.zst)
    - -r/--resolution: 轉換解析度 (預設: 1.0)
    - -p/--precision: 數值精度 (預設: 4)
    - -b/--binary: 是否使用二進制格式 (預設: True)

支援的輸出格式 (根據副檔名自動判斷):
    - .npy: NumPy 原生格式 (未壓縮)
    - .npz: NumPy 壓縮格式 (自動壓縮)
    - .zst: ZStandard 壓縮格式
"""

import argparse
import numpy as np
import sys
from pathlib import Path
import os

# 導入 convert_stl_to_numpy 函數
from cnc_genai.src.simulation.utils import convert_stl_to_numpy, save_to_zst

# 支援的輸出格式
SUPPORTED_FORMATS = {".npy", ".npz", ".zst"}


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="將 STL 檔案轉換為 NumPy 陣列",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  %(prog)s model.stl                           # 基本轉換 (預設 .zst 格式)
  %(prog)s model.stl -o result.npy             # 輸出為 .npy 格式
  %(prog)s model.stl -o result.npz             # 輸出為壓縮 .npz 格式
  %(prog)s model.stl -o result.zst             # 輸出為 .zst 格式
  %(prog)s model.stl -r 2.0 -p 3               # 自定義解析度和精度
  %(prog)s model.stl --binary                  # 使用二進制模式轉換

支援的輸出格式 (根據副檔名自動判斷):
  .npy - NumPy 原生格式 (未壓縮)
  .npz - NumPy 壓縮格式 (自動壓縮)
  .zst - ZStandard 壓縮格式
        """,
    )

    # 必要參數
    parser.add_argument("input_file", type=str, help="輸入的 STL 檔案路徑")

    # 可選參數
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="輸出檔案路徑",
    )

    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=1.0,
        help="轉換解析度，增加此值可加速轉換但降低精度 (預設: 1.0)",
    )

    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=4,
        help="數值精度，降低此值可減少記憶體使用 (預設: 4)",
    )

    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="原點座標，格式: x y z (預設: 0 0 0，僅對 zst 格式有效)",
    )

    parser.add_argument(
        "-b",
        "--binary",
        type=bool,
        default=True,
        help="是否使用二進制格式 (預設: True)",
    )

    parser.add_argument(
        "--info", action="store_true", help="僅顯示 STL 檔案資訊，不進行轉換"
    )

    return parser.parse_args()


def get_format_from_path(file_path):
    """從檔案路徑獲取格式"""
    return Path(file_path).suffix.lower()


def validate_format(format_ext):
    """驗證格式是否支援"""
    if format_ext not in SUPPORTED_FORMATS:
        return False
    return True


def validate_inputs(args):
    """驗證輸入參數"""
    # 檢查輸入檔案是否存在
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"錯誤: 輸入檔案不存在: {args.input_file}")
        return False

    if not input_path.suffix.lower() in [".stl"]:
        print(f"警告: 輸入檔案可能不是 STL 格式: {args.input_file}")

    # 檢查參數範圍
    if args.resolution <= 0:
        print(f"錯誤: 解析度必須大於 0，當前值: {args.resolution}")
        return False

    if args.precision < 1 or args.precision > 10:
        print(f"錯誤: 精度必須在 1-10 之間，當前值: {args.precision}")
        return False

    # 如果指定了輸出檔案，檢查格式是否支援
    if args.output:
        format_ext = get_format_from_path(args.output)
        if not validate_format(format_ext):
            supported_formats_str = ", ".join(SUPPORTED_FORMATS)
            print(f"錯誤: 不支援的輸出格式 '{format_ext}'")
            print(f"支援的格式: {supported_formats_str}")
            return False

    return True


def generate_output_path(input_file, output_arg=None):
    """產生輸出檔案路徑"""
    if output_arg:
        return Path(output_arg)

    # 預設使用 zst 格式
    input_path = Path(input_file)
    output_name = f"{input_path.stem}_converted.zst"
    return input_path.parent / output_name


def save_numpy_array(array, output_path, origin=None):
    """根據檔案副檔名儲存 NumPy 陣列"""
    output_path = Path(output_path)
    format_ext = get_format_from_path(output_path)

    # 確保輸出目錄存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Save] 正在儲存到: {output_path}")
    print(f"[Save] 檔案格式: {format_ext}")
    print(f"[Save] 陣列形狀: {array.shape}")
    print(f"[Save] 陣列大小: {array.nbytes / 1024**3:.2f} GB")

    if format_ext == ".npy":
        np.save(output_path, array)

    elif format_ext == ".npz":
        # npz 格式默認使用壓縮
        np.savez_compressed(output_path, array=array)

    elif format_ext == ".zst":
        if origin is None:
            origin = [0, 0, 0]
        saved_path = save_to_zst(array, str(output_path), origin=origin)
        print(f"[Save] ZST 檔案已儲存為: {saved_path}")
        return saved_path

    print(f"[Save] 檔案已成功儲存: {output_path}")
    return str(output_path)


def show_stl_info(stl_path):
    """顯示 STL 檔案資訊"""
    try:
        from cnc_genai.src.simulation.utils import analyze_stl_complexity

        print(f"[Info] 分析 STL 檔案: {stl_path}")
        print("=" * 50)

        info = analyze_stl_complexity(stl_path)
        if info:
            print(f"檔案大小: {info['file_size_mb']} MB")
            print(f"三角形數量: {info['triangle_count']:,}")
            print(f"頂點數量: {info['vertex_count']:,}")
            print(f"複雜度: {info['complexity']}")

            bounds = info["bounds"]
            print(f"邊界範圍:")
            print(
                f"  X: {bounds[0][0]:.2f} ~ {bounds[1][0]:.2f} (寬度: {bounds[1][0] - bounds[0][0]:.2f})"
            )
            print(
                f"  Y: {bounds[0][1]:.2f} ~ {bounds[1][1]:.2f} (深度: {bounds[1][1] - bounds[0][1]:.2f})"
            )
            print(
                f"  Z: {bounds[0][2]:.2f} ~ {bounds[1][2]:.2f} (高度: {bounds[1][2] - bounds[0][2]:.2f})"
            )

            volume = info.get("volume", "N/A")
            if volume != "N/A":
                print(f"體積: {volume:.2f}")

        print("=" * 50)

    except Exception as e:
        print(f"無法分析 STL 檔案: {e}")


def main():
    """主函數"""
    args = parse_arguments()

    # 驗證輸入
    if not validate_inputs(args):
        sys.exit(1)

    # 如果只要顯示資訊
    if args.info:
        show_stl_info(args.input_file)
        return

    # 產生輸出路徑
    output_path = generate_output_path(args.input_file, args.output)

    # 獲取輸出格式
    format_ext = get_format_from_path(output_path)

    print("=" * 60)
    print("STL 到 NumPy 轉換工具")
    print("=" * 60)
    print(f"輸入檔案: {args.input_file}")
    print(f"輸出檔案: {output_path}")
    print(f"輸出格式: {format_ext}")
    print(f"輸出通道: {'二進制' if args.binary else 'RGB'}, {args.binary}")
    print(f"解析度: {args.resolution}")
    print(f"精度: {args.precision}")
    if format_ext == ".zst":
        print(f"原點: {args.origin}")
    print("=" * 60)

    try:
        # 執行轉換
        print("[轉換] 開始 STL 到 NumPy 轉換...")
        numpy_array = convert_stl_to_numpy(
            stl_path=args.input_file,
            resolution=args.resolution,
            precision=args.precision,
            binary=args.binary,
        )

        # 顯示轉換結果資訊
        print(f"[轉換] 轉換完成!")
        print(f"[轉換] 結果陣列形狀: {numpy_array.shape}")
        print(f"[轉換] 資料類型: {numpy_array.dtype}")
        print(f"[轉換] 記憶體使用: {numpy_array.nbytes / 1024**3:.2f} GB")

        # 儲存結果
        saved_path = save_numpy_array(
            array=numpy_array,
            output_path=output_path,
            origin=args.origin if format_ext == ".zst" else None,
        )

        print("=" * 60)
        print("✅ 轉換完成!")
        print(f"✅ 輸出檔案: {saved_path}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[中斷] 使用者中斷轉換")
        sys.exit(1)

    except Exception as e:
        print(f"❌ 轉換失敗: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
