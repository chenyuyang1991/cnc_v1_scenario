#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 SimulationStatusReader 升級舊版本仿真模擬格式的獨立測試檔案

使用方法：
    python test_legacy_upgrade.py
    python test_legacy_upgrade.py --path ../app/dept1/simulation_master/clamp1
    python test_legacy_upgrade.py --batch
    python test_legacy_upgrade.py --check-only
"""

import sys
import os
import argparse
import json
from pathlib import Path

# 添加專案路徑到 Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # 根據您的專案結構調整
sys.path.insert(0, str(project_root))

try:
    from cnc_genai.demo_ui.simulation.simulation_dashboard import SimulationStatusReader
except ImportError as e:
    print(f"❌ 無法導入 SimulationStatusReader: {e}")
    print("請確認專案路徑設定正確")
    sys.exit(1)


def print_banner():
    """列印測試橫幅"""
    print("=" * 80)
    print("🔄 SimulationStatusReader 舊版本格式升級測試工具")
    print("=" * 80)
    print()


def print_stats(stats):
    """格式化輸出統計結果"""
    print("\n" + "=" * 60)
    print("📊 升級結果統計")
    print("=" * 60)
    print(f"📁 總檢查數: {stats['total_checked']}")
    print(f"🔍 舊版本數: {stats['legacy_detected']}")
    print(f"✅ 升級成功: {stats['upgrade_success']}")
    print(f"❌ 升級失敗: {stats['upgrade_failed']}")
    print(f"✨ 已升級數: {stats['already_upgraded']}")
    print()
    
    if stats['upgrade_details']:
        print("🔍 詳細升級記錄:")
        print("-" * 60)
        for detail in stats['upgrade_details']:
            status_icon = "✅" if detail.get('status') == 'success' else "❌"
            print(f"{status_icon} {detail['path']}")
            
            if detail.get('status') == 'success':
                print(f"   📝 創建檔案數: {detail.get('created_count', 0)}")
                print(f"   🔄 轉換檔案數: {detail.get('converted_count', 0)}")
                if detail.get('current_subprogram'):
                    print(f"   🎯 當前子程式: {detail['current_subprogram']}")
                
                # 顯示有效子程式和忽略的檔案資訊
                if detail.get('valid_sub_programs'):
                    print(f"   📋 有效子程式: {len(detail['valid_sub_programs'])} 個")
                if detail.get('ignored_files'):
                    ignored_excel = detail['ignored_files'].get('excel', [])
                    ignored_temp = detail['ignored_files'].get('temp_dirs', [])
                    if ignored_excel:
                        print(f"   🚫 忽略Excel檔案: {len(ignored_excel)} 個 ({', '.join(ignored_excel[:3])}{'...' if len(ignored_excel) > 3 else ''})")
                    if ignored_temp:
                        print(f"   🚫 忽略Temp目錄: {len(ignored_temp)} 個 ({', '.join(ignored_temp[:3])}{'...' if len(ignored_temp) > 3 else ''})")
                    
            if detail.get('errors'):
                print(f"   ⚠️  錯誤: {len(detail['errors'])} 個")
                for error in detail['errors'][:3]:  # 只顯示前3個錯誤
                    print(f"      - {error}")
                if len(detail['errors']) > 3:
                    print(f"      ... 以及其他 {len(detail['errors']) - 3} 個錯誤")
            print()


def check_single_path(path):
    """檢查單個路徑是否為舊版本格式"""
    print(f"🔍 檢查路徑: {path}")
    
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"❌ 路徑不存在: {path}")
            return False
            
        is_legacy, legacy_info = SimulationStatusReader._detect_legacy_format(path_obj)
        
        if is_legacy:
            print(f"✅ 檢測到舊版本格式")
            print(f"   📊 總Excel檔案: {legacy_info.get('all_excel_files_count', 0)} 個")
            print(f"   📊 目標Excel檔案: {legacy_info.get('target_excel_files_count', 0)} 個")
            print(f"   📊 Parquet檔案: {legacy_info.get('parquet_files_count', 0)} 個")
            print(f"   📊 Tracking檔案: {legacy_info.get('tracking_files_count', 0)} 個")
            print(f"   📊 總Temp目錄: {legacy_info.get('all_temp_dirs_count', 0)} 個")
            print(f"   📊 目標Temp目錄: {legacy_info.get('target_temp_dirs_count', 0)} 個")
            print(f"   📋 有效子程式: {len(legacy_info.get('valid_sub_programs', []))} 個")
            
            if legacy_info.get('target_excel_files'):
                print(f"   ✅ 目標Excel: {', '.join(legacy_info['target_excel_files'])}")
            if legacy_info.get('ignored_excel_files'):
                print(f"   🚫 忽略Excel: {', '.join(legacy_info['ignored_excel_files'])}")
            if legacy_info.get('target_temp_dirs'):
                print(f"   ✅ 目標Temp: {', '.join(legacy_info['target_temp_dirs'])}")
            if legacy_info.get('ignored_temp_dirs'):
                print(f"   🚫 忽略Temp: {', '.join(legacy_info['ignored_temp_dirs'])}")
            
            if legacy_info.get('missing_tracking_for_excel'):
                print(f"   🔍 缺少tracking的Excel: {legacy_info['missing_tracking_for_excel']}")
            if legacy_info.get('missing_tracking_for_temp'):
                print(f"   🔍 缺少tracking的Temp: {legacy_info['missing_tracking_for_temp']}")
        else:
            print(f"✨ 已是新版本格式或無效路徑")
            if legacy_info.get('reason'):
                print(f"   📝 原因: {legacy_info['reason']}")
        
        return is_legacy
        
    except Exception as e:
        print(f"❌ 檢查過程發生錯誤: {e}")
        return False


def upgrade_single_path(path):
    """升級單個路徑"""
    print(f"🔄 升級路徑: {path}")
    
    try:
        stats = SimulationStatusReader.upgrade_legacy_simulation_formats([path])
        print_stats(stats)
        return stats['upgrade_success'] > 0
        
    except Exception as e:
        print(f"❌ 升級過程發生錯誤: {e}")
        return False


def batch_upgrade():
    """批量升級所有仿真目錄"""
    print("🔄 開始批量升級所有仿真目錄...")
    
    try:
        stats = SimulationStatusReader.upgrade_legacy_simulation_formats()
        print_stats(stats)
        return stats['upgrade_success'] > 0
        
    except Exception as e:
        print(f"❌ 批量升級過程發生錯誤: {e}")
        return False


def check_only_mode():
    """僅檢查模式，不執行升級"""
    print("🔍 僅檢查模式 - 掃描所有仿真目錄...")
    
    try:
        import glob
        simulation_paths = glob.glob("../app/*/simulation_master/*")
        
        if not simulation_paths:
            print("❌ 沒有找到任何仿真目錄")
            return
        
        print(f"📁 找到 {len(simulation_paths)} 個仿真目錄")
        print()
        
        legacy_count = 0
        for path in simulation_paths:
            is_legacy = check_single_path(path)
            if is_legacy:
                legacy_count += 1
            print("-" * 40)
        
        print(f"\n📊 檢查完成：{legacy_count} 個舊版本目錄，{len(simulation_paths) - legacy_count} 個新版本目錄")
        
    except Exception as e:
        print(f"❌ 檢查過程發生錯誤: {e}")


def save_results(stats, output_file):
    """儲存升級結果到 JSON 檔案"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"💾 升級結果已儲存到: {output_file}")
    except Exception as e:
        print(f"❌ 儲存結果失敗: {e}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="SimulationStatusReader 舊版本格式升級測試工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python test_legacy_upgrade.py                           # 僅檢查格式，不執行升級
  python test_legacy_upgrade.py --path ../app/dept1/simulation_master/clamp1
  python test_legacy_upgrade.py --batch                   # 批量升級
  python test_legacy_upgrade.py --output results.json     # 儲存結果到檔案
        """
    )
    
    parser.add_argument(
        '--path', 
        type=str, 
        help='指定要升級的仿真目錄路徑'
    )
    
    parser.add_argument(
        '--batch', 
        action='store_true', 
        help='批量升級所有仿真目錄'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        help='儲存升級結果的 JSON 檔案路徑'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # 檢查工作目錄
    if not Path("../app").exists():
        print("❌ 找不到 ../app 目錄，請確認當前工作目錄正確")
        print(f"當前工作目錄: {os.getcwd()}")
        sys.exit(1)
    
    success = False
    stats = None
    
    try:
        if args.path:
            # 升級指定路徑
            if not Path(args.path).exists():
                print(f"❌ 指定的路徑不存在: {args.path}")
                sys.exit(1)
            
            print(f"🎯 升級指定路徑: {args.path}")
            success = upgrade_single_path(args.path)
            
        elif args.batch:
            # 批量升級
            success = batch_upgrade()
            
        else:
            # 預設：僅檢查模式
            print("� 預設模式：僅檢查格式，不執行升級")
            check_only_mode()
            success = True
        
        # 儲存結果
        if args.output and stats:
            save_results(stats, args.output)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用戶中斷操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 執行過程發生未預期錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 輸出最終結果
    print("\n" + "=" * 60)
    if success:
        print("🎉 升級測試完成！")
    else:
        print("⚠️  升級測試完成，但可能存在問題")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()