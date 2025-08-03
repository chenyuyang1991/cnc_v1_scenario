#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import json
import sys
import os
import argparse
import joblib
import traceback
from datetime import datetime

warnings.filterwarnings("ignore")

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from cnc_genai.src.v1_algo.generate_nc_code import run_generate_nc_code
from cnc_genai.src.utils import utils
import pandas as pd


def run_optimization_async(config_path, output_path):
    """
    Run optimization asynchronously and save results to joblib file
    
    Args:
        config_path (str): Path to the configuration JSON file
        output_path (str): Path where to save the joblib result file
    """
    
    # Get process info path
    scenario_folder = os.path.dirname(output_path)
    process_info_path = os.path.join(scenario_folder, "optimization_process_info.json")
    
    try:
        # Load configuration from JSON file
        with open(config_path, 'r', encoding='utf-8') as f:
            conf = json.load(f)
        
        print(f"[INFO] Starting optimization for scenario: {conf.get('scenario_name', 'unknown')}")
        print(f"[INFO] Config loaded from: {config_path}")
        print(f"[INFO] Results will be saved to: {output_path}")
        
        # Run the optimization
        new_codes, old_codes, out_df = run_generate_nc_code(conf)
        
        # Convert DataFrame to dictionary for serialization
        out_df_dict = out_df.to_dict('records') if out_df is not None else None
        
        # Prepare results
        results = {
            'new_codes': new_codes,
            'old_codes': old_codes,
            'out_df': out_df_dict,
            'success': True,
            'error': None,
            'completion_time': datetime.now().isoformat(),
            'scenario_name': conf.get('scenario_name', 'unknown')
        }
        
        # Save results to joblib file
        joblib.dump(results, output_path)
        
        # Update process info to mark completion
        update_process_info_completion(process_info_path, True)
        
        print(f"[INFO] Optimization completed successfully")
        print(f"[INFO] Results saved to: {output_path}")
        
        return True
        
    except Exception as e:
        error_msg = f"Error during optimization: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        # Save error results
        error_results = {
            'new_codes': None,
            'old_codes': None,
            'out_df': None,
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'completion_time': datetime.now().isoformat(),
            'scenario_name': 'unknown'
        }
        
        try:
            joblib.dump(error_results, output_path)
            print(f"[INFO] Error results saved to: {output_path}")
        except Exception as save_error:
            print(f"[ERROR] Could not save error results: {str(save_error)}")
        
        # Update process info to mark completion (with error)
        update_process_info_completion(process_info_path, False, error_msg)
        
        return False


def update_process_info_completion(process_info_path, success, error_msg=None):
    """Update process info file to mark completion"""
    try:
        if os.path.exists(process_info_path):
            with open(process_info_path, 'r', encoding='utf-8') as f:
                process_info = json.load(f)
            
            process_info.update({
                'finish_flag': True,
                'finish_time': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                'finish_timestamp': datetime.now().timestamp(),
                'success': success,
                'error': error_msg
            })
            
            with open(process_info_path, 'w', encoding='utf-8') as f:
                json.dump(process_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not update process info: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimization asynchronously')
    parser.add_argument('--config_path', required=True, help='Path to configuration JSON file')
    parser.add_argument('--output_path', required=True, help='Path to save joblib result file')
    
    args = parser.parse_args()
    
    success = run_optimization_async(args.config_path, args.output_path)
    sys.exit(0 if success else 1) 