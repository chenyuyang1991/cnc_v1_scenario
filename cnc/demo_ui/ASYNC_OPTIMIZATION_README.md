# Async Optimization Implementation

This document explains the async optimization solution that allows `run_generate_nc_code` to run in the background without blocking the Streamlit UI.

## Overview

The original `process_optimization()` function blocks the UI while waiting for `run_generate_nc_code(conf)` to complete. The new async implementation:

1. **Starts background process**: Uses `nohup` (cross-platform) to run optimization in separate process
2. **Saves results to file**: Results are saved as joblib file in the scenario folder
3. **Polls for completion**: UI automatically checks every 10 seconds for completion
4. **Non-blocking**: Users can navigate away and come back later

## Key Files

### 1. `run_optimization_async.py`
Standalone script that can be executed via `nohup python xxx`. This script:
- Loads configuration from JSON file
- Runs `run_generate_nc_code(conf)`
- Saves results (new_codes, old_codes, out_df) to joblib file
- Updates process info to mark completion

### 2. `processing.py` (Enhanced)
Added async functions:
- `process_optimization_async()`: Async version of original function
- `start_optimization_async()`: Starts background process
- `is_optimization_running()`: Checks if process is active
- `is_optimization_complete()`: Checks if results are ready
- `load_optimization_result()`: Loads joblib results
- `render_optimization_status()`: Handles UI polling and status display

### 3. `async_optimization_example.py`
Example code showing how to integrate async optimization into existing UI

## File Structure

When async optimization runs, these files are created in the scenario folder:
```
../app/{department}/scenario/{scenario_name}/
├── optimization_config.json          # Configuration for background process
├── optimization_result.joblib         # Results (new_codes, old_codes, out_df)
├── optimization_log.txt              # Process logs
├── optimization_process_info.json    # Process metadata and status
└── {scenario_name}.xlsx              # Original scenario config (existing)
```

## Usage Examples

### Replace Original Synchronous Call

**OLD (Blocking):**
```python
if st.button("完成設定"):
    new_codes, old_codes, out_df = process_optimization()
    st.session_state["results"] = (new_codes, old_codes, out_df)
```

**NEW (Non-blocking):**
```python
if st.button("完成設定 (背景優化)"):
    new_codes, old_codes, out_df = process_optimization_async()
    if new_codes is not None:  # Results immediately available
        st.session_state["results"] = (new_codes, old_codes, out_df)

# Always include this to handle polling
render_optimization_status()
```

### Check Status

```python
if is_optimization_running():
    st.info("優化正在後台運行...")
elif is_optimization_complete():
    st.success("優化已完成，可載入結果")
```

### Load Results

```python
results = load_optimization_result()
if results and results['success']:
    new_codes = results['new_codes']
    old_codes = results['old_codes'] 
    out_df = results['out_df']  # Already converted back to DataFrame
```

## Process Flow

1. **User clicks optimize button**
   - `process_optimization_async()` is called
   - Configuration saved to `optimization_config.json`
   - Background process started via subprocess
   - Process info saved to `optimization_process_info.json`

2. **Background processing**
   - `run_optimization_async.py` loads config and runs optimization
   - Results saved to `optimization_result.joblib`
   - Process info updated with completion status

3. **UI polling**
   - `render_optimization_status()` checks every 10 seconds
   - Shows progress, logs, and elapsed time
   - Auto-refreshes page until completion

4. **Results loading**
   - When complete, results automatically loaded
   - Scenario config saved with results
   - Ready for next steps (NC code generation, etc.)

## Benefits

- **Non-blocking UI**: Users can navigate away during optimization
- **Better resource utilization**: Uses separate CPU cores instead of blocking Streamlit
- **Progress tracking**: Real-time status and log viewing
- **Error handling**: Comprehensive error capture and display
- **Cross-platform**: Works on Windows, Linux, and macOS
- **Resumable**: Can resume/reload results after browser refresh

## Integration Notes

1. **Always include polling**: Add `render_optimization_status()` to any page that might have running optimizations
2. **Session state management**: Check for existing results before starting new optimization
3. **Error handling**: Results include success flag and error messages
4. **Log monitoring**: Optimization logs are available for debugging

## Example Integration

See `async_optimization_example.py` for complete working examples of how to integrate this into your existing UI components. 