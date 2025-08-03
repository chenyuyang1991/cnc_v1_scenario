from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import yaml
import streamlit as st
import os
import pandas as pd
import numpy as np
from pathlib import Path, PurePath
import re  # 添加正則表達式模塊
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cnc_genai.src.utils import utils
from cnc_genai.src.v1_algo.adjust_feed_rate import run_adjust_feed_rate
import argparse
import sys
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer


class QuantileDecisionTreeRegressor(DecisionTreeRegressor):
    """
    使用指定分位數作為葉子節點預測值的決策樹回歸器
    """

    def __init__(self, quantile=0.75, **kwargs):
        """
        初始化分位數決策樹回歸器

        參數:
            quantile: 用於預測的分位數，默認為0.75（75%分位數）
            **kwargs: 傳遞給DecisionTreeRegressor的其他參數
        """
        super().__init__(**kwargs)
        self.quantile = quantile
        # 存儲每個葉子節點的分位數值
        self.leaf_quantiles = {}

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        訓練決策樹模型並計算每個葉子節點的分位數

        參數:
            X: 訓練特徵
            y: 訓練目標值
            sample_weight: 樣本權重
            check_input: 是否檢查輸入

        返回:
            self: 訓練好的模型
        """
        # 使用原始方法訓練樹
        super().fit(X, y, sample_weight, check_input)

        # 獲取每個樣本所在的葉子節點
        leaf_ids = self.apply(X)

        # 計算每個葉子節點的分位數
        self.leaf_quantiles = {}
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            if np.sum(mask) > 0:  # 確保葉子節點有樣本
                self.leaf_quantiles[leaf_id] = np.quantile(y[mask], self.quantile)

        return self

    def predict(self, X, check_input=True):
        """
        使用葉子節點的分位數作為預測值

        參數:
            X: 特徵數據
            check_input: 是否檢查輸入

        返回:
            y_pred: 預測值
        """
        # 獲取每個樣本所在的葉子節點
        leaf_ids = self.apply(X, check_input=check_input)

        # 使用葉子節點的分位數作為預測值
        y_pred = np.zeros(leaf_ids.shape)
        for i, leaf_id in enumerate(leaf_ids):
            if leaf_id in self.leaf_quantiles:
                y_pred[i] = self.leaf_quantiles[leaf_id]
            else:
                # 如果沒有找到對應的葉子節點，使用最接近的葉子節點
                # 這種情況在實際上應該不會發生，除非模型在fit和predict之間被修改
                y_pred[i] = np.mean(list(self.leaf_quantiles.values()))

        return y_pred


class FeedRateRegressor:
    def __init__(self, config_path="cnc_genai/conf/v1_ml.yaml"):
        """
        初始化進給率預測器

        參數:
            config_path: 配置文件路徑，默認為None
        """
        self.F_model = None
        self.FoS_model = None  # 每转进给模型
        self.config = None
        self.X_train = None
        self.y_train = None
        self.X_eval = None
        self.y_eval = None

        # 如果提供了配置文件路徑，則加載配置
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        """
        加載配置文件

        參數:
            config_path: 配置文件路徑
        """
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
            print(f"成功加載配置文件: {config_path}")
        except Exception as e:
            print(f"加載配置文件時出錯: {e}")
            self.config = None

    def load_data(self, feature_cols=None, target_cols=["F", "FoS"]):
        """
        根據配置文件中的夾具列表加載訓練和評估數據

        參數:
            data_dir: 數據目錄，如果為None則使用默認路徑
            feature_cols: 特徵列表，如果為None則使用預定義的特徵列表
            target_col: 目標列名稱，默認為'F'，可选F_adjusted, F_multiplier
        """
        if self.config is None:
            raise ValueError("請先使用load_config方法加載配置文件")

        # 指定target列
        self.target_cols = target_cols

        # 獲取配置中的夾具列表
        train_clampings = self.config.get("data_processing", {}).get(
            "train_clampings", []
        )
        eval_clampings = self.config.get("data_processing", {}).get(
            "eval_clampings", []
        )

        if not train_clampings:
            raise ValueError("配置文件中未指定訓練夾位")

        # 加載訓練數據
        print("-" * 5, "訓練夾位", train_clampings)
        self.train_data_set = self._load_clampings_data(
            train_clampings, is_training=True
        )

        # 加載評估數據
        print("-" * 5, "驗證夾位", eval_clampings)
        self.eval_data_set = (
            self._load_clampings_data(eval_clampings) if eval_clampings else None
        )

    def _load_clampings_data(
        self,
        clampings,
        is_training=False,
    ):
        """
        加載指定夾具的數據

        參數:
            clampings: 夾具列表
            is_training: 是否為訓練數據，如果是則應用MRR過濾

        返回:
            字典，鍵為夾具名稱，值為對應的數據框
        """
        all_data = {}

        for clamping in clampings:

            # 構建數據文件路徑
            department, clamping_name = PurePath(clamping).parts
            data = load_clamping_ml_input(department, clamping_name)
            data["clamping"] = clamping
            ori_num_rows = len(data)

            # 僅在訓練時應用MRR過濾
            if is_training:
                original_count = len(data)
                data = self.filter_lines_by_mrr(data)
                filtered_count = len(data)
                print(
                    f"MRR過濾: {clamping} 從 {original_count} 行減少到 {filtered_count} 行"
                )
            # to test
            data["F"] = data["F_adjusted"]

            all_data[clamping] = data
            print(
                f"已加載夾位數據: {clamping}, 原始行數: {ori_num_rows}, 過濾後行數: {len(data)}行"
            )

        return all_data

    def train(self, target_col="F", X_train=None, y_train=None, **kwargs):
        """
        訓練決策樹模型預測進給率

        參數:
            X_train: 訓練特徵數據，如果為None則使用已加載的數據
            y_train: 訓練目標值 (F值)，如果為None則使用已加載的數據
            **kwargs: 模型的其他參數，例如:
                      - max_depth: 樹的最大深度
                      - min_samples_split: 分裂內部節點所需的最小樣本數
                      - min_samples_leaf: 葉節點所需的最小樣本數

        返回:
            self
        """

        # 如果未提供訓練數據，使用已加載的數據
        if X_train is None or y_train is None:
            if self.X_train is None or self.y_train is None:
                raise ValueError("未提供訓練數據，且未加載數據")
            X_train = self.X_train
            y_train = self.y_train[target_col]

        # 如果配置文件中有決策樹參數，則使用這些參數
        if self.config and "decision_tree" in self.config:
            dt_config = self.config["decision_tree"]
            print(f"已從配置文件中加載決策樹參數: {dt_config}")

        # 添加用戶自定義參數，這些參數會覆蓋配置文件中的參數
        dt_config.update(kwargs)

        # 應用特徵工程
        X_train_processed = self.features_engineering(X_train, is_training=True)

        # 如果配置文件中指定了特徵工程參數，則應用
        if self.config and "feature_engineering" in self.config:
            feature_eng_config = self.config["feature_engineering"]

            # 特徵選擇
            if feature_eng_config.get("feature_selection", False):

                n_features = feature_eng_config.get("n_selected_features", 10)
                selector = SelectKBest(score_func=f_regression, k=n_features)
                X_train_processed = selector.fit_transform(X_train_processed, y_train)
                self.feature_selector = selector
                print(f"已選擇 {n_features} 個最重要的特徵")

            # 多項式特徵
            if feature_eng_config.get("use_polynomial", False):

                poly_degree = feature_eng_config.get("poly_degree", 2)
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X_train_processed = poly.fit_transform(X_train_processed)
                self.poly_transformer = poly
                print(f"已應用 {poly_degree} 次多項式特徵轉換")

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            # 避免除以零
            mask = y_true != 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        # 轉換為sklearn可用的scorer
        mape_scorer = make_scorer(
            mean_absolute_percentage_error, greater_is_better=False
        )

        # 使用自定義評分函數
        if dt_config.get("criterion") == "mape":
            # 移除criterion參數，因為自定義評分不能通過criterion傳入
            dt_config.pop("criterion", None)
            param_grid = {"ccp_alpha": [0.01, 0.02, 0.05]}
            if target_col == "F":
                # self.F_model = DecisionTreeRegressor(**dt_config)
                # self.F_model = GridSearchCV(
                #     self.F_model, param_grid, scoring=mape_scorer
                # )
                # self.F_model.fit(X_train_processed, y_train)
                # self.F_model = self.F_model.best_estimator_

                self.F_model = QuantileDecisionTreeRegressor(quantile=0.75, **dt_config)
                self.F_model.fit(X_train_processed, y_train)
            elif target_col == "FoS":

                # self.FoS_model = DecisionTreeRegressor(**dt_config)
                # self.FoS_model = GridSearchCV(
                #     self.FoS_model, param_grid, scoring=mape_scorer
                # )
                # self.FoS_model.fit(X_train_processed, y_train)
                # self.FoS_model = self.FoS_model.best_estimator_

                self.FoS_model = QuantileDecisionTreeRegressor(
                    quantile=0.75, **dt_config
                )
                self.FoS_model.fit(X_train_processed, y_train)
        else:
            # 創建並訓練模型
            if target_col == "F":
                # self.F_model = DecisionTreeRegressor(**dt_config)
                self.F_model = QuantileDecisionTreeRegressor(quantile=0.75, **dt_config)
                self.F_model.fit(X_train_processed, y_train)
            elif target_col == "FoS":
                # self.FoS_model = DecisionTreeRegressor(**dt_config)
                self.FoS_model = QuantileDecisionTreeRegressor(
                    quantile=0.75, **dt_config
                )
                self.FoS_model.fit(X_train_processed, y_train)
        print(f"{target_col}模型訓練完成，使用了 {X_train_processed.shape[1]} 個特徵")
        return self

    def predict(self, X: pd.DataFrame, target_col="F"):
        """
        使用訓練好的模型來預測進給率 (F值)

        參數:
            X: 特徵數據

        返回:
            y_pred: 預測的F值
        """

        if target_col == "F":
            model = self.F_model
        elif target_col == "FoS":
            model = self.FoS_model
        else:
            raise ValueError(f"不支持的目標列: {target_col}")

        if model is None:
            raise ValueError(f"{target_col}模型尚未訓練，請先調用train方法")

        # 檢查特徵列表是否與訓練時一致
        if isinstance(X, pd.DataFrame):
            # 檢查是否有分類特徵已經被one-hot編碼
            one_hot_features = []
            if hasattr(self, "categorical_feature_names"):
                one_hot_features = self.categorical_feature_names

            # 檢查是否有未處理的分類特徵
            categorical_cols = X.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

        # 應用特徵工程
        X_processed = self.features_engineering(X, is_training=False)

        # 應用特徵選擇
        if hasattr(self, "feature_selector"):
            X_processed = self.feature_selector.transform(X_processed)

        # 應用多項式特徵
        if hasattr(self, "poly_transformer"):
            X_processed = self.poly_transformer.transform(X_processed)

        return model.predict(X_processed)

    def features_engineering(self, X, is_training=True):
        """
        對特徵進行工程處理，包括：
        1. 對字符類數據進行one-hot編碼
        2. 對數值類數據進行歸一化
        3. 處理異常值（如無窮大、NaN）

        參數:
            X: 輸入特徵數據，可以是DataFrame或numpy數組
            is_training: 是否為訓練階段，如果是則擬合轉換器，否則使用已擬合的轉換器

        返回:
            處理後的特徵數據
        """
        # 如果X是numpy數組，轉換為DataFrame
        if isinstance(X, np.ndarray):
            if self.X_train is not None and hasattr(self.X_train, "columns"):
                X = pd.DataFrame(X, columns=self.X_train.columns)
            else:
                X = pd.DataFrame(X)

        # 複製數據，避免修改原始數據
        X_processed = X.copy()

        # 业务特征工程
        X_processed = self._manipulate_customized(X_processed)

        # 数值型变量的处理
        numeric_cols = X_processed.select_dtypes(include=["int64", "float64"]).columns
        X_processed = self._manipulate_numeric_cols(
            X_processed, numeric_cols, is_training
        )

        # 識別數值列和分類列（在處理異常值後重新獲取）
        categorical_cols = X_processed.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
        X_processed = self._manipulate_categorical_cols(
            X_processed, categorical_cols, is_training
        )

        # 最後檢查：確保沒有無窮大和NaN
        # 只對數值型列進行檢查，避免對非數值類型使用 np.isinf
        numeric_cols = X_processed.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            numeric_values = X_processed[numeric_cols].values
            if np.isinf(numeric_values).any() or np.isnan(numeric_values).any():
                print("警告: 處理後的數據中仍存在無效值，请检查...")

        # 確保訓練集和測試集特征一致
        if is_training:
            self.feature_cols = X_processed.columns.to_list()
        else:
            for col in self.feature_cols:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            X_processed = X_processed[self.feature_cols]

        return X_processed

    def data_preprocess(self, target_col="F"):
        # 設置索引列
        index_cols = [
            "clamping",
            "sub_program",
            "sub_program_key",
            "function",
            "N",
            "real_ct",
            "row_id",
            "src",
        ]
        feature_cols = [
            # "F",  # 预测F_adjusted或F_multiplier时，原始的F是重要特征
            # "product",  # 产品类型
            # "clamping_n",  # 夹具编号
            # "coordinates_abs/rel",
            # "coordinates_sys",
            # "unit",
            "precision_mode",
            "move_code",
            # "move_code_prev",
            ## "G",
            "X",
            "Y",
            "Z",
            # "X_prev",
            # "Y_prev",
            # "Z_prev",
            # "X_prev_prev",
            # "Y_prev_prev",
            # "Z_prev_prev",
            # "X_pixel",
            # "Y_pixel",
            # "Z_pixel",
            # "X_prev_pixel",
            # "Y_prev_pixel",
            # "Z_prev_pixel",
            "S",
            # "S_prev",
            # "I",
            # "J",
            # "K",
            ## "A",
            ## "B",
            # "C",
            "tool_diameter",
            "tool_height",
            ## "cutting_area",
            # "same_start_end_pixel",
            ## "is_valid",
            "hit_area",
            # "is_valid_slack",
            # "hit_area_slack",
            # "ap_max_over_xy",
            # "ap_sum_voxel",
            # "path_area_xy",
            # "hit_area_xy",
            # "ap_avg_over_hit",
            # "ap_avg_over_path",
            # "hit_ratio_xy",
            # "ae_max_over_z",
            # "ae_sum_voxel",
            # "path_area_z",
            # "hit_area_z",
            # "ae_avg_over_hit",
            # "ae_avg_over_path",
            # "hit_ratio_z",
            "radius_pixel",
            # "angle_start_pixel",
            # "angle_end_pixel",
            "arc_pixel",
            "turning_angle",
            "path_length",
            # 'acc_time',
            # 'acc_dist',
            # 'const_time',
            # 'const_dist',
            # 'acc/dec',
            # 'time_physical',
            # 'time_physical_acc',
            # 'time_tool_change',
            # 'time_spindle_acc',
            # 'time_turning',
            "ap_mm",
            "ae_mm",
            # "MRR", # 由F计算得到，预测F时不适用
            # "cutting_speed_vc",
            # "power_pc",
            # "torque_mc",
            ## "is_finishing",
            # "is_finishing_outer",
            # "is_finishing_inner",
            # "continuous_duration",
            # "continuous_num_lines",
            # "is_short_duration",
            "is_curve_start",
            "is_spiral",
            "is_turning",
            "is_pre_turning",
            "is_精鐉_by_name",
            "is_玻璃面_by_name",
            "is_開粗_by_name",
            "is_打孔_by_name",
            "is_T槽_by_name",
            "is_去毛刺_by_name",
            "is_打點_by_name",
            "is_倒角_by_name",
        ]
        not_used_feature_cols = [
            "acc_time",
            "acc_dist",
            "const_time",
            "const_dist",
            "acc/dec",
            "time_physical",
            "time_physical_acc",
            "time_tool_change",
            "time_spindle_acc",
            "time_turning",
            "continuous_duration",
            "continuous_num_lines",
            "is_short_duration",
            "MRR",  # 由F计算得到，预测F时不适用
            "power_pc",
            "torque_mc",
        ]

        # 初始化為所有可能的特徵和索引列
        available_features = set(feature_cols)
        available_index_cols = set(index_cols)

        if self.train_data_set:

            for clamping, data in self.train_data_set.items():
                # 計算所有數據集的特徵列的交集
                current_features = set(
                    [col for col in feature_cols if col in data.columns]
                )
                available_features = available_features.intersection(current_features)

                # 計算所有數據集的索引列的交集
                current_index_cols = set(
                    [col for col in index_cols if col in data.columns]
                )
                available_index_cols = available_index_cols.intersection(
                    current_index_cols
                )

                if len(current_features) < len(feature_cols):
                    missing_features = set(feature_cols) - current_features
                    print(
                        f"警告: 訓練夾位 {clamping} 中以下特徵不存在: {missing_features}"
                    )

                if len(current_index_cols) < len(index_cols):
                    missing_index_cols = set(index_cols) - current_index_cols
                    print(
                        f"警告: 訓練夾位 {clamping} 中以下索引列不存在: {missing_index_cols}"
                    )

            # 轉換為列表
            available_features = list(available_features)
            available_index_cols = list(available_index_cols)

            # 合併訓練數據
            train_data_list = []
            for clamping, data in self.train_data_set.items():
                data = data[
                    available_features
                    + available_index_cols
                    + not_used_feature_cols
                    + self.target_cols
                ]
                # data['F_multiplier'] = data['F_adjusted'] / data['F']
                data.set_index(available_index_cols, inplace=True)
                train_data_list.append(data)

            # 合併所有訓練數據
            self.train_data_df = pd.concat(train_data_list, axis=0)
            for col in self.target_cols:
                self.train_data_df = self.train_data_df[
                    ~pd.isna(self.train_data_df[col])
                ]
                self.train_data_df = self.train_data_df[
                    ~np.isinf(self.train_data_df[col])
                ]

            # 分離特徵和目標變量
            self.X_train = self.train_data_df[available_features]
            self.y_train = self.train_data_df[self.target_cols]

        # 如果沒有有效的訓練數據，則拋出異常
        else:
            print("沒有有效的訓練數據")
            self.X_train = None
            self.y_train = None

        if self.X_train is not None:
            print(f"成功加載訓練數據: {len(self.X_train)}行, {len(feature_cols)}個特徵")

        # 處理評估數據
        if self.eval_data_set:
            eval_data_list = []
            available_features = list(available_features)
            available_index_cols = list(available_index_cols)
            for clamping, data in self.eval_data_set.items():
                # 確保所有特徵都存在於數據中
                missing_features = set(available_features) - set(data.columns)
                if missing_features:
                    print(
                        f"警告: 驗證夾位 {clamping} 中以下特徵不存在: {missing_features}，填充為0"
                    )
                    for col in missing_features:
                        data[col] = 0

                # 選擇需要的列並設置索引
                data = data[
                    available_features
                    + available_index_cols
                    + not_used_feature_cols
                    + self.target_cols
                ]
                # data['F_multiplier'] = data['F_adjusted'] / data['F']
                data.set_index(available_index_cols, inplace=True)

                # 添加到列表
                eval_data_list.append(data)

            # 如果沒有有效的評估數據，則設置為None
            if not eval_data_list:
                print("警告: 沒有有效的評估數據")
                self.X_eval = None
                self.y_eval = None
            else:
                # 合併所有評估數據
                self.eval_data_df = pd.concat(eval_data_list, axis=0)
                for col in self.target_cols:
                    self.eval_data_df = self.eval_data_df[
                        ~pd.isna(self.eval_data_df[col])
                    ]
                    self.eval_data_df = self.eval_data_df[
                        ~np.isinf(self.eval_data_df[col])
                    ]
                # self.eval_data_df = self.eval_data_df[~pd.isna(self.eval_data_df['time_physical'])]

                # 分離特徵和目標變量
                self.X_eval = self.eval_data_df[available_features]
                self.y_eval = self.eval_data_df[self.target_cols]
        else:
            print("沒有有效的評估數據")
            self.X_eval = None
            self.y_eval = None

        if self.X_eval is not None:
            print(f"成功加載評估數據: {len(self.X_eval)}行")

    def _manipulate_numeric_cols(self, X_processed, numeric_cols, is_training):
        for col in numeric_cols:
            # 1. 找到正負inf和正負大數值
            pos_inf_mask = np.isposinf(X_processed[col])
            neg_inf_mask = np.isneginf(X_processed[col])
            large_pos_mask = (X_processed[col] > 1e10) & (
                ~np.isposinf(X_processed[col])
            )
            large_neg_mask = (X_processed[col] < -1e10) & (
                ~np.isneginf(X_processed[col])
            )

            # 2. 找到NaN
            nan_mask = X_processed[col].isna()

            # 3. 排除正負inf、正負大數值、NaN後，進行歸一化
            # 創建一個臨時數據框，排除異常值
            temp_data = X_processed[col].copy()
            temp_data[pos_inf_mask] = np.nan
            temp_data[neg_inf_mask] = np.nan
            temp_data[large_pos_mask] = np.nan
            temp_data[large_neg_mask] = np.nan

            # 處理數值特徵（歸一化）
            if is_training:
                # 訓練階段：擬合並轉換
                self.scaler = StandardScaler()
                # 只使用非異常值進行擬合
                valid_data = temp_data.dropna().values.reshape(-1, 1)
                if len(valid_data) > 0:
                    self.scaler.fit(valid_data)
                else:
                    print(f"警告: 列 '{col}' 中沒有有效數據用於擬合標準化器")
                    self.scaler = StandardScaler()  # 使用默認設置

                # 保存訓練時使用的數值特徵列表
                self.numeric_feature_cols = numeric_cols.tolist()

                # 轉換數據（只轉換非異常值）
                valid_indices = ~(
                    pos_inf_mask
                    | neg_inf_mask
                    | large_pos_mask
                    | large_neg_mask
                    | nan_mask
                )
                if valid_indices.any():
                    X_processed.loc[valid_indices, col] = self.scaler.transform(
                        X_processed.loc[valid_indices, col].values.reshape(-1, 1)
                    )
            else:
                # 預測階段：使用已擬合的轉換器
                if hasattr(self, "scaler"):
                    # 確保只使用在訓練階段已知的列
                    if hasattr(self, "numeric_feature_cols"):
                        available_num_cols = [
                            col
                            for col in self.numeric_feature_cols
                            if col in X_processed.columns
                        ]
                        if col in available_num_cols:
                            # 只轉換非異常值
                            valid_indices = ~(
                                pos_inf_mask
                                | neg_inf_mask
                                | large_pos_mask
                                | large_neg_mask
                                | nan_mask
                            )
                            if valid_indices.any():
                                X_processed.loc[valid_indices, col] = (
                                    self.scaler.transform(
                                        X_processed.loc[
                                            valid_indices, col
                                        ].values.reshape(-1, 1)
                                    )
                                )
                    else:
                        # 如果沒有記錄訓練階段的數值特徵，則使用當前可用的
                        available_num_cols = [
                            col for col in numeric_cols if col in X_processed.columns
                        ]
                        if col in available_num_cols:
                            # 只轉換非異常值
                            valid_indices = ~(
                                pos_inf_mask
                                | neg_inf_mask
                                | large_pos_mask
                                | large_neg_mask
                                | nan_mask
                            )
                            if valid_indices.any():
                                X_processed.loc[valid_indices, col] = (
                                    self.scaler.transform(
                                        X_processed.loc[
                                            valid_indices, col
                                        ].values.reshape(-1, 1)
                                    )
                                )

            # 4. 將正負inf分別填充為+-99
            if pos_inf_mask.any():
                print(f"警告: 列 '{col}' 中發現+inf值，將替換為+99")
                X_processed.loc[pos_inf_mask, col] = 99

            if neg_inf_mask.any():
                print(f"警告: 列 '{col}' 中發現-inf值，將替換為-99")
                X_processed.loc[neg_inf_mask, col] = -99

            # 5. 將正負大數值分別填充為+-9
            if large_pos_mask.any():
                print(f"警告: 列 '{col}' 中發現過大的正值，將替換為+9")
                X_processed.loc[large_pos_mask, col] = 9

            if large_neg_mask.any():
                print(f"警告: 列 '{col}' 中發現過大的負值，將替換為-9")
                X_processed.loc[large_neg_mask, col] = -9

            # 6. 將NaN填充為-2
            if nan_mask.any():
                # print(f"警告: 列 '{col}' 中發現NaN值，將替換為-2")
                X_processed.loc[nan_mask, col] = -2

        return X_processed

    def _manipulate_categorical_cols(self, X_processed, categorical_cols, is_training):
        # 存儲訓練階段的分類特徵列表
        if is_training:
            self.original_categorical_cols = categorical_cols.tolist()

        # 確保預測階段使用與訓練階段相同的分類特徵
        if not is_training and hasattr(self, "original_categorical_cols"):
            # 使用訓練階段的分類特徵列表，無論在測試集是否中也存在
            categorical_cols = self.original_categorical_cols

        # 處理分類特徵（one-hot編碼）
        if len(categorical_cols) > 0:
            if is_training:
                # 訓練階段：擬合並轉換
                from sklearn.preprocessing import OneHotEncoder

                self.one_hot_encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                )
                self.one_hot_encoder.fit(X_processed[categorical_cols])

                # 獲取one-hot編碼後的特徵名稱
                self.categorical_feature_names = (
                    self.one_hot_encoder.get_feature_names_out(categorical_cols)
                )

                # 轉換數據
                one_hot_features = self.one_hot_encoder.transform(
                    X_processed[categorical_cols]
                )
                one_hot_df = pd.DataFrame(
                    one_hot_features,
                    columns=self.categorical_feature_names,
                    index=X_processed.index,
                )

                # 移除原始分類列並添加one-hot編碼列
                X_processed = X_processed.drop(columns=categorical_cols)
                X_processed = pd.concat([X_processed, one_hot_df], axis=1)
            else:
                # 預測階段：使用已擬合的轉換器
                if hasattr(self, "one_hot_encoder") and hasattr(
                    self, "categorical_feature_names"
                ):
                    # 只轉換在訓練階段已知的列
                    available_cat_cols = [
                        col for col in categorical_cols if col in X_processed.columns
                    ]

                    if available_cat_cols:
                        one_hot_features = self.one_hot_encoder.transform(
                            X_processed[available_cat_cols]
                        )
                        one_hot_df = pd.DataFrame(
                            one_hot_features,
                            columns=self.categorical_feature_names,
                            index=X_processed.index,
                        )

                        # 移除原始分類列並添加one-hot編碼列
                        X_processed = X_processed.drop(columns=available_cat_cols)
                        X_processed = pd.concat([X_processed, one_hot_df], axis=1)

                        # 檢查是否缺少訓練階段使用的one-hot特徵
                        for col in self.categorical_feature_names:
                            if col not in X_processed.columns:
                                print(
                                    f"警告: 缺少one-hot特徵 '{col}'，將添加該特徵並設置為0"
                                )
                                X_processed[col] = 0
        return X_processed

    def _manipulate_customized(self, X_processed):

        X_processed["ae_mm_over_tool_diameter"] = (
            X_processed["ae_mm"] / X_processed["tool_diameter"]
        )
        X_processed["ap_mm_over_tool_height"] = (
            X_processed["ap_mm"] / X_processed["tool_height"]
        )
        X_processed["tool_diameter_over_tool_height"] = (
            X_processed["tool_diameter"] / X_processed["tool_height"]
        )
        # X_processed["same_xy"] = (
        #     (X_processed["X"] == X_processed["X_prev"])
        #     & (X_processed["Y"] == X_processed["Y_prev"])
        # ).astype(int)
        # X_processed["same_z"] = (X_processed["Z"] == X_processed["Z_prev"]).astype(int)
        # X_processed["same_xy_pixel"] = (
        #     (X_processed["X_pixel"] == X_processed["X_prev_pixel"])
        #     & (X_processed["Y_pixel"] == X_processed["Y_prev_pixel"])
        # ).astype(int)
        # X_processed["same_z_pixel"] = (
        #     X_processed["Z_pixel"] == X_processed["Z_prev_pixel"]
        # ).astype(int)

        # 刀具：是否有R角、是否T刀、刀头刀柄比例、锥度刀等

        # 工艺：倒角、反倒角、左旋刀、螺旋下刀等

        return X_processed

    def filter_lines_by_mrr(self, df, lower=0.05, upper=0.95):
        """
        按照N分組，對每個組僅保留MRR在指定百分位範圍內的行

        參數:
            df: DataFrame，輸入數據
            lower: float，保留的下界百分位（默認0.05，即5%）
            upper: float，保留的上界百分位（默認0.95，即95%）

        返回:
            過濾後的DataFrame
        """
        # 檢查DataFrame是否包含"MRR"列
        if "MRR" not in df.columns:
            print("警告: DataFrame中沒有'MRR'列，無法進行過濾")
            return df

        # 檢查DataFrame是否包含"N"列
        if "N" not in df.columns:
            print("警告: DataFrame中沒有'N'列，無法按N分組")
            return df

        # 創建一個空DataFrame來存儲過濾後的結果
        filtered_df = pd.DataFrame()

        # 按N分組
        for n_value, group in df.groupby(["N", "clamping", "sub_program"]):
            # 對每個組，計算MRR的下界和上界百分位
            lower_bound = group["MRR"].quantile(lower)
            upper_bound = group["MRR"].quantile(upper)

            # 只保留在範圍內的行
            filtered_group = group[
                (group["MRR"] >= lower_bound) & (group["MRR"] <= upper_bound)
            ]

            # 將過濾後的組添加到結果DataFrame
            filtered_df = pd.concat([filtered_df, filtered_group])

        return filtered_df

    def calc_mape(self, df, lower=0.3, upper=0.7, input_col="F_multiplier_pred_final"):
        """
        按 clamping、sub_program、N 分組，對每組內的 F_pred 計算統計值
        並在原始 DataFrame 中添加三個統計列

        參數:
            df: DataFrame，包含預測的 F_multiplier_pred 和實際值
            lower: float，下界百分位數（默認0.3，即30%）
            upper: float，上界百分位數（默認0.7，即70%）

        返回:
            tuple: (添加了統計值列的原始 DataFrame, 按分組聚合的統計值 DataFrame)
        """

        # 使用 transform 添加 30% 分位數列
        df[f"F_multiplier_pred_{int(lower*100)}p_N"] = df.groupby(
            ["clamping", "sub_program", "N"]
        )[input_col].transform(lambda x: x.quantile(lower))
        df[f"F_multiplier_pred_{int(lower*100)}p_subprogram"] = df.groupby(
            ["clamping", "sub_program"]
        )[input_col].transform(lambda x: x.quantile(lower))

        # 使用 transform 添加 70% 分位數列
        df[f"F_multiplier_pred_{int(upper*100)}p_N"] = df.groupby(
            ["clamping", "sub_program", "N"]
        )[input_col].transform(lambda x: x.quantile(upper))
        df[f"F_multiplier_pred_{int(upper*100)}p_subprogram"] = df.groupby(
            ["clamping", "sub_program"]
        )[input_col].transform(lambda x: x.quantile(upper))

        # 定義安全的加權平均函數，處理權重和為零的情況
        df["time_physical"] = df["time_physical"].fillna(0)

        def weighted_avg(x):
            try:
                weights = df.loc[x.index, "time_physical"]
                # 檢查權重是否全為零或負數
                if weights.sum() <= 0:
                    # 如果權重和為零或負數，使用普通平均值
                    return x.mean()
                else:
                    return np.average(x, weights=weights)
            except Exception as e:
                # 如果出現任何錯誤，使用普通平均值
                print(f"加權平均計算出錯: {e}，使用普通平均值代替")
                return x.mean()

        # 使用 transform 應用到每組
        df["F_multiplier_pred_wavg_N"] = df.groupby(["clamping", "sub_program", "N"])[
            input_col
        ].transform(weighted_avg)
        df["F_multiplier_pred_wavg_subprogram"] = df.groupby(
            ["clamping", "sub_program"]
        )[input_col].transform(weighted_avg)

        # 計算 F_multiplier_pred_N，按 N 提速
        for each in [f"{int(upper*100)}p_N", f"{int(lower*100)}p_N", "wavg_N"]:
            df[f"F_pred_{each}"] = df[f"F_multiplier_pred_{each}"] * df["F"]
            df[f"F_gap_{each}"] = (df[f"F_multiplier_pred_{each}"] - 1) * df["F"]
            df[f"F_gap_{each}_percent"] = df[f"F_gap_{each}"] / df["F"]
            df[f"F_gap_{each}_percent_abs"] = df[f"F_gap_{each}_percent"].abs()

        for each in [
            f"{int(upper*100)}p_subprogram",
            f"{int(lower*100)}p_subprogram",
            "wavg_subprogram",
        ]:
            df[f"F_pred_{each}"] = df[f"F_multiplier_pred_{each}"] * df["F"]
            df[f"F_gap_{each}"] = (df[f"F_multiplier_pred_{each}"] - 1) * df["F"]
            df[f"F_gap_{each}_percent"] = df[f"F_gap_{each}"] / df["F"]
            df[f"F_gap_{each}_percent_abs"] = df[f"F_gap_{each}_percent"].abs()

        # 返回兩個 DataFrame
        return df

    def visualize_decision_tree(self, target_col="F"):
        """
        在Streamlit中可視化決策樹模型，提供多種可視化選項
        """

        if target_col == "F":
            model = self.F_model
        elif target_col == "FoS":
            model = self.FoS_model
        else:
            raise ValueError(f"不支持的目標列: {target_col}")

        if model is None:
            st.error("模型尚未訓練，請先訓練模型")
            return

        st.write("## 決策樹視覺化")

        # 添加視覺化選項
        viz_method = "使用sklearn和graphviz"

        if viz_method == "使用sklearn和graphviz":
            self._visualize_with_graphviz(model)
        elif viz_method == "使用plotly":
            self._visualize_with_plotly(model)
        else:
            self._visualize_with_mermaid(model)

    def _visualize_with_graphviz(self, model):
        """使用sklearn.tree.export_graphviz和graphviz可視化決策樹"""
        try:
            import graphviz
            from sklearn.tree import export_graphviz

            # 獲取特徵名稱
            feature_names = None
            if hasattr(self, "feature_cols"):
                feature_names = self.feature_cols

            # 匯出決策樹圖形
            dot_data = export_graphviz(
                model,
                out_file=None,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                special_characters=True,
                max_depth=12,  # 限制深度以保持可讀性
            )

            # 渲染決策樹
            graph = graphviz.Source(dot_data)
            st.graphviz_chart(dot_data)

            # 顯示特徵重要性
            if hasattr(model, "feature_importances_"):
                st.write("### 特徵重要性")

                if feature_names:
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]

                    # 建立特徵重要性表格
                    importance_df = pd.DataFrame(
                        {
                            "特徵": [
                                feature_names[i] for i in indices[:20]
                            ],  # 顯示前20個重要特徵
                            "重要性": importances[indices[:20]],
                        }
                    )

                    # 選擇繪圖工具
                    plot_method = st.radio(
                        "選擇繪圖工具", ["Matplotlib", "Plotly"], index=0
                    )

                    if plot_method == "Matplotlib":
                        # 使用Matplotlib繪製特徵重要性條形圖
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df["特徵"], importance_df["重要性"])
                        ax.set_xlabel("特徵重要性")
                        ax.set_title("決策樹特徵重要性 (前20個)")
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        # 使用Plotly繪製特徵重要性條形圖
                        try:
                            import plotly.express as px

                            fig = px.bar(
                                importance_df,
                                x="重要性",
                                y="特徵",
                                orientation="h",
                                title="決策樹特徵重要性 (前20個)",
                            )

                            fig.update_layout(
                                xaxis_title="特徵重要性",
                                yaxis_title="特徵",
                                height=600,
                                width=800,
                            )

                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            st.error("請安裝plotly: pip install plotly")

                    # 顯示特徵重要性表格
                    st.dataframe(importance_df)

        except ImportError as e:
            st.error(f"缺少依賴庫: {e}。請執行 'pip install graphviz'")
        except Exception as e:
            st.error(f"可視化時出錯: {e}")

    def _visualize_with_plotly(self, model):
        """使用plotly繪製簡化版決策樹"""
        try:
            import plotly.graph_objects as go
            from sklearn.tree import _tree

            tree = model.tree_
            feature_names = (
                self.feature_cols
                if hasattr(self, "feature_cols")
                else [f"feature_{i}" for i in range(tree.n_features)]
            )

            def get_node_trace(node_name, x, y, marker_size=20, text=None):
                return go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(size=marker_size, color="lightblue"),
                    text=[text if text else node_name],
                    textposition="middle center",
                    hoverinfo="text",
                    name=node_name,
                )

            def get_edge_trace(x0, y0, x1, y1):
                return go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=1, color="gray"),
                    hoverinfo="none",
                )

            # 建立樹的簡化表示(僅顯示前3層)
            max_depth = min(99, model.get_depth())

            def tree_to_plotly(tree, node_id=0, depth=0, x=0, width=2, traces=None):
                if traces is None:
                    traces = []

                if depth > max_depth:
                    return traces

                if tree.feature[node_id] != _tree.TREE_UNDEFINED:
                    feature_name = feature_names[tree.feature[node_id]]
                    threshold = tree.threshold[node_id]
                    node_text = f"{feature_name}<br>{threshold:.4f}<br>樣本數:{tree.n_node_samples[node_id]}"
                    node_trace = get_node_trace(
                        f"node_{node_id}", x, -depth, marker_size=30, text=node_text
                    )
                    traces.append(node_trace)

                    # 計算左右子節點位置
                    left_child = tree.children_left[node_id]
                    right_child = tree.children_right[node_id]

                    if left_child != _tree.TREE_UNDEFINED:
                        left_x = x - width / (2 ** (depth + 1))
                        traces.append(get_edge_trace(x, -depth, left_x, -(depth + 1)))
                        tree_to_plotly(
                            tree, left_child, depth + 1, left_x, width, traces
                        )

                    if right_child != _tree.TREE_UNDEFINED:
                        right_x = x + width / (2 ** (depth + 1))
                        traces.append(get_edge_trace(x, -depth, right_x, -(depth + 1)))
                        tree_to_plotly(
                            tree, right_child, depth + 1, right_x, width, traces
                        )
                else:
                    # 葉節點
                    value = tree.value[node_id][0][0]
                    node_text = f"葉節點<br>值: {value:.4f}<br>樣本數:{tree.n_node_samples[node_id]}"
                    node_trace = get_node_trace(
                        f"leaf_{node_id}", x, -depth, marker_size=20, text=node_text
                    )
                    traces.append(node_trace)

                return traces

            traces = tree_to_plotly(tree)

            # 創建圖形
            fig = go.Figure(data=traces)
            fig.update_layout(
                title="決策樹可視化 (簡化版 - 僅前3層)",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )

            st.plotly_chart(fig, use_container_width=True)

            # 顯示模型信息
            st.write(f"完整決策樹深度: {model.get_depth()}")
            st.write(f"決策樹節點數量: {model.tree_.node_count}")

        except ImportError:
            st.error("缺少依賴庫。請執行 'pip install plotly'")
        except Exception as e:
            st.error(f"使用plotly可視化時出錯: {e}")

    def _visualize_with_mermaid(self, model):
        """使用mermaid.js繪製決策樹"""
        try:
            from sklearn.tree import _tree

            tree = model.tree_
            feature_names = (
                self.feature_cols
                if hasattr(self, "feature_cols")
                else [f"feature_{i}" for i in range(tree.n_features)]
            )

            # 生成mermaid圖表代碼
            mermaid_code = ["flowchart TD"]

            def tree_to_mermaid(
                node_id=0, depth=0, parent_id=None, direction=None, max_depth=3
            ):
                if depth > max_depth:
                    return

                if tree.feature[node_id] != _tree.TREE_UNDEFINED:
                    # 非葉節點
                    feature_name = feature_names[tree.feature[node_id]]
                    threshold = tree.threshold[node_id]
                    node_text = f"{feature_name}<br/>{threshold:.4f}"

                    # 添加節點定義
                    mermaid_code.append(f'    {node_id}["{node_text}"]')

                    # 添加連線
                    if parent_id is not None:
                        if direction == "left":
                            mermaid_code.append(f"    {parent_id} -->|是| {node_id}")
                        else:
                            mermaid_code.append(f"    {parent_id} -->|否| {node_id}")

                    # 處理左右子節點
                    left_child = tree.children_left[node_id]
                    right_child = tree.children_right[node_id]

                    if left_child != _tree.TREE_UNDEFINED:
                        tree_to_mermaid(
                            left_child, depth + 1, node_id, "left", max_depth
                        )

                    if right_child != _tree.TREE_UNDEFINED:
                        tree_to_mermaid(
                            right_child, depth + 1, node_id, "right", max_depth
                        )

                else:
                    # 葉節點
                    value = tree.value[node_id][0][0]
                    mermaid_code.append(f'    {node_id}("預測值: {value:.4f}")')

                    if parent_id is not None:
                        if direction == "left":
                            mermaid_code.append(f"    {parent_id} -->|是| {node_id}")
                        else:
                            mermaid_code.append(f"    {parent_id} -->|否| {node_id}")

            # 生成mermaid代碼
            tree_to_mermaid()

            # 顯示mermaid圖表
            mermaid_chart = "\n".join(mermaid_code)
            st.write("### 決策樹視覺化 (前3層)")
            st.markdown(f"```mermaid\n{mermaid_chart}\n```")

            # 顯示模型信息
            st.write(f"完整決策樹深度: {model.get_depth()}")
            st.write(f"決策樹節點數量: {model.tree_.node_count}")

        except Exception as e:
            st.error(f"使用mermaid可視化時出錯: {e}")

    def save_model(self, path="cnc_genai/ml_models"):
        """
        保存模型到指定路徑

        參數:
            path: 模型保存的目錄路徑，默認為"cnc_genai/ml_models"

        返回:
            bool: 是否成功保存模型
        """
        import joblib
        import os
        from datetime import datetime

        try:
            # 創建目錄（如果不存在）
            os.makedirs(path, exist_ok=True)

            # 獲取當前時間戳作為文件名的一部分
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存狀態
            state_dict = {
                "target_cols": self.target_cols,
                "F_model": self.F_model,
                "FoS_model": self.FoS_model,
                "config": self.config,
                "feature_cols": getattr(self, "feature_cols", None),
                "numeric_feature_cols": getattr(self, "numeric_feature_cols", None),
                "categorical_feature_names": getattr(
                    self, "categorical_feature_names", None
                ),
                "original_categorical_cols": getattr(
                    self, "original_categorical_cols", None
                ),
                "scaler": getattr(self, "scaler", None),
                "one_hot_encoder": getattr(self, "one_hot_encoder", None),
                "feature_selector": getattr(self, "feature_selector", None),
                "poly_transformer": getattr(self, "poly_transformer", None),
                "timestamp": timestamp,
            }

            # 保存模型文件
            model_file = os.path.join(path, f"feed_rate_model_{timestamp}.joblib")
            joblib.dump(state_dict, model_file)

            print(f"模型已成功保存到: {model_file}")
            return True

        except Exception as e:
            print(f"保存模型時出錯: {e}")
            return False

    def load_model(self, model_path=None, directory="cnc_genai/ml_models"):
        """
        從指定路徑加載模型

        參數:
            model_path: 具體的模型文件路徑，如果提供，將直接加載該文件
            directory: 模型所在的目錄，如果model_path為None，則自動選擇目錄中最新的模型

        返回:
            bool: 是否成功加載模型
        """
        import joblib
        import os
        import glob

        try:
            # 確定模型文件路徑
            if model_path is None:
                # 查找目錄中的所有模型文件
                model_files = glob.glob(
                    os.path.join(directory, "feed_rate_model_*.joblib")
                )

                if not model_files:
                    print(f"錯誤: 在目錄 {directory} 中未找到模型文件")
                    return False

                # 按文件修改時間排序，選擇最新的模型
                model_path = max(model_files, key=os.path.getmtime)

            # 檢查文件是否存在
            if not os.path.exists(model_path):
                print(f"錯誤: 模型文件不存在: {model_path}")
                return False

            # 加載模型
            state_dict = joblib.load(model_path)

            # 恢復模型狀態
            self.F_model = state_dict.get("F_model")
            self.FoS_model = state_dict.get("FoS_model")
            self.config = state_dict.get("config")

            # 恢復特徵工程相關屬性
            if "feature_cols" in state_dict and state_dict["feature_cols"] is not None:
                self.feature_cols = state_dict["feature_cols"]

            if "target_cols" in state_dict and state_dict["target_cols"] is not None:
                self.target_cols = state_dict["target_cols"]

            if (
                "numeric_feature_cols" in state_dict
                and state_dict["numeric_feature_cols"] is not None
            ):
                self.numeric_feature_cols = state_dict["numeric_feature_cols"]

            if (
                "categorical_feature_names" in state_dict
                and state_dict["categorical_feature_names"] is not None
            ):
                self.categorical_feature_names = state_dict["categorical_feature_names"]

            if (
                "original_categorical_cols" in state_dict
                and state_dict["original_categorical_cols"] is not None
            ):
                self.original_categorical_cols = state_dict["original_categorical_cols"]

            if "scaler" in state_dict and state_dict["scaler"] is not None:
                self.scaler = state_dict["scaler"]

            if (
                "one_hot_encoder" in state_dict
                and state_dict["one_hot_encoder"] is not None
            ):
                self.one_hot_encoder = state_dict["one_hot_encoder"]

            if (
                "feature_selector" in state_dict
                and state_dict["feature_selector"] is not None
            ):
                self.feature_selector = state_dict["feature_selector"]

            if (
                "poly_transformer" in state_dict
                and state_dict["poly_transformer"] is not None
            ):
                self.poly_transformer = state_dict["poly_transformer"]

            print(f"成功加載模型: {model_path}")
            if "timestamp" in state_dict:
                print(f"模型創建時間: {state_dict['timestamp']}")

            return True

        except Exception as e:
            print(f"加載模型時出錯: {e}")
            import traceback

            traceback.print_exc()
            return False


def load_clamping_ml_input(department, clamping_name):
    excel_path = f"../app/{department}/simulation_master/{clamping_name}/simulation/latest/ml_input.xlsx"
    try:
        data = pd.read_excel(excel_path)

        # 過濾掉S和F為0的行
        data = data[data["S"] > 0]
        data = data[data["F"] > 0]

        # 只保留切割代码
        data = data[data["move_code"].isin(["G01", "G02", "G03", "G81", "G82", "G83"])]

        # 計算每轉進給
        data["FoS"] = data["F"] / data["S"]

        # 新特征
        data["is_精鐉_by_name"] = data["function"].str.contains("精").astype(int)
        data["is_玻璃面_by_name"] = (
            data["function"].str.contains("玻璃面|玻璃面").astype(int)
        )
        data["is_開粗_by_name"] = data["function"].str.contains("粗").astype(int)
        data["is_打孔_by_name"] = data["function"].str.contains("孔|鉆|钻").astype(int)
        data["is_T槽_by_name"] = data["function"].str.contains("T").astype(int)
        data["is_去毛刺_by_name"] = (
            data["function"].str.contains("毛刺|毛刺").astype(int)
        )
        data["is_打點_by_name"] = (
            data["function"].str.contains("打点|打點|刻字|刻字").astype(int)
        )
        data["is_倒角_by_name"] = data["function"].str.contains("倒角|倒角").astype(int)

        product = clamping_name.rsplit("-", 1)[0]  # 取最後一個"-"前的部分
        clamping_n = clamping_name.rsplit("-", 1)[1]  # 取最後一個"-"後的部分

        # 使用正則表達式提取clamping_n中的數字部分(支持整數和小數)
        number_match = re.search(r"(\d+\.\d+|\d+)", clamping_n)
        extracted_number = number_match.group(1) if number_match else ""

        if product in ["X2867", "Lisboa-DH"]:
            data["product"] = "imac"
            data["clamping_n"] = extracted_number  # 存儲提取出的數字
        elif product in ["Eve-Cell", "Diamond-Cell", "Tiga-WF"]:
            data["product"] = "ipad"
            data["clamping_n"] = extracted_number  # 存儲提取出的數字
        else:
            data["product"] = "other"
            data["clamping_n"] = extracted_number  # 存儲提取出的數字

    except:
        # todo 更新訓練數據，要在所有的无缺陷scenario的load_analysis表中，选择F_adjusted最大的那一行
        data.to_excel(excel_path, index=False)
    return data
