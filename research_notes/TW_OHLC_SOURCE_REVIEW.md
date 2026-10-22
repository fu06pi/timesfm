# TimesFM 3.0 官方原始碼與台股 OHLC 接法審閱

核對日期：2026-10-22。官方來源 google-research/timesfm；fork fu06pi/timesfm。
官方 master 與本專案既有 pin 同為 `e51928e27119cb17bebc005be2696b75e0a9e688`，套件 3.0.2。此次 fork 並沒有取得一個比現用版更新的模型。
對照台股專案 commit `a88eb85`。本分支僅增加閱讀報告與合成核對，不更改官方模型。

## 原始碼地圖

| 來源 | 已核對內容 |
|---|---|
| `src/timesfm3/__init__.py` | 頂層 API 延遲匯出 torch backend；MLX 是另一個 backend |
| `src/timesfm3/torch/model.py` | 可微 forward、RevIN、mask、CPM、關閉梯度的 decode、去趨勢與 stitching |
| `src/timesfm3/torch/timesfm3_forecaster.py` | checkpoint config 同步、輸入整理、缺值內插、future covariate 對齊、quantile 排序及 median |
| `src/timesfm3/torch/evaluator.py` | benchmark 預設、32 variates 上限策略、特徵抽樣與 target 分塊 |
| `src/timesfm3/torch/transformer.py` | 時間方向 causal attention 與非 causal 的跨序列 attention |
| `src/timesfm3/torch/cpm_revin_refine.py` | 預測期 masked patches 的 iterative RevIN refinement |
| `timesfm3-usage/benchmarks/` | FEV/TIME/GIFT 的研究評估入口，不是台股驗收證據 |
| `timesfm-forecasting/examples/finetuning/README.md` | 明確是 TimesFM 2.5 + Transformers/PEFT，非 3.0 trainer |

## 核實結論

1. **3.0 forward 可以反向傳播，但未找到官方 3.0 訓練器。** `forward()` 輸出 logits；`decode()` 有 `@torch.no_grad()`，forecaster 外層有 `torch.inference_mode()`。現有 output-head-only 自訂訓練有合法梯度路徑；不可直接拿 2.5 LoRA 範例稱為官方 3.0 微調。
2. **3.0 分位數是九個，不是 2.5 的十個欄位。** q10–q90，median index 4。舊 skill 的 mean 欄位與索引說明不適用，且 q10–q90 的名義中央覆蓋為 80%，不是 skill 某段所稱 90%。
3. **forward 與預設 decode 不等價。** decode 額外做條件式線性去趨勢及趨勢加回；專案自訂一步 forward 沒有這層。不能只設定 use_linear_detrending=True 就假設 direct forward 已使用。這是管線差異，不是今日失敗的已證實原因，也不代表補上就會提高準確率。
4. **可用未來特徵的遮罩方向正確。** 當日已知夜盤訊號放在預測期第一格，其餘未知未來格保留 masked，日盤 targets 全 masked；不要用重複值補滿 64 格假裝可知未來。官方 decode 可接 context+horizon 的 future covariates。
5. **獨立市場 31 variates；目前聯合設計 62。** 每市場 4 targets+27 features，聯合 8+54。官方 Evaluator >32 時會抽樣 covariates 並分塊 targets；底層 forward 不等於這個策略，不能宣稱 62-channel custom joint 完全沿用官方 benchmark。應先去除兩市場重複共用特徵，再固定縮減規則。
6. **OHLC 聯合區間不是官方功能。** 本專案把 GBUD 各 marginal quantile 同索引還原再取包絡，沒有學到已驗證的聯合依賴。尤其 L 隨 d 增大而降低，g/b/u/d 同升並不代表 OHLC 同一個聯合分位數。這是區間覆蓋不平衡的合理機制，尚需用因果殘差向量校準作實證。
7. **原始日序列缺漏不能交給官方預設內插隱藏。** forecaster 會內插 NaN、全 NaN 轉零；台股資料應在進入模型前明確拒絕或記錄缺漏。
8. **載入方式影響設定。** HF repo/local directory 讀 config.json 並同步；單一權重檔走預設架構。直接 model constructor 與 forecaster builder 的部分預設不同，應持續用 pinned config+weights，不另憑推測重建架構。
9. **授權分開。** 原始碼 Apache-2.0；3.0 自託管權重為 non-commercial/non-production。fork 不會改變權重授權。

## 實際執行的核對

官方 PyTorch 模型、forecaster、transformer、CPM、primitives 與 backward compatibility 六組測試：**51 passed**。使用小型隨機模型／mock weights，未下載或訓練大型模型。

額外 `research_checks/check_tw_adapter.py` 直接匯入台股專案 adapter，使用 synthetic 4-target/27-feature 小模型：

- 關閉去趨勢時，custom 一步 forward 與 official decode 第一步最大差 **0**。
- 開啟去趨勢的合成趨勢序列，兩路徑最大差 **5.411826**（合成單位，非台股點數）。
- 把 masked future target 改成 999，輸出變化 **0**。
- 改變已知 future covariate，輸出確實改變。
- 梯度僅在 output_head.weight/bias，且有限、非零。

結果見 `research_checks/tw_adapter_result.json`。這些證明接法及差異，不是市場預測準確率或 full-checkpoint parity。

## 重現

先使用台股專案 frozen environment；假設兩個倉庫為同層目錄。

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../tw-market-scenario-engine/.venv/bin/python -m pytest src/timesfm3/torch/model_test.py src/timesfm3/torch/timesfm3_forecaster_test.py src/timesfm3/torch/transformer_test.py src/timesfm3/torch/cpm_revin_refine_test.py src/timesfm3/torch/primitives_test.py src/timesfm3/backward_compat_test.py -q
PYTHONPATH=src:../tw-market-scenario-engine/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../tw-market-scenario-engine/.venv/bin/python research_checks/check_tw_adapter.py
```

本機 Git LFS 未安裝，僅此閱讀 clone 停用 LFS filters，以 pointer 保留示例媒體；不影響模型 Python 原始碼。舊的 timesfm-official 工作目錄未覆寫。

## 下一輪優先順序

1. 保存每次預測的可知資訊與標籤；盤前與盤中更新設為不同模型任務。
2. 在既定開發 folds 比較 direct-forward 與含官方去趨勢的可微路徑；先 parity/gradient，再昂貴訓練。不可把訓練後 checkpoint 靜默切到不同 inference pipeline。
3. 對 g/b 與 u/d 分別檢查損失尺度；目前 g+b 收盤損失和 log-extension 損失的梯度貢獻需量測，不能假設 close_weight=2 就足夠。
4. 比較同資訊集的簡單強基準／殘差模型、特徵精簡與 regime 穩定性；不要只加長 context。
5. 區間採預先固定的因果校準並保留全部日期；新 forward holdout 前不宣稱通過。
