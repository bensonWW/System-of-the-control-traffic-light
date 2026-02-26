# 交通預測模型：深度訓練解析與神經網路機理 (`train_model.py`)

這份文件旨在向具備基礎深度學習觀念的開發者，提供 `train_model.py` 中資料工程與模型架構的**深度技術剖析**。我們將探討時間特徵萃取、長尾分佈的對數縮放、解決記憶體溢出 (OOM) 的 Lazy Loading 設計，並以 Tensor 流向與數學公式深入解析 GRU 模型在此 Sequence-to-Sequence (Seq2Seq) 任務中的具體行為。

---

## 1. 資料前處理與特徵工程 (Feature Engineering)

在時間序列預測中，原始資料直接輸入網路的效果通常極差，必須經過特徵工程轉換。

### 1.1 週期性時間特徵嵌入 (Cyclical Time Embedding)
模型的輸入維度並非只有 208 個路段的車流，而是 `num_edges + 2`。這額外的 2 個維度是用來解決時間序列的「連續性假設」問題。

*   **問題背景**：若將單日秒數 $t \in [0, 86400)$ 作為單一特徵直接輸入，模型會認為 23:59:59 與隔日的 00:00:00 在數值上相差極大，破壞了時間流動的連續性。
*   **數學轉換**：我們將一維的線性時間映射至二維的單位圓 (Unit Circle) 空間上，定義週期 $T = 86400$：
    $$Time_{sin} = \sin\left(\frac{2\pi \cdot t}{T}\right)$$
    $$Time_{cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$
*   **神經網路意義**：這樣的二維座標使得任何相鄰時間點（包含跨日）在特徵空間中的歐式距離 (Euclidean Distance) 均保持平滑連續，幫助神經網路學習車流的日夜週期規律。

### 1.2 長尾分佈的對數轉換 (Log1p Scaling)
原模型使用 `MinMaxScaler` 將車流 $[0, Max]$ 線性壓縮至 $[0, 1]$ 區間。
*   **問題背景（線性縮放的缺陷）**：交通數據通常呈現**長尾分佈 (Long-tail distribution)**。95% 的時間車流在 $0 \sim 5$ 之間，但尖峰時段可能飆高至 $100$。若發生 100 輛車的極端值，一般車流 $2 \rightarrow 3$ 輛的變化，在正規化後會變成極微小的 $0.02 \rightarrow 0.03$，這導致在反向傳播時這類微小差異產生的梯度幾乎消失，模型難以學習非尖峰時段的波動。
*   **對數轉換矩陣 (`np.log1p`)**：使用的公式為 $X_{scaled} = \ln(1 + X)$。
*   **神經網路意義**：對數轉換能顯著壓縮尖峰數值（將 $100$ 壓至約 $4.6$），同時在低數值區域保持較陡峭的斜率，放大低車流之間的差異度。這使得優化區間變得平滑，加速模型收斂。

---

## 2. 大規模時序資料工程：Lazy Dataset 架構

處理數百萬筆時間點的 Seq2Seq 任務時，**記憶體溢出 (OOM)** 是首要工程挑戰。

### 2.1 Eager Loading 的瓶頸
若預先將龐大的一維時間序列陣列，透過 Sliding Window 展開成 $N$ 個 shape 為 `(INPUT_LEN, FEATURES)` 的二維陣列（即 Eager Loading），記憶體佔用會呈現指數級別的膨脹（產生高度資料重複），輕易佔用數十 GB 甚至上百 GB 的 RAM。

### 2.2 延遲實體化 (Lazy Materialization)
`LazySequenceDataset` 放棄了預先切割，改為維護一個輕量級的**索引映射矩陣 (`index_map`)**：
1.  **Index Mapping**: 該矩陣僅存放有效樣本的 Tuple `(File Index, Start Offset)`。
2.  **On-the-fly Slicing (`__getitem__`)**: 僅在 DataLoader 請求 Mini-Batch 數據時，才動態根據 Offset 去原始的一維全量陣列中執行 Python Slice，並將其轉換為 PyTorch Tensor 傳送至 VRAM（顯示卡記憶體）。

### 2.3 降階取樣與去零化 (Zero-Bias Mitigation)
在構建 `index_map` 迴圈中，實作了兩個影響網路訓練成效的關鍵機制：
1.  **Stride Sub-sampling**: 設定 `stride=3`，在不遺失整體趨勢特徵的前提下，減少約 66% 結構高度重疊的樣本，大幅加速單輪 Epoch 訓練時間。
2.  **Target Filtration (抗零偏置)**：若未來標籤區間 `Y` 內的全局最大車流量低於閾值（例如 $\ln(1+5)$），則直接捨棄此樣本。這能強逼模型避免陷入「總預測為 0」的局部最佳解（Local Minima），使其專注於有實質車流波動的樣本。

---

## 3. GRU 模型機理與序列架構 (Seq2Seq Direct Mapping)

模型 `GRUSequence` 的核心是由 GRU (Gated Recurrent Unit) 與全連接層 (Fully Connected Layer) 組成的非自迴歸 (Non-autoregressive) 架構。

### 3.1 Tensor 流向與維度轉換 (Shape Transformation)
假設 $Batch = 64, Input = 15, Horizon = 15, Edges = 208, Hidden = 256$。
1.  **輸入 $X$**: Shape `(64, 15, 210)`。
2.  **GRU 處理**: 輸出為整段過程中的狀態變化 `out` 形狀為 `(64, 15, 256)`，以及最後一個時間步的隱藏狀態 `h_n` 形狀為 `(1, 64, 256)`。我們提取該序列最後一步的隱藏狀態 $h_{15}$。
3.  **線性解碼 (Linear Decoding)**: 將形狀 `(64, 256)` 的 $h_{15}$ 通過 `nn.Linear(256, 15 * 208)` 層，直接映射出一個展開的 1D 預測向量，形狀為 `(64, 3120)`。
4.  **Reshape**: 利用 `.view(-1, 15, 208)`，將展開的向量重組為目標矩陣格式，一次性給定未來 15 步的所有路段預測（Direct Sequence Output）。

### 3.2 深度解析 GRU 的內部運作 (Gating Mechanism)
RNN 容易發生梯度消失，而 LSTM 參數過於龐大。GRU 結合了 LSTM 的遺忘與輸入思想進行了簡化，其內部包含兩種控制資訊流動的「門 (Gate)」：

隨著時間步 $t$ ($t \in [1, 15]$) 推進，給定當前輸入 $x_t$ 與前一步隱藏狀態 $h_{t-1}$，GRU 計算如下：
1.  **重置門 (Reset Gate, $r_t$)**：
    $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
    *目的*：決定過去的記憶 $h_{t-1}$ 有多少是無關的。若值接近 0，則神經網路傾向於忽略過去的局部狀態（例如突發但很快結束的車流）。

2.  **更新門 (Update Gate, $z_t$)**：
    $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$
    *目的*：決定有多少前一時刻的狀態 $h_{t-1}$ 應該被直接保留至當前狀態 $h_t$。若值接近 1，則信息可以被長久傳遞，完美解決長時間依賴 (Long-Term Dependency) 與梯度消失問題。

3.  **候選隱藏狀態 (Candidate State, $\tilde{h}_t$)**：
    $$\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t] + b)$$
    *目的*：基於重置門處理過的舊記憶與新輸入，計算出這一步的「新特徵摘要」。

4.  **最終狀態更新 (Final State, $h_t$)**：
    $$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$$
    *目的*：透過線性插值（結合更新門參數），由舊狀態與新候選狀態融合出當前時間點的最終記憶。在 $t=15$ 時的結算結果 $h_{15}$，即包含了這 5 分鐘內所有關於空間（208 個路段互動）與時間（車流消長）的關鍵濃縮特徵 (Context Encoding)。

---

## 4. 針對性損失設計：動態加權均方誤差 (`weighted_sequence_loss`)

神經網路訓練的目標是最小化 Loss function，這指引了梯度下降的方向。標準的均方誤差 (MSE) 為 $Loss = \frac{1}{N} \sum (\hat{y} - y)^2$。

### 4.1 高峰懲罰機制的數學設計
在交通預測任務中，對實體紅綠燈控制最具毀滅性影響的是「未預測出真實發生的大車流」。若全城市 208 個路段中大部分車流平緩，模型可能會傾向以稍微降低這 200 個低車流路段的誤差，來換取在少數 8 個尖峰路段的極大預測失敗，總體 MSE 看起來還是很低。

為了解決這項痛點，我們實作了**二元條件加權損失函數 (Condition-based Weighted Loss)**：
$$Weight(y) = \begin{cases} 
5.0, & \text{if } y > \ln(1+10) \\
1.0, & \text{otherwise} 
\end{cases}$$
$$Loss_{weighted} = \frac{1}{N} \sum (\hat{y} - y)^2 \times Weight(y)$$

### 4.2 反向傳播的影響 (Impact on Backpropagation)
在計算 Backpropagation 時，網路的參數更新量與 Loss 的梯度成正比。
*   當真實標籤 $y$ 指示某個路段在某個未來時間點出現超過 10 輛車的累積時，權重因子 `weights = 5` 會使得該特定輸出單元的梯度放大 5 倍。
*   這個巨大的梯度信號會沿著運算圖逆向傳播至 GRU 單元內部，強制網路權重 $W, b$ 去修正那些「錯過尖峰特徵」的神經元。
*   神經網路在經歷數次 Epoch 的優化後，會展現出一種防禦機制（對代價函數妥協）：**它寧願在低車流時段犧牲一點點平均準確率，也絕對不會放掉任何可能發展成交通尖峰的特徵信號**。這正是本模型在測試集中 Peak MAE 表現如此優異的關鍵原因。
