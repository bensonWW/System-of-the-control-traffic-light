# -*- coding: utf-8 -*-
"""一次性：把 ARS 第二輪修訂併入使用者已排版的 REVISED_DRAFT (1).docx。
輸出到 REVISED_DRAFT_merged.docx（不動原檔）。以文字錨點定位，保留圖片/樣式/清單編號。
"""
import copy, sys
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

SRC = "REVISED_DRAFT (1).docx"
OUT = "REVISED_DRAFT_merged.docx"

doc = Document(SRC)

def body_paras():
    out = []
    for ch in doc.element.body.iterchildren():
        if ch.tag == qn('w:p'):
            out.append(Paragraph(ch, doc))
    return out

def find(prefix, contains=None):
    for p in body_paras():
        t = p.text.strip()
        if t.startswith(prefix) and (contains is None or contains in t):
            return p
    raise LookupError(f"NOT FOUND: {prefix!r}")

def _new_run_like(template_r, text, bold):
    nr = copy.deepcopy(template_r)
    for t in nr.findall(qn('w:t')):
        nr.remove(t)
    # strip existing bold toggles in rPr; we'll set explicitly
    rpr = nr.find(qn('w:rPr'))
    if bold is not None and rpr is not None:
        for b in rpr.findall(qn('w:b')) + rpr.findall(qn('w:bCs')):
            rpr.remove(b)
    te = nr.makeelement(qn('w:t'), {qn('xml:space'): 'preserve'})
    te.text = text
    nr.append(te)
    return nr

def set_rich(p, segments):
    """segments: [(text, bold_or_None), ...]. 保留首個 run 的字型 rPr。"""
    runs = p.runs
    if not runs:
        for txt, b in segments:
            r = p.add_run(txt)
            if b is not None:
                r.bold = b
        return p
    tmpl = runs[0]._r
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    # first run
    runs[0].text = segments[0][0]
    if segments[0][1] is not None:
        runs[0].bold = segments[0][1]
    prev = runs[0]._r
    for txt, b in segments[1:]:
        nr = _new_run_like(tmpl, txt, b)
        prev.addnext(nr)
        if b is not None:
            Paragraph(p._p, p._parent)  # noop
            from docx.text.run import Run
            Run(nr, p).bold = b
        prev = nr
    return p

def set_text(p, text):
    set_rich(p, [(text, None)])
    return p

def insert_para_after(ref_p, segments):
    """clone ref_p（保留 pPr：樣式＋numPr），覆寫文字，插在其後。回傳新 Paragraph。"""
    new_el = copy.deepcopy(ref_p._p)
    ref_p._p.addnext(new_el)
    np = Paragraph(new_el, ref_p._parent)
    set_rich(np, segments)
    return np

def remove_para(p):
    p._p.getparent().remove(p._p)

# ───────────────────────── 摘要 CN ─────────────────────────
set_text(find("本研究以國立臺北科技大學周邊路網"),
 "本研究建構一套可驗證、可解釋的多策略動態號誌優化框架 TrafficVision，整合台北市即時 VD 開放資料、SUMO 微觀模擬、短時車流預測與 LLM 解釋介面。系統以雙通道 GRU 預測未來五分鐘車流與車速，於多種號誌策略間並行模擬、加權選優，且每個決策皆可追溯。")
set_text(find("在預測層，GRU pair model 於模擬 holdout"),
 "在 406 次有效模擬中，所選最佳策略相較無控制基準（路網內建固定時制）平均降低等待時間 8.6%、行程延誤 7.3%（p<.001），高壅塞情境達 10.4–12.5%；此為五策略取優相對未調校人造基準之改善。為誠實定位效能，另作兩項對照：標準自適應基準 Max-Pressure 顯著較優（38.1 vs 41.6 秒，p=0.002）；預測器消融顯示以當前觀測 persistence 驅動即略優於 GRU（p=0.003）。GRU 於車流通道全 holdout 較 persistence 改善約 58%、有車路段車速改善 87.7%，其價值在預測保真與儀表板呈現，非端到端增益。")
set_text(find("系統並以 React 與 Leaflet 儀表板呈現當前、預測與優化三種視角"),
 "本研究貢獻在於可重現、可解釋、接真實即時資料的閉環決策框架，而非最佳控制效能；Max-Pressure 等細粒度控制器可作為候選策略納入，未來並將導入 GNN、更細策略空間與公平性指標。")

# ───────────────────────── 摘要 EN ─────────────────────────
set_text(find("This study develops TrafficVision"),
 "This study presents TrafficVision, a verifiable and interpretable multi-strategy framework for dynamic traffic-signal optimization, integrating Taipei's real-time VD open data, SUMO microscopic simulation, short-term prediction, and an LLM explanation interface. A dual-channel GRU forecasts five-minute flow and speed; several signal strategies are then simulated in parallel and selected by a weighted score, with a traceable rationale per decision.")
set_text(find("For prediction, the GRU pair model reduces speed-channel MAE"),
 "Across 406 valid runs, the selected strategy cuts average waiting time by 8.6% and travel delay by 7.3% versus the uncontrolled built-in fixed timing (p<.001), rising to 10.4-12.5% under congestion - a best-of-five gain over an untuned baseline. Two honest comparisons position the framework: the standard adaptive baseline Max-Pressure is significantly better (38.1 vs 41.6 s, p=0.002), and a predictor ablation shows a simple persistence baseline slightly outperforms the GRU end-to-end (p=0.003). The GRU improves flow-channel MAE by ~58% on the full holdout and speed by 87.7% on active edges, so its value lies in prediction fidelity and the dashboard view, not end-to-end gain.")
set_text(find("A React/Leaflet dashboard presents current"),
 "A React/Leaflet dashboard presents current, predicted, and optimized views, while a locally deployed fine-tuned Gemma model explains decisions in natural language (explanation only). The contribution is a reproducible, interpretable, real-data closed-loop decision framework - not state-of-the-art control; controllers like Max-Pressure can enter as candidate strategies. Future work adds GNNs, a finer strategy space, and fairness metrics.")

# ───────────────────────── 1.3 RQ ─────────────────────────
set_text(find("本研究的核心研究問題為："),
 "本研究的核心研究問題為：能否建立一套接入真實即時開放資料、可驗證且可解釋的閉環多策略號誌決策框架，於高壅塞情境下穩健地改善交通指標？而短時車流預測在此框架中扮演何種角色——是端到端效益的來源，抑或高保真的輔助元件？後一子問題將透過與當前觀測（persistence）基準的消融予以誠實檢驗（見第五章）。")

# ───────────────────────── 1.5 貢獻 ─────────────────────────
set_text(find("本研究貢獻可歸納為四點"),
 "本研究的核心貢獻為一套可驗證、可解釋的多策略動態號誌優化框架，其餘貢獻為支撐此框架的元件與管線。依重要性排序如下：")
set_rich(find("高精度雙通道短時預測："), [
 ("可驗證、可解釋的多策略號誌選擇框架", True),
 ("：以顯式多策略並行模擬與加權評分取代黑箱學習，每個決策皆可追溯、可重現；並透過權重敏感度分析、配對統計檢定與一個標準自適應基準（Max-Pressure）對照，系統性檢驗其行為，與 RL 黑箱控制 [16] 形成差異化。此為本研究的主軸。", False)])
set_rich(find("預測驅動的可驗證、可解釋號誌優化流程："), [
 ("接真實即時開放資料的閉環自動化管線", True),
 ("：建立每五分鐘自動執行「擷取—模擬—預測—選優—呈現」的可運行流程，使上述框架得以在真實資料條件下持續運作。", False)])
set_rich(find("接真實即時開放資料的閉環自動化管線：建立"), [
 ("雙通道短時預測元件（輔助）", True),
 ("：提出同時預測車流量與平均車速的 GRU pair model，於模擬 holdout 之車流通道 MAE 較 persistence 基準改善約 58%（全 holdout，模型確實較佳）、有車路段之車速通道改善 87.7%（侷限活躍子集，全 holdout 不優於 persistence，見 4.6）。須誠實指出，預測器消融顯示框架的端到端效益並未受益於 GRU（見 5.11），故 GRU 為高保真輔助元件而非端到端增益來源。", False)])
set_rich(find("三視角視覺化與 LLM 輔助解釋："), [
 ("三視角視覺化與 LLM 輔助解釋（輔助）", True),
 ("：以儀表板與自然語言介面提升系統結果的可視性與可理解性；LLM 僅作解釋，未參與決策或自動排程。", False)])

# ───────────────────────── 2.6 ─────────────────────────
set_text(find("綜合上述，既有研究在四個面向各有侷限"),
 "綜合上述，既有研究在四個面向各有侷限：強化學習號誌控制難驗證、難解釋 [16]；短時預測多在離線資料上評估 [13] 或未與控制串接 [14]；GRU 僅捕捉時間維度，缺乏路段間空間依賴 [15]。換言之，尚無研究將「接真實即時資料的閉環、可驗證且可解釋的多策略選擇優化、可解釋介面」整合為一條完整流程。本研究即補上此缺口，並以可重現的方式驗證其效益。")

# ───────────────────────── 4.6 Holdout（1 段 → 4 段）─────────────────────────
p46 = find("Holdout 測試結果為本研究的重要成果之一")
set_text(p46, "Holdout 測試結果（20,005 對樣本）需區分兩個通道誠實呈現：")
p46b = insert_para_after(p46, [
 ("車流通道（全 holdout，模型確實優於基準）", True),
 ("：模型 MAE 為 0.129，相較 persistence baseline（0.309）改善 58.2%、相較 zero baseline（0.482）好 3.7 倍，且於 15 個預測步逐步皆勝出；高流量路段（target > 10 輛/步）MAE 為 0.779，相較 persistence（2.27）改善 65.6%。此為整體 holdout（含空、佔用 cell）的結果，反映模型確實學到車流的時間演化。", False)])
p46c = insert_para_after(p46b, [
 ("車速通道（須限定條件解讀）", True),
 ("：於有車（活躍）路段（target > 20 km/h），模型 MAE 為 3.78 km/h，相較 persistence（30.73）改善 87.7%。惟此數字須誠實設限：(1) 車速通道採 has-vehicle mask 訓練（僅監督有車 cell），模型並未被要求在空 cell 輸出零速，故於全 holdout（含空 cell）其車速 MAE 約 14 km/h、反而高於 persistence 約 5.6 km/h；(2) persistence 於活躍路段的 30.73 km/h 誤差，部分源於密集矩陣對剛佔用 cell 以零填充。因此車速通道的 87.7% 應理解為「在有車路段、相對於零填充式 persistence」的改善，而非無條件的準確度優勢；此亦與 5.11 的端到端發現一致。", False)])
insert_para_after(p46c, [
 ("上述準確度均在模擬域的 holdout 上評估，反映模型對 SUMO 生成序列的預測能力，而非對真實道路的直接驗證（見 6.3）。本段數字可由 python test_model.py（車流通道）與 python eval_speed_holdout.py（車速通道）重現。", None)])

# ───────────────────────── 4.7 安全下限段（插入）─────────────────────────
p47 = find("其設計直覺為：壅塞越嚴重")
insert_para_after(p47, [
 ("為使產生的號誌計畫符合基本可部署性，策略覆寫在既有相位結構上強制下列實體安全下限（於 tools/traffic_optimizer_signal.py 強制，可由環境變數覆寫）：最小綠燈 5 秒、最小黃燈清道 3 秒、最小全紅淨空 1 秒；低於下限者夾回。須誠實界定：本系統沿用路網檔原有相位序列（含行人通行與清道相位）僅調整綠燈秒數分配，並未針對行人專用相位獨立優化（列為限制，見 6.3）。", None)])

# ───────────────────────── 5.2 ─────────────────────────
set_text(find("系統以 SUMO 建立微觀模擬，並以即時 VD"),
 "系統以 SUMO 建立微觀模擬，並以即時 VD 與模擬生成資料建立不同流量條件。本章結果彙整自 2026 年 5 月期間的自動排程模擬：共 468 次執行紀錄，其中 62 次因即時資料長度不足（少於 GRU 所需的 15 個 timestep）而跳過，得 406 次有效模擬（以 data/runtime_data/_metrics.jsonl 之有效紀錄為宇集，全章分析均以此為準）。各策略皆於相同路網、相同需求與相同模擬條件下執行，以確保比較一致。")

# ───────────────────────── 5.4 ─────────────────────────
set_text(find("本研究比較五種策略："),
 "本研究比較五種策略：無控制基準（no_control，號誌維持路網檔內建之原始固定時制、不施加優化）、baseline_original、baseline_more_edges、baseline_more_edges_more_tls，以及結合 GRU 預測的 adaptive。須說明 no_control 為「不優化的現況」基準，而非「號誌全關」；各 baseline 策略則以不同積極程度施加號誌覆寫。在 406 次有效模擬中，各策略獲選為最佳的次數為：baseline_more_edges_more_tls 96 次（23.6%）、no_control 92 次（22.7%）、adaptive 90 次（22.2%）、baseline_original 70 次（17.2%）、baseline_more_edges 58 次（14.3%）。")

# ───────────────────────── 5.5 → 表 5.1 ─────────────────────────
set_text(find("為初步檢核 SUMO 模擬是否反映真實路況"),
 "為初步檢核 SUMO 模擬是否反映真實路況，本研究以同期 VD 快照比較主要幹道的真實車速與模擬車速（模擬車速取有車 edge 之平均）。如表 5.1，五條主要幹道的模擬車速與 VD 車速高度相關（Pearson r = 0.895，平均絕對差 2.3 km/h）：市民大道、八德路、忠孝東路三條幹道差距均在 1 km/h 內；新生南路與建國高架差距較大（+4.3、−6.1 km/h），推測與該路段車流合成密度的差異有關。須說明此為車速層級、單一同期快照、路名級彙整之檢核，而非流量層級驗證。")
set_rich(find("表 5.5　主要幹道：真實 VD 車速"), [("表 5.1　主要幹道：真實 VD 車速 vs 模擬車速（同期快照）", True)])

# ───────────────────────── 5.6 整體結果 + 表 5.2 + 圖 5.1 ─────────────────────────
set_text(find("表 5.2 彙整最佳策略相較無控制基準"),
 "表 5.2 彙整最佳策略相較無控制基準的六項指標改善。此處「最佳策略」指該次模擬中五種候選策略內綜合分數最低者（即對策略集合的逐次選優），「無控制基準」則為路網檔內建之原始固定時制、不施加任何優化，並非交通局實際部署之號誌時制。系統最佳策略平均降低等待時間 8.6%、行程延誤 7.3%，改善經配對 Wilcoxon 檢定具統計顯著性（p < .001，Cohen d = 0.48）。teleport 次數由平均 1.93 降至 1.24；惟須中性看待——teleport 為 SUMO 在車輛長時間卡滯（預設逾 300 秒）時的強制脫困機制，屬模擬內部指標而非真實交通效益，故僅列為輔助觀察（見 5.3），不計入加權評分。")
set_rich(find("表 5.2　各指標：無控制基準"), [("表 5.2　各指標：無控制基準 vs 最佳策略平均（n = 406）", True)])
set_text(find("圖 5.2　當前、預測與優化三種視角之車流熱力圖"),
 "圖 5.1　當前、預測與優化三種視角之車流熱力圖。")

# ───────────────────────── 5.7 + 表 5.3 ─────────────────────────
set_text(find("為檢驗「高壅塞情境下效益更大」的主張"),
 "為檢驗「高壅塞情境下效益更大」的主張，本研究以無控制基準的平均等待時間作為壅塞代理，將 406 次模擬分層。表 5.3 顯示改善幅度隨壅塞加劇而單調上升：高壅塞前 25%（等待 ≥ 51.4 秒）等待改善達 10.4%，前 10%（≥ 76.1 秒）達 12.5%。同時，高壅塞時「無控制」獲選為最佳的比例由整體的 22.7% 降至前 10% 的 4.9%。（全文壅塞改善幅度統一以一位小數表示：等待 8.6%／10.4%／12.5%。）")

# ───────────────────────── 5.8 + 圖 5.2（去重 caption）─────────────────────────
set_text(find("為檢驗綜合評分權重對策略選擇的影響"),
 "為檢驗綜合評分權重對策略選擇的影響，本研究利用各策略子目錄保存的逐策略指標，於 w_wait ∈ {0.5, 0.6, 0.7} 下重算各策略之綜合分數並重新選優。結果顯示等待時間改善與行程延誤改善高度相關（Pearson r = 0.991，見圖 5.2），故權重變動對選擇影響極小：三組權重下，最佳策略完全一致者達 98.8%（401/406）；由 0.6/0.4 調至 0.7/0.3 僅 0.3% 決策反轉。此結果顯示策略選擇對權重設定穩健，0.6/0.4 並非關鍵假設。")
remove_para(find("圖 5.1　等待時間比 vs 行程延誤比散點圖"))  # 去掉重複 caption
set_text(find("圖 5.1　號誌策略選擇對評分權重之穩健性"),
 "圖 5.2　號誌策略選擇對評分權重之穩健性。橫軸為勝出策略之等待時間比（優化後/基準），縱軸為行程延誤比；每點為一次由非基準策略勝出之模擬（n = 314）。Pearson r = 0.991。")

# ───────────────────────── 5.9 ─────────────────────────
set_rich(find("5.9 動態選擇的價值"), [("5.9 逐情境選擇 vs 固定單一策略", True)])
p59 = find("GRU 的價值不在於提出單一最強策略")
set_text(p59,
 "本節量化「依情境選擇策略」相對於「永遠承諾單一策略」的一致性增益。表 5.4 以平均行程延誤比較兩者：任何單一固定策略的平均行程延誤約為 46.70 秒（最佳固定策略 baseline_more_edges_more_tls），而逐情境選擇達 44.35 秒，於 74.9%（304/406）的情境中優於最佳固定策略，配對 Wilcoxon p ≈ 3.1×10⁻⁵²。")
insert_para_after(p59, [
 ("須對此結果的強度誠實設限：本系統的選擇是在每次模擬中挑出五個候選裡綜合分數最低者，因此「逐情境選擇優於任一固定策略」在設計上幾近必然成立——選擇者擁有嚴格更多的自由度。此處的數值並非令人意外的實證發現，而是對「為每個情境選對策略」之一致性優勢的量化；其 p 值極小亦為此構造的必然推論，不應解讀為強烈的實證發現。框架的價值定位另見 5.10 與標準自適應基準的對照、以及 5.11 對預測器角色的釐清。", None)])
set_rich(find("表 5.4　動態選擇 vs 永遠固定單一策略"),
 [("表 5.4　逐情境選擇 vs 永遠固定單一策略（指標：平均行程延誤，n = 406）", True)])

# ───────────────────────── 舊 5.10→5.11 ；舊 5.11→5.12（先改標題與內文）─────────────────────────
set_rich(find("5.10 預測器消融"), [("5.11 預測器消融：GRU 預測 vs 當前觀測", True)])
set_text(find("為檢驗 GRU 預測相較於簡單基準（persistence"),
 "為檢驗 GRU 預測相較於簡單基準（persistence，即以當前觀測重複作為預測）對端到端優化的實際貢獻，本研究進行消融實驗。在與第五章其餘分析相同的 406 次有效模擬宇集內隨機抽取 60 次（與 5.10 之 Max-Pressure 對照為同一批 run），分別以 GRU 預測與 persistence 預測驅動完整的策略評估流程；兩者使用相同路網與車流需求，差異僅在驅動號誌計畫的預測來源，因此為乾淨的對照。")
set_text(find("結果如表 5.6：GRU 驅動之平均行程延誤為 140.9 秒"),
 "結果如表 5.6（n = 60）：GRU 驅動之平均行程延誤為 41.62 秒（中位數 31.90），persistence 驅動為 40.21 秒（中位數 31.96）；persistence 於 60 次中有 37 次（61.7%）較佳，GRU 僅 15 次（25.0%），其餘平手。配對 Wilcoxon 檢定顯示 persistence 顯著優於 GRU（排除平手 n = 52，p = 0.003），幅度約 3.5%。換言之，在五分鐘預測水平與五策略的選擇任務下，以更簡單、無需訓練的 persistence 驅動選優，端到端反而略優於 GRU。此處延誤量級（約 40–42 秒）與表 5.2 一致；早期版本曾誤報約 140 秒，係因消融腳本以檔案系統掃描納入未列入 _metrics.jsonl 有效宇集的鎖死（gridlock）run（延誤動輒數百秒），現已修正為與正文同宇集。")
set_text(find("此結果可由本研究先前的發現解釋"),
 "此結果可由本研究先前的發現解釋：策略選擇對微小擾動本即穩健（見 5.8，r = 0.991），而五分鐘水平下交通狀態高度自相關，當前觀測已是強預測；故 GRU 於預測層的準確度優勢（見 4.6）在粗粒度的策略選擇上不僅被稀釋，其預測誤差於少數情境甚至誤導了策略選擇，使端到端表現略遜於 persistence。GRU 的價值因此主要展現於預測保真度與儀表板的預測視圖；如何讓預測精度真正轉化為端到端效益（更細的策略空間、更長的水平），列為未來工作（見 6.4）。")
set_rich(find("表 5.6　預測器消融"),
 [("表 5.6　預測器消融：GRU vs persistence（n = 60，指標：最佳策略行程延誤）", True)])
set_text(find("註：平手 12 次；配對 Wilcoxon"),
 "註：平手 8 次；配對 Wilcoxon p = 0.003（persistence 顯著較佳）。宇集與表 5.2 相同，seed = 0 之 60 run 與表 5.5 為同一批。可由 python tools/da1_ablation.py --sample 60 重現，明細見 data/analysis/da1_ablation_clean_n60.csv。")

set_rich(find("5.11 結果討論"), [("5.12 結果討論", True)])
p512 = find("實驗結果支持本研究優化流程的可行性")
set_text(p512,
 "綜合本章結果，可將本框架的效能與限制誠實定位如下。其一，可行性成立：框架的逐情境動態選擇相較無控制基準改善等待時間 8.6%（p < .001），於高壅塞情境達 10.4–12.5%；惟此 8.6% 係「五策略取優」相對於未經調校之人造基準，並非相對於實際部署號誌或最佳控制器的改善。其二，相對於標準自適應控制仍有差距：細粒度的 Max-Pressure 基準顯著優於本框架的粗粒度五策略選擇（5.10，p = 0.002）。其三，端到端效益未受益於預測器精度：預測器消融（5.11）顯示以簡單的 persistence 驅動端到端反而略優於 GRU 且達顯著（p = 0.003）。")
insert_para_after(p512, [
 ("三者合而指向同一結論：本研究的核心貢獻應理解為「可驗證、可解釋、接真實即時開放資料的閉環選擇框架」本身，而非最佳的控制效能或 GRU 的端到端增益。限制端到端表現的瓶頸在於粗粒度的策略空間與五分鐘的選優節奏，而非預測精度；Max-Pressure 等細粒度控制器、以及更細的策略參數空間，正是此框架後續可納入的候選（見 6.4）。", None)])
set_text(find("號誌優化仍需考量整體路網平衡"),
 "號誌優化仍需考量整體路網平衡。延長某方向綠燈可能增加其他方向的等待，本研究目前以全路網平均指標衡量，尚未納入交叉路口公平性、用路人等待代價之分配，亦未對行人專用相位獨立優化（見 6.3）。未來可引入更多路網層級與公平性指標。")

# ───────────────────────── 6.1 ─────────────────────────
set_text(find("本研究完成並驗證一套整合即時開放資料"),
 "本研究建立並驗證一套可驗證、可解釋的多策略動態號誌優化框架 TrafficVision，整合即時開放資料、SUMO 微觀模擬、雙通道短時預測、多策略號誌評估、前端儀表板與大型語言模型解釋介面。其核心為一條可重現、可追溯的閉環決策流程：以 GRU 預測短時車流，於多策略間並行模擬、加權選優，並對每個決策保留依據。")
set_text(find("在 405 次有效自動排程模擬中"),
 "在 406 次有效自動排程模擬中，框架選出的最佳策略相較無控制基準（路網內建固定時制）平均降低等待時間 8.6%、行程延誤 7.3%（p < .001），且改善隨壅塞加劇而上升（高壅塞情境達 10.4–12.5%）；此係「五策略取優」相對於未調校之人造基準。前端儀表板與 AI 解釋介面進一步提升結果的可視性與可理解性。")
set_text(find("須誠實指出，預測器消融顯示此端到端優化於五分鐘水平下對預測器選擇穩健"),
 "須誠實指出本框架在效能上的定位。其一，與標準自適應基準 Max-Pressure 對照，細粒度的 Max-Pressure 顯著優於本框架之粗粒度五策略選擇（見 5.10，p = 0.002）；其二，預測器消融顯示端到端優化於五分鐘水平下並未受益於 GRU，以簡單的 persistence 驅動反而略優且達顯著（見 5.11，p = 0.003）。因此本研究的核心成果應理解為「可驗證、可解釋、接真實即時開放資料的閉環選擇框架」之可行性驗證，而非最佳控制效能或 GRU 的端到端增益；Max-Pressure 等細粒度控制器可作為候選策略納入此框架。")

# ───────────────────────── 6.3 限制（list 插入）─────────────────────────
set_rich(find("模擬域驗證（sim-to-real 落差）"), [
 ("模擬域驗證（sim-to-real 落差）", True),
 ("：GRU 的訓練與 holdout、以及號誌優化的評估，皆在 SUMO 內進行；所報告的預測準確度與優化效益是在模擬域內證明，真實道路的效益尚未驗證。本研究因此定位為部署前的模擬驗證平台。", False)])
set_rich(find("合成 OD 之需求保真度"), [
 ("合成 OD 之需求保真度", True),
 ("：VD 不含 OD，模擬需求係由可得資料重建，與真實 OD 存在落差。惟車速層級的保真度檢核顯示模擬與真實 VD 速度高度相關（r = 0.895，平均絕對差 2.3 km/h，見 5.5）；流量層級的驗證因單位族不同，仍待後續處理。", False)])
p_l3 = find("GRU 端到端增益有限")
set_rich(p_l3, [
 ("相對標準自適應控制仍有差距", True),
 ("：本研究納入標準自適應基準 Max-Pressure 作對照（見 5.10）。在 n = 60 配對模擬中，細粒度、即時佇列驅動的 Max-Pressure 顯著優於本框架的粗粒度五策略選擇（38.1 vs 41.6 秒，p = 0.002）。本框架目前效能上限受限於僅五個粗粒度候選策略與每五分鐘的選優節奏；其貢獻在於可驗證、可解釋的框架本身，而非最佳控制效能。", False)])
p_l4 = insert_para_after(p_l3, [
 ("預測器端到端未具增益（反略遜）", True),
 ("：n = 60 之消融（與正文同宇集）顯示，於五分鐘、五策略的設定下，以 persistence 驅動反而略優且達顯著（見 5.11，p = 0.003，約 3.5%）。GRU 的價值因此主要在預測保真度與儀表板呈現。", False)])
insert_para_after(p_l4, [
 ("行人相位與公平性未納入優化", True),
 ("：策略覆寫雖強制最小綠燈／黃燈清道／全紅淨空等實體安全下限（見 4.7），並沿用原相位之行人通行，但未對行人專用相位獨立優化，亦未納入交叉路口間的等待公平性指標；對行人需求高的路口，實務部署須補上行人最短綠燈與專用相位約束。", False)])

# ───────────────────────── 6.4 未來展望（list 插入）─────────────────────────
p_f1 = find("放大 GRU 端到端價值的條件")
set_rich(p_f1, [
 ("將細粒度控制器納入候選策略集合", True),
 ("：對照顯示 Max-Pressure 等細粒度、即時佇列驅動的控制器顯著優於目前的粗粒度五策略選擇（見 5.10）。由於本框架的選優機制與控制器無關，未來可直接將 Max-Pressure（及 actuated、RL 控制器等）納入候選集合，由框架在可驗證的前提下為每個情境選用最適控制器（對應限制 3）。", False)])
insert_para_after(p_f1, [
 ("放大預測器的端到端價值", True),
 ("：消融顯示 GRU 於目前設定的端到端增益有限（見 5.11）。未來可改用更細緻或連續的策略參數空間，或延長預測水平（如 15–30 分鐘，此時 persistence 會明顯退化），使 GRU 的預測精度轉化為端到端優勢（對應限制 4）。", False)])
set_rich(find("導入圖神經網路（GNN）"), [
 ("導入圖神經網路（GNN）", True),
 ("：學習路段與路口間的空間依賴 [15]，提升預測與選優效果（對應限制 6）。", False)])

# ───────────────────────── 參考文獻 [19] ─────────────────────────
p18 = find("[18] Y. Hu, L. Du, and S. M. Easa")
insert_para_after(p18, [
 ("[19] P. Varaiya, “Max pressure control of a network of signalized intersections,” Transportation Research Part C: Emerging Technologies, vol. 36, pp. 177–195, 2013, doi: 10.1016/j.trc.2013.08.014.", None)])

doc.save(OUT)
print("STAGE1 (text) done ->", OUT)
PY = True
