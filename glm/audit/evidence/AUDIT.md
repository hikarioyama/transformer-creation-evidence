# next-token-machine 監査報告

## 総合判定

**総合判定: 一部に欠陥。**

確認できた事実は、出荷された固定重みをブラウザ内の自前実装で推論でき、独立 NumPy 参照実装による数値・因果性・KV cache 検証は全件 PASS だったこと、そして実ブラウザの操作も起動・編集・cache 切替まで実行できたことである。一方、KV append 後の embedding trace 欠落、attention 弧の層/深度不一致、段階 marker の stale/wrong-stage 表示、編集後の Verify 表示の stale、chip 押下と選択位置の表示不一致が確認された。

最も重大な表示欠陥は、実計算の trace を表示するという中心契約が cache append 後に崩れることと、attention 弧が選択した層/トークン post の位置に描画されないことである。後者は少なくとも S08/S09/S32 で再現した。

制作時の訓練禁止、制作・試験・起動の実行履歴、2 時間以内という時間制限、性能時間は、保存ログなしには確定できない。`tools/train.py` の存在、README の説明、ファイルの mtime から訓練時間・token 数・履歴を推定していない。ブラウザの実行時に訓練や外部推論 API を使わなかったことと、制作時のプロセス履歴を使い分けて判定する。

## 1. 対象、完全性、監査方針

対象は変更前の `[REDACTED_LOCAL_PATH]` とそこから参照されるアプリ資産である。監査中にアプリ本体を修正していない。開始時点では `audit/` は存在せず、初回の簡略列挙は 25 件だった。その後、依存物も含む完全な baseline を補正取得し、全 251 エントリ（regular file 202、symlink 2、directory 47）を記録した。以下の比較を最終集計時および validator で再実行した。

```text
python3 audit/baseline/verify_integrity.py
{"ok":true,"expected_count":251,"observed_count":251,"added":[],"removed":[],"changed":[]}
```

したがって、上記の defect は不変 original に対する証拠であり、監査用成果物が対象を変更した結果ではない。git repository ではないため git diff や commit 履歴を完全性の根拠にはしていない。

判定単位は、REQ domain 1 では原要求の 1 checklist item、数値 domain では独立に実行した 1 sequence/pair/cache sequence、ブラウザ domain では 1 完全シナリオである。シナリオ内の assertion は測定値として保持し、分母を水増ししない。PASS 比率は `PASS/(PASS+FAIL)`、BLOCKED と UNEXECUTED は隣接表示して比率から除外する。ブラウザは primary 欄だけを選別せず、最終 `audit/browser/results.json` の全シナリオを集計する。

## 2. 宣言されたモデルと実装の照合

|項目|宣言/実装で確認した内容|根拠|
|---|---|---|
|入力/位置|V=16 の token id、D=16、学習済み絶対位置 P (16×16)、`E[token_i]+P[i]`|`js/weights.js:1-4`, `js/model.js:75-98`|
|decoder|pre-LN の 2 block、各 block の LN → self-attention → residual → LN/MLP → residual|`js/model.js:146-229`|
|attention|Q/K/V は 16×16、2 heads、head dim 8、score は `q·k/√8`、j≤i の causal mask、softmax、concat/Wo|`js/model.js:103-124`|
|MLP|16→64→16、GELU tanh 近似、residual|`js/model.js:20-23`, `js/model.js:146-229`|
|normalization|LayerNorm は population variance、ε=1e-5|`js/model.js:58-71`|
|出力|final LN、`Wu` から 16 logits、softmax probability、token id|`js/model.js:146-229`|
|数値/重み|JavaScript Number/Float64Array（binary64）、固定 seed 1234 の出荷 weights、context=16|`js/weights.js:1-4`, `js/model.js:75-98`|
|cache|prefix K/V を保持し、新規位置の Q/K/V を計算して既存 K/V と attention に利用|`js/model.js:232-357`|

learned absolute position、pre-LN、GELU、context 16、no BOS、RoPE を使わないこと、量子化を行わないことは、それ自体を欠陥とはしない。今回の原要求から training/fluency/RoPE/quantization/API などを勝手に追加した評価はしていない。

README は offline training と trained/frozen weights を説明し、`tools/train.py` は training tool を含む。しかし entry page は `js/weights.js` を読み、ブラウザで `js/model.js` の固定重み推論を行う。ブラウザが training tool を実行した証拠はなく、production process の履歴はログ不在である。また README の training steps 2000 と `tools/train.py` の source 6000（`tools/train.py:283` 付近）には記述差がある。これは履歴を推測せず、再現性/証跡の限界として記録する。

## 3. 独立数値監査

`audit/numerical/reference.py` は NumPy の行列式・softmax・LayerNorm・GELU を用いた独立 oracle であり、対象の forward function の翻訳ではない。対象 unchanged の integrity verifier を通し、instrumented model は call counter、capture、tag のみを追加した。比較前に保存された `audit/numerical/tolerances.json` を読み、float64 の規則

```text
abs(a - r) <= 1e-9 + 1e-7 * abs(r)
```

を適用した。最大絶対誤差は `3.019806626980426e-14` であった。E/P/embed、両 block の LN/Q/K/V/scores/attention/headOut/concat/proj/residual/MLP、final LN、logits/probs を比較した。

- full reference: 固定 seed `20260906` の boundary/short/repeated/alternating/length 15/16/random を含む 100 sequence、長さ 1–16、全 PASS。
- causality: 12 の同長 suffix pair（length 16/15/12/8/4、複数 split）で、suffix 変更が prefix の全中間値・logits/probs を変えないことを確認、全 PASS。
- cache incremental: 100 sequence、全 prefix/forced-token append、合計 938 prefix append step。各 append で fresh Q/K/V projection 一回、prefix K/V object identity 保持、stored K/V と attention の利用、full/cache 各位置の出力一致を確認、全 PASS。これは counter/identity/read-dependence に基づく実再利用であり、経過時間からの推定ではない。
- cache read-dependence: 190 PASS、長さ 1 の過去 KV がない 5 件は明示的 N/A。過去 K または V に +1.0 を加えたとき fresh logits/attention が閾値 `1e-12` を超えて変化した。
- supporting reproducibility: 1 件を六 domain の分母外に保持。

## 4. ブラウザ監査

最終対象 URL は `http://localhost:8765/`（unchanged target）で、実ブラウザの DOM/state/render snapshot、page/request/console error、スクリーンショットを保存した。最終 runner は 33 シナリオを実行し、**26 PASS / 7 FAIL / 0 BLOCKED、未実行 0** となった。FAIL は次の 7 件である。

```text
S04-step-walk-all-15
S05-token-chip-selection
S08-attention-L1H1
S09-L1H2-selector
S22-cache-append-twice
S29-verify-stale-after-edit
S32-attention-causal-render-inputs
```

各ケースの scenario JSON、raw state、screenshot は `audit/browser/scenarios/`、`audit/browser/raw/`、`audit/browser/screenshots/` にあり、集計 JSON は全 33 件を保持する。独立 browser comparator は 51 raw snapshots/11 unique sequences を NumPy trace と比較し、49 PASS/2 FAIL/0 BLOCKED。S20 と S22 の `tokEmb/posEmb/embed` および block `x2_in` の shape mismatch は D4 の独立補強であり、DOM の見た目だけからの推測ではない。画面の確率表示は 0.1 percentage point 刻みの丸め (`toFixed(1)%`) なので、runner の表示対 trace assertion は確率差 5e-4 を丸め誤差として許容する。これは数値 oracle の 1e-9/1e-7 tolerance とは別である。

S32 のスクリーンショットでは token id `tN` は読めるが、3D label の重なり・行列背後での fade と tofu（フォントにない記号の代替表示）が見える。これは数値計算の誤りとは別の可読性/環境依存表示限界として記録する。S22 は cache ON と「reused 9 cached rows, computed 1 new」を表示し、色付き slab も見栄えがするが、trace の embedding row 欠落を隠せない。

## 5. 原要求 checklist

|ID|判定|要約|
|---|---|---|
|REQ-01|PASS|ローカル canvas、予測、編集の interactive 3D surface を起動・操作。|
|REQ-02|PASS|V16/D16、2 block/2 head の metadata と実装 shape が一致。|
|REQ-03|PASS|出荷固定 weights の local inference と独立再現を確認。再訓練履歴までは立証しない。|
|REQ-04|BLOCKED|原文の「訓練や外部推論 API を使わず自前計算」を保持。ブラウザ実行は no-training/no-external-API PASS だが、制作時に訓練を使わなかった履歴はログ不在で独立立証不能。|
|REQ-05|PASS|画面/README が toy、offline、frozen weights、no API を開示。|
|REQ-06|FAIL|通常 forward の中間値は実値だが、cache append の embedding trace と semantic marker に確認済み不整合。|
|REQ-07|PASS|edit/append/pop/demo と KV toggle の存在・状態更新・切替 equivalence を実行。chip click の任意選択表示不一致は usability/display defect として別計上し、この要件を FAIL にしない。|
|REQ-08|FAIL|arc の層/深度、cache embedding append、Verify stale のため、計算結果との faithful display を満たさない。|
|REQ-09|PASS|独立 NumPy oracle、因果性、cache の全数値ケース PASS。|
|REQ-10|PASS (制限付き)|導入説明・教育的 stage/value UI・視覚品質を実画面で確認。dense layout、tofu、label overlap、arc/marker 不整合は限界として、数学判定と相殺しない。|
|REQ-11|BLOCKED|README/package の起動手順はあるが、制作/試験/起動履歴と 2 時間以内の証跡ログはない。|

REQ-04 の複合判定は、観測可能なブラウザ subcheck（training 起動なし、外部推論 API なし）と、観測不能な production-process provenance を混同しないため BLOCKED とした。README の trained weights や `tools/train.py` は履歴の独立証明ではない。

## 6. 六つの集計 domain

|domain|PASS|FAIL|BLOCKED|N/A|UNEXECUTED|実行|PASS/(P+F)|単位|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|original_requirement_compliance|7|2|2|0|0|9|0.7777777777777778|要求 item|
|independent_numerical_parity|100|0|0|0|0|100|1.0|full sequence|
|causality|12|0|0|0|0|12|1.0|suffix pair|
|cache_outputs_and_actual_reuse|100|0|0|0|0|100|1.0|cache sequence|
|post_interaction_state_consistency|16|3|0|0|0|19|0.8421052631578947|browser scenario|
|display_vs_real_computation|10|4|0|0|0|14|0.7142857142857143|browser scenario|

REQ-04 と REQ-11 の BLOCKED 2 件は domain 1 の隣接値であり、比率の分母に入れていない。全 33 件を state/display のどちらか一つに割り当てた。各 ID、status、assertion、証拠パスは `audit/results.json` の `cases` と `domain_summaries` に機械可読で保存する。

## 7. 確認済み defect

### D1 — high: L1 attention 弧が L2 scene に所属
- 違反する原要求: REQ-08（可視化と cache の計算結果を忠実に一致させる）。

- 最小再現: intro を閉じ、step 5（Block 1 / Attention）、または S08 の L1/H1 を選択。
- 期待: 選択した L1/H1 の弧/ring が Block-1 attention scene に属し、選択 token の post と対応。
- 実測: Block-1 の tubes=0/rings=0、Block-2 の tubes=28/rings=1。
- 原因: `arcGroup` が global に再代入され、stage/post group の構築後に最後の layer group を参照する (`js/viz3d.js:272-275,394-420,498-540`)。
- 証拠: `audit/browser/scenarios/S08-attention-L1H1.json`, raw、screenshot、および S09-L1H2 supplemental fail。

### D2 — high: attention 弧/ring と token post の深度がずれる
- 違反する原要求: REQ-08（可視化と cache の計算結果を忠実に一致させる）。

- 最小再現: S32 の Block-2 attention で弧 endpoint と token post を比較。
- 期待: endpoint/ring が token-post plane に一致。
- 実測: postGroup local z=3.4、arc/ring local z=0、world-depth delta=3.4。
- 原因: `stageZ` は計算されるが未使用で、TubeGeometry/ring は z=0 のまま (`js/viz3d.js:395-419,508-521,533-540`)。
- 証拠: S32 scenario/raw/screenshot。

### D3 — medium: token chip の押下が selected position を更新しない
- 違反する原要求: なし（REQ-07 の edit/KV-toggle 要件には含めず、表示/使いやすさの観察として記録）。

- 最小再現: Embedding/Values で position 2 の chip を押す。
- 期待: chip が選択可能と表示するなら selPos=2、inspector が position 2 (`t5`) を示す。
- 実測: palette は開くが selPos は 7 のまま、タイトルは position 7 (`fish`) のまま。
- 原因: chip handler は `openPalette` のみで `state.selPos` を設定しない (`js/app.js:424-435,449-489`)。
- 分類: これは任意 chip-selection の display/usability defect であり、edit/KV-toggle の原要求 REQ-07 を FAIL にする根拠にはしない。
- 証拠: S05 scenario/raw/screenshot。

### D4 — high: cache append で embedding trace rows が欠落
- 違反する原要求: REQ-06（段階表示と実際の中間値）および REQ-08（可視化と cache の計算結果を忠実に一致）。

- 最小再現: KV ON で 8-token demo を prefill し、prediction append を二回実行して S22 を再生。
- 期待: 現在 T の `tokEmb/posEmb/embed` が T 行で、新行を inspect できる。
- 実測: S22 は T=10、IDs/logits=10 だが `tokEmb=8, posEmb=8, embed=8`。独立 comparator でも S20 は 1 vs 16、S22 は 8 vs 10 の shape mismatch。新しい row index 8,9 が欠落する。
- 原因: `mergePartial` は fresh block/final rows をコピーするが `tokEmb/posEmb/embed` を append/copy しない (`js/app.js:262-282,323-333,498-505`)。
- 証拠: S22 scenario/raw/screenshot、`audit/browser/independent_compare.json`、`audit/numerical/findings.md:17-19`。

### D5 — medium: semantic stage marker が stale/wrong-stage に蓄積
- 違反する原要求: REQ-06（段階表示と実際の中間値）および REQ-08（可視化と cache の計算結果を忠実に一致）。

- 最小再現: step 1–15 を順番に移動し、focused semantic stage に対する marker を確認。
- 期待: marker state は現在の semantic stage に更新される。重複する embed/output step が一つの stage を共有することは許容するが、過去 step の marker が current として残らない。
- 実測: step 2 の focused stage=[0] に対し actual=[0,1]、step 5 の focused stage=[3] に対し actual=[0,1,2,3,4]、step 15 の focused stage=[12] に対し actual=[0..11]。複数 highlight 自体を欠陥としたのではなく、現在選択との対応が崩れている点を defect とした。
- 原因: `updateSelectionMarkers` が semantic `STEP2STAGE` mapping ではなく `state.step` を使い、既存 marker の clear/update も不適切 (`js/app.js:383-390,704-708`; `js/viz3d.js:117-126,180-185`)。
- 証拠: S04 scenario/raw/screenshot。

### D6 — medium: edit 後の Verify 表示が stale
- 違反する原要求: REQ-08（可視化と cache の計算結果を忠実に一致）。

- 最小再現: Verify を開いて PASS を表示した後、先頭/中間 token を edit。
- 期待: 旧検証値を消す、または stale と明示する。
- 実測: IDs が変わった後も旧 `✓ max |Δ| = 0 (bitwise identical)` が残る（S29 FAIL）。
- 原因: state の verify 値が edit/recompute で無効化されず、`renderVerifyTab` が旧値を描画する (`js/app.js:9-23,284-318,684-703`)。
- 証拠: S29 scenario/raw/screenshot。

## 8. 未確定の source risk（欠陥件数に含めない）

- Attention selector の `state.attL` と 3D arc 更新の同期には source 上の risk がある (`js/app.js:576-595`, `js/app.js:372-381`)。今回の実行で D1/D2 は確認したが、全 selector/layer 組合せの原因分離をこの risk 単独の defect としては数えない。
- MLP panel のラベルに typo の疑いがある (`js/app.js:538-544`)。数値自体の誤りや原要求違反を示す独立証拠がないため、source-only risk とする。

## 9. 定性的評価（数学判定とは別）

- **明瞭さ:** 比較的明瞭。1680×1000 では左の 15 stage、中央の色付き tensor/bar、右の Attention/Values/Weights/Verify、下の edit/cache 操作が分離され、導入 card が toy/offline/real inference を説明する。一方、情報密度、カメラ角度、label overlap、tofu、marker/arc 不整合が意味の明瞭さを下げる。
- **使いやすさ:** 部分的に良いが重要操作に欠陥。intro close、15 段階移動、edit/predict/pop/demo、KV toggle、Verify は操作できる。chip selection、cache append 後の Embedding、edit 後 Verify 表示に不整合がある。
- **視覚品質/訴求:** 高い演出性。暗色背景、cyan/amber 値表現、段階 camera、色分け token chip/post、card panel は統一されている。S08 は colorful な post/landscape が見えるが L1 弧がない。S22 は cache reuse 表示と slab が視覚的に分かりやすいが trace 欠落を補わない。S30 の Verify panel は暗色 UI と整合する。美観や教育的 appeal は数学的欠陥を相殺しない。

## 10. 実行・再現

全コマンド、環境、終了コード、成果物パスは `audit/commands.txt` に記録する。主な最終コマンドは次の通りである。

```text
python3 audit/baseline/verify_integrity.py
python3 audit/report/build_results.py
python3 audit/report/validate_results.py
```

数値側の環境は Python 3.14.5、NumPy 2.4.4、Node v26.7.0、Linux、binary64。ブラウザ側は既存の audit-browser-server が提供する 127.0.0.1:8765 を使用した。GPU job は開始・停止していない。formatter、linter、project-wide test suite は実行していない。

## 11. 結論と範囲外

この audit は、固定重みモデルの数学、因果性、cache 再利用、実ブラウザで観測できる state/display、画面の可読性を対象にした。数値コアは独立 oracle の範囲で強く再現できるが、UI の faithful display 契約は D1–D6 により未達である。ブラウザ DOM assertions は画面操作と trace/render input の観測であり、全ての文字セルの意味理解や別環境フォント表示を保証しない。独立 comparator は 51 raw snapshots を比較するが、S20/S22 の shape failure を含み、すべての DOM layout を数値で証明するものではない。cross-runtime bitwise identity は主張せず、保存された tolerance 結果だけを根拠とする。

本報告は、性能 benchmark、GPU speed、training history、生成品質/fluency、RoPE の採否、quantization、公開 API の設計を評価しない。production process と「2 時間以内」の証跡ログが追加されない限り、その部分は BLOCKED のままであり、README の主張や mtime から補完しない。
