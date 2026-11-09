# MIRIX 安全性與資料完整性修復

基準為 `main` 的 `60cd444cc92d425f001d690734d901c78ed46b3c`。
修復工作區位於 `fix/security-snapshots-evaluation`。

## 核對結果

使用程式碼檢查、修正前的失敗案例與回歸測試，確認以下 11 類問題存在。

| 問題 | 修復內容 | 驗證 |
|---|---|---|
| API 身份驗證可繞過 | 私有路由統一驗證 JWT／API key，使用 request state 保存驗證結果；client/org header 不再是憑證；無效 JWT 不降級接受其他身份 | ASGI 測試覆蓋匿名、偽造 header、無效 key／JWT 與有效憑證 |
| client 修改／刪除缺少驗證 | client 查詢、更新、刪除要求自己的 dashboard JWT；限制跨組織與自行提升權限 | 自己的 token 成功，其他 client 的 token 被拒；拒絕後不呼叫修改 manager |
| 回應包含密碼雜湊 | 公開 `ClientResponse` 與內部 `Client` 分開；登入仍保有內部雜湊 | HTTP 回應不含雜湊，實際 SQLite 登入驗證仍成功 |
| Docker 共用 JWT／密碼預設值 | 要求設定 JWT、PostgreSQL、Redis secret；JWT 拒絕舊公開預設值；服務埠限制在 loopback；Redis 啟用原生密碼驗證 | Compose 缺少任一必要 secret 時失敗；檢查所有公開埠；實際 Redis 匿名連線被拒 |
| 快照還原失敗卻顯示成功 | PostgreSQL 的 truncate 與 restore 在同一交易，SQL 錯誤立即中止；Neo4j 使用交易；失敗不印出 loaded | 實際 PostgreSQL 約束衝突保留還原前資料；Neo4j 約束衝突回滾圖清除 |
| 已刪記憶仍可搜尋 | 補上 SQL 軟刪除條件；Redis 結果回查資料庫；圖引用只接受仍存在的記憶；不把孤立 anchor 當作記憶輸出 | SQLite 五種記憶搜尋、殘留快取、PostgreSQL 圖搜尋與 Neo4j 引用測試 |
| 不同數值錯誤合併 | 保留符號、小數、日期、百分比、幣別、單位、順序與重複次數；frame 身份包含 literal；Dream 合併增加硬性檢查 | 數值差異、相同事實、frame 身份與錯誤 LLM 合併建議測試 |
| 空快照留下舊圖 | 空圖也必須執行交易式清除；區分「刻意未擷取圖」與「擷取失敗」 | 將空快照還原到有舊節點的實際 Neo4j，結果為零節點 |
| 時間型別導致快照失敗 | 對 Python／Neo4j 日期、時間、duration 等使用帶型別的 JSON 編碼 | 實際 Neo4j 日期、時間與 duration 往返；保留奈秒精度及關係屬性 |
| 評測日期錯配 | 使用 session 自己攜帶的日期，避免以 chunk 序號查找 session | 測試重疊視窗及不連續的 session 編號 |
| QA-only 仍可修改記憶 | 載入既有 agent；跳過初始化、ingest 與所有 Dream；加入寫入拒絕檢查 | 初始化、尾端 Dream、記憶新增及 HTTP 寫入拒絕測試 |

另外補上：註冊完成後的管理員驗證、首次註冊的併發保護、API key 的
read-only 權限、user/org 與搜尋參數的租戶檢查、快照名稱路徑限制、
完整快照才發布，以及還原後清理相關 MIRIX 快取。
沒有 organization_id 的舊 client 也維持 default organization 範圍，不會因此
省略查詢的租戶條件。

## 驗證結果

- 完整非 integration 測試：**3,229 通過、52 跳過、242 排除**。
- 額外執行隔離 PostgreSQL／Neo4j integration 測試：**9 通過**。
- Docker Compose 設定、Python 致命 lint、shell 語法、`git diff --check` 通過。
- 實際 Redis：匿名存取被拒；有效密碼可存取。

主要新增測試：

- `tests/test_security_regressions.py`
- `tests/test_auth_storage_regressions.py`
- `tests/test_memory_identity_regressions.py`
- `tests/test_snapshot_regressions.py`
- `tests/test_eval_read_only_regressions.py`
- `tests/test_docker_security.py`
- `tests/test_security_storage_integration.py`

測試使用隔離資料庫與模擬模型，沒有對正式資料執行刪除或還原，也沒有重新跑
付費 LLM 評測。此結果驗證所列缺陷及相關回歸，不代表完整的滲透測試或整個專案
沒有其他漏洞。

## 使用與相容性

- SDK／評測程式需要有效 `MIRIX_API_KEY`；僅提供 client/org ID 不再能通過驗證。
- 第一個 dashboard 帳號可 bootstrap；之後的註冊需要管理員 JWT。
- Docker 使用前，設定獨立的 `MIRIX_JWT_SECRET_KEY`、`MIRIX_PG_PASSWORD`、
  `MIRIX_REDIS_PASSWORD`。舊公開 JWT 預設值不再能啟動服務。
- 新快照不覆寫同名目錄；PostgreSQL 工具須在 PATH 或 `MIRIX_PG_BIN` 中。
  QA-only 需要既有 agent；保存快照時可用 `--agents` 納入 agents/messages。
- 快照仍是指定資料表的備份，不是整個資料庫的完整備份。還原保留原有的
  `TRUNCATE ... CASCADE` 行為，需留意依賴表並停止並行寫入。
- **PostgreSQL、Neo4j、Redis 不是單一共同交易。** 各資料庫的失敗會明確回報；
  若 PostgreSQL 已提交，後續 Neo4j／快取失敗無法回滾已提交的 PostgreSQL。
- 已經被錯誤合併的舊記憶不會憑空復原；需從可信的原始資料或備份重新建立。

歷史上已外洩的 provider key 仍須由持有人撤銷。移除程式中的 literal 或
修改 Docker 設定不會使那些金鑰失效。
