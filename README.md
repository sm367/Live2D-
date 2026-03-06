# PNG1枚からLive2D用PSDを生成するWebツール（SAM2版）

FastAPIベースのローカルWebツールです。1枚のPNGをアップロードし、SAM2互換のマスク候補を生成して、クリック選択でパーツ切り出し→のりしろ補完→PSD書き出しを行います。

> 現在の実装は **SAM2の実推論を差し替え可能な構成** で、既定ではOpenCV候補生成フォールバックを使います。

## 構成

- `backend/main.py` : APIサーバー本体（メモリセッション管理 / 候補配信）
- `backend/services/sam2_service.py` : SAM2互換サービス（候補生成・クリック選択）
- `backend/services/inpaint_service.py` : のりしろ補完（OpenCV inpaint）
- `backend/services/psd_service.py` : `psd-tools` を使ったPSD出力
- `frontend/index.html`, `frontend/app.js` : フロントエンド（候補表示 + クリック選択UI）

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 起動（ポート8001）

```bash
py -3.12 -m uvicorn backend.main:app --reload --port 8001
```

ブラウザで `http://localhost:8001` を開いて利用します。

## 処理フロー

1. PNGアップロード
2. 候補マスクを複数生成
3. 画像クリックで最適候補を選択してレイヤー化
4. 必要に応じてのりしろ補完
5. PSD書き出し

## API概要

- `POST /api/session` : PNGアップロードでセッション作成 + 候補生成
- `GET /api/session/{id}/candidates` : 候補マスクプレビュー取得
- `POST /api/session/{id}/mask` : クリック座標で候補を1つ選んでレイヤー追加
- `POST /api/session/{id}/inpaint` : マスク統合領域ののりしろ補完
- `POST /api/session/{id}/export` : PSDダウンロード
- `DELETE /api/session/{id}` : セッション削除

## 補足

- セッションは `sessions` 辞書でメモリ管理しています（永続化なし）。
- `backend/services/sam2_service.py` の `generate_candidates` 内にSAM2本実装を接続できます。
