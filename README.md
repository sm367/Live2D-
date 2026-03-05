# PNG1枚からLive2D用PSDを生成するWebツール

FastAPIベースのローカルWebツールです。1枚のPNGをアップロードし、クリック選択でパーツを切り出し、OpenCVでのりしろ補完後にPSDへ書き出します。

## 構成

- `backend/main.py` : APIサーバー本体（メモリセッション管理）
- `backend/services/sam2_service.py` : クリック選択マスク生成（SAM2互換インターフェース / GrabCutフォールバック）
- `backend/services/inpaint_service.py` : のりしろ補完（OpenCV inpaint）
- `backend/services/psd_service.py` : `psd-tools` を使ったPSD出力
- `frontend/index.html`, `frontend/app.js` : フロントエンド

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 起動

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

ブラウザで `http://localhost:8000` を開いて利用します。

## API概要

- `POST /api/session` : PNGアップロードでセッション作成
- `POST /api/session/{id}/mask` : クリック座標でマスク追加
- `POST /api/session/{id}/inpaint` : マスク統合領域ののりしろ補完
- `POST /api/session/{id}/export` : PSDダウンロード
- `DELETE /api/session/{id}` : セッション削除

## 補足

- セッションは `sessions` 辞書でメモリ管理しています（永続化なし）。
- `SAM2Service` は実運用でSAM2推論に差し替え可能な形にしてあります。
