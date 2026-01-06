# URL短縮 Cloudflare Worker

html-preview用のURL短縮プロキシ。

## セットアップ

### 1. Cloudflareアカウント作成
https://dash.cloudflare.com/sign-up

### 2. wranglerインストール
```bash
npm install -g wrangler
```

### 3. ログイン
```bash
wrangler login
```

### 4. デプロイ
```bash
cd worker
npm install
npm run deploy
```

### 5. URLを取得
デプロイ完了後、以下のようなURLが表示される：
```
https://url-shortener.<your-account>.workers.dev
```

### 6. html-previewに設定
`index.html` の `SHORTENER_URL` を更新：
```javascript
const SHORTENER_URL = 'https://url-shortener.<your-account>.workers.dev';
```

## ローカルテスト
```bash
npm run dev
```

## 無料枠
- 10万リクエスト/日
- 個人利用なら十分
