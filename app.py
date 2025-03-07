import requests
import json

# APIキーの設定
api_key = 'YOUR_GEMINI_API_KEY'  # ここに自身のAPIキーを入力してください

# エンドポイントURL
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

# ヘッダー情報
headers = {
    'Content-Type': 'application/json',
}

# リクエストデータ
data = {
    "contents": [{
        "parts": [{"text": "Explain how AI works"}]
    }]
}

# POSTリクエストの送信
response = requests.post(url, headers=headers, data=json.dumps(data))

# レスポンスの表示
if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2))
else:
    print(f"リクエストに失敗しました。ステータスコード: {response.status_code}")
    print(response.text)
