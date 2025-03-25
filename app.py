# まず、グローバルに verify_with_tavily を定義
def verify_with_tavily(radius, wind_direction, water_volume):
    if not TAVILY_TOKEN:
        return ["Tavilyのトークンが設定されていないため、検証できません。"]
    try:
        url = "https://api.tavily.com/search"
        query = "火災 拡大半径 一般的"
        payload = {
            "query": query,
            "topic": "fire",
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": 1,
            "time_range": None,
            "days": 3,
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
            "include_image_descriptions": False,
            "include_domains": [],
            "exclude_domains": []
        }
        headers = {
            "Authorization": f"Bearer {TAVILY_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        messages = []
        if "answer" in result and result["answer"]:
            messages.append(f"Tavily検索結果: {result['answer']}")
        else:
            messages.append("Tavily検索結果が見つかりませんでした。元のデータを使用します。")
        return messages
    except Exception as e:
        st.error(f"Tavily検証中にエラーが発生しました: {e}")
        return ["Tavily検証中にエラーが発生しました。"]

# その後、run_simulation 関数内で verify_with_tavily を呼び出す
def run_simulation(duration_hours, time_label):
    if not st.session_state.get("weather_data"):
        st.error("気象データが取得されていません。")
        return
    if not st.session_state.get("points"):
        st.error("発生地点が設定されていません。")
        return

    # (中略)

    # ここで verify_with_tavily を呼び出す
    verification_msgs = verify_with_tavily(radius_m, st.session_state.weather_data.get("winddirection", 0), water_volume_tons)
    st.write("#### Tavily 検証結果")
    for msg in verification_msgs:
        st.write(msg)

# 以降、その他のコード...
