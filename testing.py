import chess
import chess.engine

# 设置 Lc0 引擎的路径
engine_path = r'D:\吳冠頡\ChessGPT\lc0\lc0.exe'  # 请将此路径替换为您本地 lc0 可执行文件的路径


# 初始化棋局
board = chess.Board()

# 啟動引擎
with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
    # 設定引擎選項以啟用 WDL
    engine.configure({"UCI_ShowWDL": True})

    # 分析棋局，設定思考時間為 1 秒
    print("正在分析棋局...")
    result = engine.analyse(board, chess.engine.Limit(depth = 10))

    # 打印返回的完整資訊，幫助調試
    print("分析結果:", result)

    # 嘗試獲取 WDL 資訊
    wdl = result.get("wdl")  # 嘗試從結果中提取 WDL
    if wdl is None:
        # 若沒有直接的 WDL，從 "info" 或其他欄位中查找
        info = result.get("info", {})
        wdl = info.get("wdl", None)

    if wdl:
        # 如果成功獲取 WDL，計算勝和負比例
        total = sum(wdl)
        win_rate = wdl[0] / total
        draw_rate = wdl[1] / total
        loss_rate = wdl[2] / total