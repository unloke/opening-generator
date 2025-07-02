import chess
import chess.engine
import chess.pgn
import os
import time
import requests
from functools import wraps

# === 參數設定 ===
stockfish_path   = r'D:\python\my chess game code\stockfish\stockfish-windows-x86-64-avx2.exe'
lc0_path         = r'D:\吳冠頡\ChessGPT\lc0\lc0.exe'  # 只給 Maia 用
model_directory  = r'D:\吳冠頡\opening generator\weights'
patricia_path    = r'D:\吳冠頡\opening generator\patricia_4_dev_4cb_64_ja_avx2_zen2.exe'

stockfish_depth_main      = 20
stockfish_depth_variation = 20 

advantage_threshold_base    = 150 # 局面優勢超過這個值(centipawns)就停止延伸
branching_probability_threshold = 0.1 # Lichess 資料庫中，著法出現機率低於此值則不考慮

# === 工具函式 ===
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"\n▶️ 函式 {func.__name__} 執行時間: {time.time()-start:.2f} 秒")
        return result
    return wrapper

def check_engine_path(path: str, name: str):
    if not os.path.exists(path):
        print(f"錯誤：找不到 {name} 執行檔：{path}")
        exit(1)

def check_model_path(path: str, name: str):
    if not os.path.exists(path):
        print(f"錯誤：找不到 {name} 檔案：{path}")
        exit(1)

def get_maia_model_path(elo: int) -> str:
    maia_levels = [
        (1100,'maia-1100.pb'), (1200,'maia-1200.pb'),
        (1300,'maia-1300.pb'), (1400,'maia-1400.pb'),
        (1500,'maia-1500.pb'), (1600,'maia-1600.pb'),
        (1700,'maia-1700.pb'), (1800,'maia-1800.pb'),
        (1900,'maia-1900.pb'),
    ]
    fn = min(maia_levels, key=lambda x:abs(x[0]-elo))[1]
    if not (fn.endswith('.pb') or fn.endswith('.pb.gz')):
        fn = fn.replace('.pb','.pb.gz')
    return os.path.join(model_directory, fn)

def parse_position_input(board: chess.Board, game: chess.pgn.Game, inp: str):
    node = game
    try:
        if inp.startswith('startpos moves'):
            moves = inp.split()[2:]
            if not moves: # 處理 "startpos" 或 "startpos moves" 的情況
                game.setup(board)
                return node
            for uci in moves:
                mv = chess.Move.from_uci(uci)
                board.push(mv)
                node = node.add_main_variation(mv)
        elif inp.startswith('fen'):
            parts = inp[4:].split(' moves ',1)
            board.set_fen(parts[0].strip())
            game.setup(board)
            if len(parts)>1:
                for uci in parts[1].split():
                    mv = chess.Move.from_uci(uci)
                    board.push(mv)
                    node = node.add_main_variation(mv)
        else:
            print("錯誤：無效的局面輸入格式。請使用 'startpos moves ...' 或 'fen ... moves ...'")
            exit(1)
    except ValueError as e:
        print(f"錯誤：解析局面或著法時出錯 - {e}")
        exit(1)
    return node

def get_stockfish_analysis(engine, board, depth, multipv):
    return engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        info=chess.engine.INFO_ALL,
        multipv=multipv
    )

def evaluate_score_from_info(info: dict, board: chess.Board, pov_is_white: bool) -> int:
    score = info.get('score')
    if score is None:
        return 0 # 如果沒有分數資訊，回傳 0
    return score.pov(pov_is_white).score(mate_score=100000)

def has_clear_advantage(sf_engine, board, pov_white, base_score):
    infos = get_stockfish_analysis(sf_engine, board, depth=stockfish_depth_variation, multipv=1)
    if not infos: return False
    cur = evaluate_score_from_info(infos[0], board, pov_white)
    return (cur - base_score) >= advantage_threshold_base

# === 重構與優化部分 ===

# 1. 簡化 `select_best_move_for_us`，移除無用參數
def select_best_move_for_us(
    patricia_engine: chess.engine.SimpleEngine,
    board:           chess.Board
) -> chess.Move | None:
    """直接請 Patricia 建議「最佳走法」"""
    try:
        res = patricia_engine.play(board, chess.engine.Limit(depth=stockfish_depth_main))
        return res.move
    except chess.engine.EngineError as e:
        print(f"警告: Patricia 引擎在局面 {board.fen()} 中出錯: {e}")
        return None

# 2. 增強 `get_lichess_top_moves` 的穩定性
def get_lichess_top_moves(fen: str, num: int = 10) -> list[dict]:
    """從 Lichess API 取得熱門著法，並加入錯誤處理。"""
    params = {
        'fen':fen, 'variant':'standard',
        'speeds':'blitz,rapid,classical',
        'ratings':'2000,2200,2500',
        'topGames':0, 'recentGames':0
    }
    try:
        r = requests.get('https://explorer.lichess.ovh/lichess', params=params, timeout=5)
        r.raise_for_status()  # 如果請求不成功 (如 404, 500)，會拋出例外
        data = r.json()
        total = data.get('white',0)+data.get('draws',0)+data.get('black',0)
        moves = data.get('moves',[])
        if not moves: return [] # 如果 API 回傳空的 moves 列表

        # 如果頂層沒有統計數據，則從 moves 內部加總
        if total == 0:
            total = sum(m.get('white',0)+m.get('draws',0)+m.get('black',0) for m in moves)
        
        if total == 0: return [] # 避免除以零

        for m in moves:
            games = m.get('white',0)+m.get('draws',0)+m.get('black',0)
            m['probability'] = games / total
        
        return sorted(moves, key=lambda x:x['probability'], reverse=True)[:num]
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"警告: 無法從 Lichess API 取得資料 (FEN: {fen}). 錯誤: {e}")
        return []

# 3. 【核心改動】新增函式以實現 Lichess -> Maia 的備用邏輯
def get_opponent_candidate_moves(
    board: chess.Board,
    sf_engine: chess.engine.SimpleEngine,
    maia_engine: chess.engine.SimpleEngine
) -> list[chess.Move]:
    """
    取得對手的候選著法列表。
    策略：
    1. Stockfish 的最佳著法永遠是主線，必須包含。
    2. 優先嘗試從 Lichess API 取得熱門人類著法。
    3. 如果 Lichess API 無回傳 (網路問題、無此局面資料)，則改用 Maia 引擎模擬人類著法作為備案。
    """
    candidate_moves = []
    
    # 步驟 1: 取得 Stockfish 最佳著法
    sf_analysis = get_stockfish_analysis(sf_engine, board, stockfish_depth_variation, 1)
    if not sf_analysis or 'pv' not in sf_analysis[0] or not sf_analysis[0]['pv']:
        print(f"警告: Stockfish 未能為局面 {board.fen()} 提供著法。")
        return []
    
    sf_best_move = sf_analysis[0]['pv'][0]
    candidate_moves.append(sf_best_move)

    # 步驟 2: 嘗試 Lichess API
    lichess_moves_data = get_lichess_top_moves(board.fen(), 10)

    if lichess_moves_data:
        for move_data in lichess_moves_data:
            if move_data['probability'] < branching_probability_threshold:
                continue
            move = chess.Move.from_uci(move_data['uci'])
            if move in board.legal_moves and move not in candidate_moves:
                candidate_moves.append(move)
        return candidate_moves

    # 步驟 3: Lichess 無資料，使用 Maia 作為備案
    try:
        maia_analysis = maia_engine.analyse(
            board,
            chess.engine.Limit(nodes=1), # Lc0/Maia 使用 nodes=1 即可取得策略網路的輸出
            multipv=1
        )
        for info in maia_analysis:
            move = info['pv'][0]
            if move in board.legal_moves and move not in candidate_moves:
                candidate_moves.append(move)
    except chess.engine.EngineError as e:
        print(f"警告: Maia 引擎在局面 {board.fen()} 中出錯: {e}")

    return candidate_moves


def end_variation_if_our_turn(board, node, pov_white, pat_engine):
    """如果輪到我方下且遊戲未結束，則由 Patricia 走一步來結束這個分支。"""
    if (board.turn == chess.WHITE) == pov_white and not board.is_game_over():
        mv = select_best_move_for_us(pat_engine, board)
        if mv and mv in board.legal_moves:
            # 這裡不使用 add_main_variation，因為它是在一個分支的結尾
            node.add_variation(mv)

# 4. 重構遞迴函式，使其更簡潔
@measure_time
def _generate_tree_recursively(
    board: chess.Board,
    node: chess.pgn.GameNode,
    ply: int,
    max_ply: int,
    pov_white: bool,
    sf_engine, maia_engine, pat_engine,
    base_score: int
):
    if board.is_game_over() or ply >= max_ply or has_clear_advantage(sf_engine, board, pov_white, base_score):
        end_variation_if_our_turn(board, node, pov_white, pat_engine)
        return
    if (board.turn == chess.WHITE) == pov_white:
        my_move = select_best_move_for_us(pat_engine, board)
        if my_move and my_move in board.legal_moves:
            board.push(my_move)
            next_node = node.add_main_variation(my_move)
            _generate_tree_recursively(
                board, next_node, ply + 1, max_ply, pov_white,
                sf_engine, maia_engine, pat_engine, base_score
            )
            board.pop()
        else:
            print(f"警告: Patricia 未能為局面 {board.fen()} 提供合法著法。")

    else:
        candidate_moves = get_opponent_candidate_moves(board, sf_engine, maia_engine)
        if not candidate_moves:
            return

        score_before_branching = evaluate_score_from_info(
            get_stockfish_analysis(sf_engine, board, stockfish_depth_variation, 1)[0],
            board, pov_white
        )

        is_main_line = True
        for move in candidate_moves:
            if move not in board.legal_moves:
                print(f"警告: 候選著法 {move.uci()} 在局面 {board.fen()} 中並非合法著法，已跳過。")
                continue

            board.push(move)

            if is_main_line:
                next_node = node.add_main_variation(move)
                is_main_line = False 
                current_score = evaluate_score_from_info(
                    get_stockfish_analysis(sf_engine, board, stockfish_depth_variation, 1)[0],
                    board, pov_white
                )
                _generate_tree_recursively(
                    board, next_node, ply + 1, max_ply, pov_white,
                    sf_engine, maia_engine, pat_engine, current_score
                )
            else:
                var_node = node.add_variation(move)
                _generate_tree_recursively(
                    board, var_node, ply + 1, max_ply, pov_white,
                    sf_engine, maia_engine, pat_engine, score_before_branching
                )
            
            board.pop() 

# === 主程式入口 ===
if __name__ == "__main__":
    print("========= Tricky 開局準備產生器 (v2: Lichess/Maia Fallback) =========")
    pos_in = input("局面輸入 (e.g., 'startpos moves e2e4 e7e5'): ").strip()
    col = ""
    while col not in ("white","black"):
        col = input("你的顏色 (white/black)：").strip().lower()

    pov_white = (col == "white")
    
    depth_ply = 0
    while depth_ply <= 0:
        try:
            depth_ply = int(input("主準備深度 (ply, e.g., 10)："))
        except ValueError:
            depth_ply = 0

    opponent_elo = 0
    while not (1000 <= opponent_elo <= 3000):
        try:
            opponent_elo = int(input("對手 Maia 等級分 (1000-3000)："))
        except ValueError:
            opponent_elo = 0

    # 檢查執行檔與模型
    check_engine_path(stockfish_path, "Stockfish")
    check_engine_path(lc0_path,      "Lc0")
    check_engine_path(patricia_path, "Patricia")
    maia_model = get_maia_model_path(opponent_elo)
    check_model_path(maia_model, f"Maia 模型 (ELO {opponent_elo})")

    # 啟動各引擎
    pat_cmd = [patricia_path]
    lc0_maia_cmd = [lc0_path, f'--weights={maia_model}']

    print("\n正在啟動引擎...")
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf,\
         chess.engine.SimpleEngine.popen_uci(lc0_maia_cmd)   as maia,\
         chess.engine.SimpleEngine.popen_uci(pat_cmd)        as pat:

        sf.configure({"UCI_ShowWDL": True, "Threads": os.cpu_count() // 2 or 1})
        # Lc0/Maia 不需要太多線程
        maia.configure({"Threads": 2})

        board = chess.Board()
        game  = chess.pgn.Game()
        game.headers["FEN"] = board.fen()
        
        node = parse_position_input(board, game, pos_in)
        if pos_in.startswith('fen'): # 如果是 FEN 開局，更新 Header
            game.headers["FEN"] = board.fen()
            game.headers["SetUp"] = "1"


        print("正在分析初始局面...")
        init_info = get_stockfish_analysis(sf, board, stockfish_depth_main, 1)
        base_score = evaluate_score_from_info(init_info[0], board, pov_white) if init_info else 0
        
        print(f"初始局面分析完畢，基礎分數: {base_score/100:.2f}。開始生成開局樹...")
        start_time = time.time()

        _generate_tree_recursively(
            board, node, board.ply(), board.ply() + depth_ply,
            pov_white, sf, maia, pat, base_score
        )

        print(f"\n✅ 開局樹生成完畢，總耗時: {time.time() - start_time:.2f} 秒。")

        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=False)
        pgn = game.accept(exporter)
        print(pgn)