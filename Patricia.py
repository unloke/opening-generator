import chess
import chess.engine
import chess.pgn
import os
import time
import requests
import statistics  # 引入 statistics 模組
from functools import wraps
from typing import Dict, List, Optional, Tuple
from maia2 import model, inference

# === 參數設定 ===
# 請根據您的系統路徑修改以下設定
stockfish_path   = r'D:\python\my chess game code\stockfish\stockfish-windows-x86-64-avx2.exe'
patricia_path    = r'D:\吳冠頡\opening generator\patricia_4_dev_4cb_64_ja_avx2_zen2.exe'

# 引擎分析參數
BASE_ANALYSIS_DEPTH = 16  # 用於計算複雜度的引擎分析深度
pat_engine_depth = 20  # Patricia 引擎的分析深度
MULTIPV_FOR_COMPLEXITY = 3 # 計算複雜度時需要的多PV數量

# 複雜度與深度控制參數
ABSOLUTE_MAX_PLY = 40               # 安全閥，不論局面多複雜，總回合數到此一定停止

# 其他參數
advantage_threshold_base = 150  # 局面優勢超過這個值(centipawns)就停止延伸
branching_probability_threshold = 0.1  # Lichess 資料庫和 Maia2 中，著法出現機率低於此值則不考慮
# 若 Lichess 總對局數低於此閾值，改用 Maia2 推論
LICHESS_GAME_THRESHOLD = 1000

# 用於快取每個局面的候選走法，確保重複局面產生一致結果
position_cache: Dict[str, List[chess.Move]] = {}

# === 工具函式 ===
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"\n▶️ 函式 {func.__name__} 執行時間: {time.time() - start_time:.2f} 秒")
        return result
    return wrapper

def check_engine_path(path: str, name: str):
    if not os.path.exists(path):
        print(f"錯誤：找不到 {name} 執行檔：{path}")
        exit(1)

def parse_position_input(board: chess.Board, game: chess.pgn.Game, inp: str):
    node = game
    try:
        if inp.startswith('startpos moves'):
            moves = inp.split()[2:]
            if not moves:
                game.setup(board)
                return node
            for uci in moves:
                mv = chess.Move.from_uci(uci)
                board.push(mv)
                node = node.add_main_variation(mv)
        elif inp.startswith('fen'):
            parts = inp[4:].split(' moves ', 1)
            board.set_fen(parts[0].strip())
            game.setup(board)
            if len(parts) > 1:
                for uci in parts[1].split():
                    mv = chess.Move.from_uci(uci)
                    board.push(mv)
                    node = node.add_main_variation(mv)
        else:
            print("錯誤：無效的局面輸入格式。")
            exit(1)
    except ValueError as e:
        print(f"錯誤：解析局面或著法時出錯 - {e}")
        exit(1)
    return node

def get_stockfish_analysis(engine, board, depth, multipv):
    safe_depth = max(12, min(depth, 35))
    return engine.analyse(board, chess.engine.Limit(depth=safe_depth), info=chess.engine.INFO_ALL, multipv=multipv)

def evaluate_score_from_info(info: dict, pov_white: bool) -> int:
    score = info.get('score')
    if score is None: return 0
    pov_score = score.pov(pov_white)
    return pov_score.score(mate_score=100000)

# === 複雜度計算與決策函式 ===

def calculate_complexity_score(infos: List[chess.engine.InfoDict], board: chess.Board) -> float:
    """
    根據建議的公式 C 計算局面複雜度分數。
    C = 0.15 * |M| + 4 * (seldepth - depth) + σ_PV
    """
    if not infos:
        return 0.0

    # 1. 合法走法數 |M|
    num_legal_moves = board.legal_moves.count()
    c_moves = 0.15 * num_legal_moves

    # 2. 搜尋深度差 (seldepth - depth)
    first_info = infos[0]
    depth = first_info.get('depth', BASE_ANALYSIS_DEPTH)
    seldepth = first_info.get('seldepth', depth)
    c_seldepth = 4.0 * (seldepth - depth)

    # 3. 多主變評分波動 σ_PV
    c_sigma_pv = 0.0
    if len(infos) >= 2:
        scores = []
        for info in infos:
            pov_score = info['score'].pov(board.turn)
            if pov_score.is_mate():
                # 將 mate 轉換為一個固定的高分 (e.g., 1500cp) 以避免標準差失真
                score_val = 1500 * (1 if pov_score.mate() > 0 else -1)
            else:
                score_val = pov_score.score()
            scores.append(score_val if score_val is not None else 0)
        
        if len(scores) >= 2:
            c_sigma_pv = statistics.pstdev(scores)

    complexity = c_moves + c_seldepth + c_sigma_pv
    return complexity

def has_clear_advantage(infos: List[chess.engine.InfoDict], pov_white: bool, base_score: int) -> bool:
    if not infos: return False
    current_score = evaluate_score_from_info(infos[0], pov_white)
    threshold = advantage_threshold_base
    should_stop = (current_score - base_score) >= threshold
    if should_stop:
        print(f"  └─ 達到優勢閾值: {current_score - base_score}cp >= {threshold}cp (基準分:{base_score})，停止分支。")
    return should_stop

def select_best_move_for_us(pat_engine: chess.engine.SimpleEngine, board: chess.Board) -> Optional[chess.Move]:
    try:
        res = pat_engine.play(board, chess.engine.Limit(depth=pat_engine_depth))
        return res.move
    except chess.engine.EngineError as e:
        print(f"警告: Patricia 引擎在局面 {board.fen()} 中出錯: {e}")
        return None

def get_lichess_top_moves(fen: str, num: int = 10) -> Tuple[List[Dict], int]:
    params = {'fen': fen, 'variant': 'standard', 'speeds': 'blitz,rapid,classical', 'ratings': '2000,2200,2500'}
    try:
        r = requests.get('https://explorer.lichess.ovh/lichess', params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        total_games = data.get('white', 0) + data.get('draws', 0) + data.get('black', 0)
        moves_data = data.get('moves', [])
        if not moves_data or total_games == 0:
            return [], total_games
        for m in moves_data:
            m['probability'] = (m.get('white', 0) + m.get('draws', 0) + m.get('black', 0)) / total_games
        sorted_moves = sorted(moves_data, key=lambda x: x['probability'], reverse=True)[:num]
        return sorted_moves, total_games
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"警告: Lichess API 獲取失敗: {e}")
        return [], 0

def get_maia2_candidate_moves(
    board: chess.Board, 
    maia2_model, 
    prepared_inference, 
    opponent_elo: int
) -> List[chess.Move]:
    """使用 Maia2 獲取候選著法"""
    try:
        # 假設玩家等級與對手相同，您可以根據需要調整
        player_elo = opponent_elo
        
        # 進行 Maia2 推理
        move_probs, win_prob = inference.inference_each(
            maia2_model, 
            prepared_inference, 
            board.fen(), 
            player_elo, 
            opponent_elo
        )
        
        # 篩選機率高於閾值的著法
        candidate_moves = []
        for move_uci, prob in move_probs.items():
            if prob > branching_probability_threshold:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        candidate_moves.append((move, prob))
                except ValueError:
                    continue
        
        # 按機率排序並返回著法列表
        candidate_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, prob in candidate_moves]
        
    except Exception as e:
        print(f"警告: Maia2 推理出錯: {e}")
        return []

def get_opponent_candidate_moves(
    board: chess.Board,
    infos: List[chess.engine.InfoDict],
    maia2_model,
    prepared_inference,
    opponent_elo: int,
    allow_branching: bool
) -> List[chess.Move]:
    if not infos or 'pv' not in infos[0] or not infos[0]['pv']:
        return []

    sf_best_move = infos[0]['pv'][0]
    candidate_moves = {sf_best_move}

    if not allow_branching:
        return [sf_best_move]

    # 優先嘗試從 Lichess 獲取著法
    fen = board.fen()
    if fen in position_cache:
        return position_cache[fen]

    lichess_moves, total_games = get_lichess_top_moves(fen)
    if lichess_moves and total_games >= LICHESS_GAME_THRESHOLD:
        for move_data in lichess_moves:
            try:
                candidate_moves.add(chess.Move.from_uci(move_data['uci']))
            except ValueError:
                continue
    else:
        # Lichess 資料不足時，使用 Maia2 作為備選
        maia2_moves = get_maia2_candidate_moves(board, maia2_model, prepared_inference, opponent_elo)
        for move in maia2_moves:
            candidate_moves.add(move)

    final_list = [m for m in candidate_moves if m != sf_best_move]
    final_list = sorted(final_list, key=lambda m: m.uci())
    final_list.insert(0, sf_best_move)
    position_cache[fen] = final_list
    return final_list

def end_variation_if_our_turn(board, node, pov_white, pat_engine):
    if (board.turn == chess.WHITE) == pov_white and not board.is_game_over():
        mv = select_best_move_for_us(pat_engine, board)
        if mv and mv in board.legal_moves:
            node.add_variation(mv)

# === 主遞迴函式 ===
@measure_time
def _generate_tree_recursively(
    board: chess.Board,
    node: chess.pgn.GameNode,
    ply: int,
    initial_target_ply: int,
    absolute_max_ply: int,
    pov_white: bool,
    sf_engine: chess.engine.SimpleEngine,
    maia2_model,
    prepared_inference,
    opponent_elo: int,
    pat_engine: chess.engine.SimpleEngine,
    base_score: int,
    is_main_line: bool
):
    # 步驟 1: 檢查硬性停止條件
    if board.is_game_over():
        print(f"ply:{ply:2d} | 遊戲結束，停止分支。")
        return
    if ply >= absolute_max_ply:
        print(f"ply:{ply:2d} | 達到絕對深度上限 {absolute_max_ply}，強制停止分支。")
        end_variation_if_our_turn(board, node, pov_white, pat_engine)
        return

    # 步驟 2: 進行局面分析，計算複雜度分數
    analysis_infos = get_stockfish_analysis(sf_engine, board, BASE_ANALYSIS_DEPTH, MULTIPV_FOR_COMPLEXITY)
    if not analysis_infos:
        print(f"ply:{ply:2d} | 引擎分析失敗，停止分支。")
        return

    complexity_score = calculate_complexity_score(analysis_infos, board)
    #node.comment = f"C: {complexity_score:.1f}"
    print(f"ply:{ply:2d} | 局面複雜度 C = {complexity_score:.1f}")

    # 步驟 3: 檢查基於複雜度的軟性停止條件
    if ply >= initial_target_ply and complexity_score < COMPLEXITY_THRESHOLD_STABLE:
        print(f"  └─ 在深度 {ply} (已達基礎目標 {initial_target_ply}) 局面穩定 (C < {COMPLEXITY_THRESHOLD_STABLE})，停止分支。")
        end_variation_if_our_turn(board, node, pov_white, pat_engine)
        return

    # 步驟 4: 檢查優勢停止條件
    if has_clear_advantage(analysis_infos, pov_white, base_score):
        end_variation_if_our_turn(board, node, pov_white, pat_engine)
        return

    # 步驟 5: 如果沒有停止，則繼續遞迴
    if (board.turn == chess.WHITE) == pov_white:
        my_move = select_best_move_for_us(pat_engine, board)
        if my_move and my_move in board.legal_moves:
            board.push(my_move)
            next_node = node.add_main_variation(my_move)
            _generate_tree_recursively(
                board, next_node, ply + 1, initial_target_ply, absolute_max_ply, pov_white,
                sf_engine, maia2_model, prepared_inference, opponent_elo, pat_engine, base_score, is_main_line
            )
            board.pop()
        else:
            print(f"警告: Patricia 在局面 {board.fen()} 未能提供合法著法。")
    else:
        candidate_moves = get_opponent_candidate_moves(
            board, analysis_infos, maia2_model, prepared_inference, opponent_elo, is_main_line
        )
        if not candidate_moves: return

        main_move = candidate_moves[0]
        board.push(main_move)
        next_node = node.add_main_variation(main_move)
        current_score = evaluate_score_from_info(analysis_infos[0], pov_white)
        _generate_tree_recursively(
            board, next_node, ply + 1, initial_target_ply, absolute_max_ply, pov_white,
            sf_engine, maia2_model, prepared_inference, opponent_elo, pat_engine, current_score, is_main_line
        )
        board.pop()

        if is_main_line and len(candidate_moves) > 1:
            score_before_branching = current_score
            for move in candidate_moves[1:]:
                board.push(move)
                var_node = node.add_variation(move)
                _generate_tree_recursively(
                    board, var_node, ply + 1, initial_target_ply, absolute_max_ply, pov_white,
                    sf_engine, maia2_model, prepared_inference, opponent_elo, pat_engine, score_before_branching, False
                )
                board.pop()

# === 主程式入口 ===
if __name__ == "__main__":
    print("========= 智慧動態開局準備產生器  =========")
    pos_in = input("局面輸入 (e.g., 'startpos moves e2e4 e7e5'): ").strip()
    col = ""
    while col not in ("white", "black"):
        col = input("你的顏色 (white/black)：").strip().lower()

    pov_white = (col == "white")

    base_depth_ply = 0
    while base_depth_ply <= 0:
        try:
            base_depth_ply = int(input(f"基礎準備深度 (ply, 達到此深度後遇穩定局面即停止, 建議 8-16)："))
        except ValueError:
            base_depth_ply = 0
    
    opponent_elo = 0
    while not (1000 <= opponent_elo <= 3000):
        try:
            opponent_elo = int(input("對手 Elo 等級分 (1000-3000, 用於 Maia2 推理)："))
        except ValueError:
            opponent_elo = 0

    # 檢查執行檔
    check_engine_path(stockfish_path, "Stockfish")
    check_engine_path(patricia_path, "Patricia")

    # 載入 Maia2 模型
    print("\n正在載入 Maia2 模型...")
    try:
        maia2_model = model.from_pretrained(type="rapid", device="cpu")  # 可改為 "gpu"
        prepared_inference = inference.prepare()
        print("Maia2 模型載入成功。")
    except Exception as e:
        print(f"錯誤：無法載入 Maia2 模型: {e}")
        exit(1)

    # 啟動引擎
    pat_cmd = [patricia_path]

    print("\n正在啟動引擎...")
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf,\
         chess.engine.SimpleEngine.popen_uci(pat_cmd) as pat:

        sf.configure({"UCI_ShowWDL": True, "Threads": os.cpu_count() // 2 or 1})

        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "AI Opening Preparation (Maia2)"
        game.headers["Site"] = "Local"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        
        node = parse_position_input(board, game, pos_in)
        if pos_in.startswith('fen'):
            game.headers["FEN"] = board.fen()
            game.headers["SetUp"] = "1"

        print("\n正在分析初始局面...")
        init_infos = get_stockfish_analysis(sf, board, BASE_ANALYSIS_DEPTH, MULTIPV_FOR_COMPLEXITY)
        if not init_infos:
            print("錯誤：無法分析初始局面，程式終止。")
            exit(1)
            
        init_complexity = calculate_complexity_score(init_infos, board)
        base_score = evaluate_score_from_info(init_infos[0], pov_white)

        print(f"初始局面分析完畢，基礎分數: {base_score/100:.2f}。")
        print(f"初始複雜度 C = {init_complexity:.1f}")
        COMPLEXITY_THRESHOLD_STABLE = init_complexity
        print(f"設定穩定複雜度閾值 C < {COMPLEXITY_THRESHOLD_STABLE:.1f}。")
        print("開始生成智慧決策控制的開局樹...")
        start_time = time.time()

        current_ply = board.ply()
        initial_target_ply = current_ply + base_depth_ply
        absolute_max_ply = current_ply + ABSOLUTE_MAX_PLY
        print(f"設定 -> 基礎目標深度: {initial_target_ply} ply, 絕對上限深度: {absolute_max_ply} ply")

        _generate_tree_recursively(
            board, node, current_ply, initial_target_ply, absolute_max_ply,
            pov_white, sf, maia2_model, prepared_inference, opponent_elo, pat, base_score, True
        )

        print(f"\n✅ 開局樹生成完畢，總耗時: {time.time() - start_time:.2f} 秒。")

        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        pgn = game.accept(exporter)
        print("\n--- PGN 結果 ---")
        print(pgn)