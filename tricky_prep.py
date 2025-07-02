# tricky_prep.py

import chess
import chess.engine
import chess.pgn
import os
import time
from functools import wraps

# ==============================
# 全域參數 (可自行調整)
# ==============================
main_line_depth = 20  # 主線最大深度 
side_line_depth_multiplier = 2 # 支線深度倍率 
stockfish_path = r'D:\吳冠頡\ChessGPT\stockfish-windows-x86-64-avx2.exe'
lc0_path = r'D:\吳冠頡\ChessGPT\lc0\lc0.exe'
model_directory = r'D:\吳冠頡\opening generator\weights'
stockfish_depth_main = 20 # 主線 Stockfish 分析深度 
stockfish_depth_variation = 20 # 支線 Stockfish 分析深度 
multipv_main = 5 # 主線 Stockfish 多變化數量 
multipv_variation = 1 # 支線 Stockfish 多變化數量 
advantage_threshold_base = 150 # 顯著優勢基礎閾值 
candidate_move_tolerance = 20 # 候選步與最佳分的容忍度 
move_selection_iterations = 5 # 我方選步模擬回合數 
branching_score_diff_threshold = 0 # 產生支線的評分差異閾值 (降低以更積極分支，提高以更保守分支)

# ==============================
# 裝飾器：計算函式執行時間
# ==============================
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"\n{func.__name__} 執行時間: {end - start:.2f} 秒")
        return result
    return wrapper

# ==============================
# 基本路徑與檔案檢查函式
# ==============================
def check_engine_path(engine_path: str):
    if not os.path.exists(engine_path):
        print("無法找到指定的引擎執行檔：")
        print(f"  {engine_path}")
        print("請確認路徑是否正確，並再次執行程式。")
        exit()

def check_model_path(model_path: str):
    if not os.path.exists(model_path):
        print("找不到對應的 Maia 模型檔案：")
        print(f"  {model_path}")
        print("請確認檔案是否存在，或調整路徑後重試。")
        exit()

def get_maia_model_path(opponent_elo: int) -> str:
    maia_levels = [
        (1100, 'maia-1100.pb'), (1200, 'maia-1200.pb'),
        (1300, 'maia-1300.pb'), (1400, 'maia-1400.pb'),
        (1500, 'maia-1500.pb'), (1600, 'maia-1600.pb'),
        (1700, 'maia-1700.pb'), (1800, 'maia-1800.pb'),
        (1900, 'maia-1900.pb'),
    ]
    closest_level = min(maia_levels, key=lambda x: abs(x[0] - opponent_elo))
    return os.path.join(model_directory, closest_level[1])

# ==============================
# 棋局初始化與解析
# ==============================
def parse_position_input(board: chess.Board, game: chess.pgn.Game, position_input: str) -> chess.pgn.GameNode:
    node = game
    if position_input.startswith('startpos moves'):
        moves = position_input[len('startpos moves '):].split()
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                print("偵測到非法走法：")
                print(f"  {move_uci}")
                print("請輸入正確的走法後重新執行程式。")
                exit()
            board.push(move)
            node = node.add_main_variation(move)
    elif position_input.startswith('fen'):
        fen_and_moves = position_input[len('fen '):].strip()
        parts = fen_and_moves.split(' moves ')
        fen = parts[0]
        board.set_fen(fen)
        game.setup(board)
        if len(parts) > 1:
            moves = parts[1].split()
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    print("偵測到非法走法：")
                    print(f"  {move_uci}")
                    print("請輸入正確的走法後重新執行程式。")
                    exit()
                board.push(move)
                node = node.add_main_variation(move)
    else:
        print("無效的局面輸入格式。")
        print("請使用下列其中一種形式：")
        print("  - startpos moves e2e4 e7e5 ...")
        print("  - fen <FEN字串> moves e2e4 ...")
        exit()
    return node

# ==============================
# 與Stockfish / Maia引擎互動函式
# ==============================
def get_stockfish_analysis(engine: chess.engine.SimpleEngine,
                           board: chess.Board,
                           depth: int,
                           multipv: int) -> list:
    return engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        info=chess.engine.INFO_ALL,
        multipv=multipv
    )

def get_maia_analysis(engine: chess.engine.SimpleEngine, board: chess.Board) -> dict:
    return engine.analyse(
        board,
        chess.engine.Limit(nodes=1),
        info=chess.engine.INFO_ALL
    )

def evaluate_score_from_info(info: dict, board: chess.Board, pov_is_white: bool) -> int:
    return info['score'].pov(pov_is_white).score(mate_score=100000)

# ==============================
# 顯著優勢判斷 (比較「目前評分」與「base_score」)
# ==============================
def has_clear_advantage(stockfish_engine: chess.engine.SimpleEngine,
                        board: chess.Board,
                        your_color: str,
                        base_score: int) -> bool:
    """
    與「base_score」做比較，若(目前評分 - base_score) >= 某門檻，則視為顯著提升，
    門檻會依主線或支線有所調整，並依子力差、中心控盤等條件輔助判斷。
    """
    pov_is_white = (your_color == 'white')
    info_list = get_stockfish_analysis(stockfish_engine, board, depth=stockfish_depth_variation, multipv=1)
    if not info_list:
        return False

    info = info_list[0]
    current_score = evaluate_score_from_info(info, board, pov_is_white)

    score_diff = current_score - base_score
    advantage_threshold = advantage_threshold_base

    # (1) 若評分相對提升(或下降) 絕對值 >= 門檻
    if abs(score_diff) >= advantage_threshold:
        print(f"[has_clear_advantage] 分差達 {score_diff} (>= {advantage_threshold:.0f}) -> True")
        return True

    # (2) 子力優勢明顯且可動步數少
    piece_map = board.piece_map()
    our_pieces = sum(1 for _, piece in piece_map.items() if piece.color == pov_is_white)
    their_pieces = sum(1 for _, piece in piece_map.items() if piece.color != pov_is_white)
    if (our_pieces > their_pieces and abs(score_diff) >= advantage_threshold * 0.6
        and len(list(board.legal_moves)) < 15):
        print(f"[has_clear_advantage] 子力多、分差 {score_diff}、可動步數少 -> True")
        return True

    # (3) 中心控制
    center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
    our_center_control = sum(1 for sq in center_squares
                             if board.is_attacked_by(pov_is_white, sq))
    if abs(score_diff) >= advantage_threshold * 0.8 and our_center_control >= 2:
        print(f"[has_clear_advantage] 中心控制佳、分差 {score_diff} (>= {advantage_threshold * 0.8:.0f}) -> True")
        return True

    return False

# ==============================
# 「我方」選擇最佳步驟 (多次往返: SF->Maia->SF->Maia->... )
# ==============================
def select_best_move_for_us(
    stockfish_engine: chess.engine.SimpleEngine,
    maia_engine: chess.engine.SimpleEngine,
    board: chess.Board,
    your_color: str,
    is_main_line: bool
) -> chess.Move:
    """
    從Stockfish多變化分析中，挑出與最佳評分差距小的候選步。
    對每個候選步模擬多回合(Maia->SF)，最後用SF評估局面，取評分最高者。
    """

    pov_is_white = (your_color == 'white')
    depth = stockfish_depth_main if is_main_line else stockfish_depth_variation
    multipv = multipv_main if is_main_line else multipv_variation

    # (A) 先用Stockfish取得多變化分析
    infos = get_stockfish_analysis(stockfish_engine, board, depth=depth, multipv=multipv)
    if not infos or 'pv' not in infos[0] or not infos[0]['pv']:
        return None

    best_score = evaluate_score_from_info(infos[0], board, pov_is_white)

    # (B) 篩選出與最佳分差小的候選
    candidate_moves = []
    for info in infos:
        if 'pv' not in info or not info['pv']:
            continue
        move = info['pv'][0]
        score = evaluate_score_from_info(info, board, pov_is_white)
        if (best_score - score) <= candidate_move_tolerance:
            candidate_moves.append((move, score))

    if not candidate_moves:
        return None
    if len(candidate_moves) == 1:
        return candidate_moves[0][0]

    # (C) 逐一模擬
    best_final_score = float('-inf')
    best_move = None

    for move, initial_score in candidate_moves:
        move_stack = []
        temp_board = board.copy()
        temp_board.push(move)
        move_stack.append(move)

        current_score = initial_score # 使用初始評分作為起點
        for _ in range(move_selection_iterations):
            if temp_board.is_game_over():
                break

            # Maia 的回應
            maia_info = get_maia_analysis(maia_engine, temp_board)
            maia_move = None
            if 'pv' in maia_info and maia_info['pv']:
                candidate = maia_info['pv'][0]
                if candidate in temp_board.legal_moves:
                    maia_move = candidate
            if maia_move is None:
                break
            temp_board.push(maia_move)
            move_stack.append(maia_move)

            if temp_board.is_game_over():
                break

            # Stockfish 的回應
            sf_infos = get_stockfish_analysis(stockfish_engine, temp_board, depth=depth, multipv=1)
            if not sf_infos or 'pv' not in sf_infos[0] or not sf_infos[0]['pv']:
                break
            sf_move = sf_infos[0]['pv'][0]
            if sf_move not in temp_board.legal_moves:
                break
            temp_board.push(sf_move)
            move_stack.append(sf_move)

        # 用 SF 評估最終局面
        if not temp_board.is_game_over():
            final_infos = get_stockfish_analysis(stockfish_engine, temp_board, depth=depth, multipv=1)
            if final_infos and 'score' in final_infos[0]:
                final_score = evaluate_score_from_info(final_infos[0], temp_board, pov_is_white)
            else:
                final_score = current_score # 若無法評估，則使用模擬前的評分
        else:
            final_score = current_score

        # 比較
        if final_score > best_final_score:
            best_final_score = final_score
            best_move = move

    return best_move

# ==============================
# 確保最後都在我方行棋後才結束的邏輯
# ==============================
def end_variation(board: chess.Board, node: chess.pgn.GameNode, your_color: str,
                  stockfish_engine: chess.engine.SimpleEngine,
                  maia_engine: chess.engine.SimpleEngine):
    pov_is_white = (your_color == 'white')
    turn_is_white = board.turn
    is_our_turn = (turn_is_white and pov_is_white) or (not turn_is_white and not pov_is_white)

    if is_our_turn and not board.is_game_over():
        move = select_best_move_for_us(stockfish_engine, maia_engine, board, your_color, is_main_line=False)
        if move:
            board.push(move)
            node.add_main_variation(move)
            board.pop()

# ==============================
# 產生「支線」的遞迴函式
# ==============================
@measure_time
def generate_variation(board: chess.Board,
                       node: chess.pgn.GameNode,
                       your_color: str,
                       stockfish_engine: chess.engine.SimpleEngine,
                       maia_engine: chess.engine.SimpleEngine,
                       current_depth: int,
                       side_line_depth: int,
                       base_score: int = None
                       ):
    """
    遞迴產生支線變化。
    """
    pov_is_white = (your_color == 'white')

    if base_score is None:
        info_list = get_stockfish_analysis(stockfish_engine, board, depth=stockfish_depth_variation, multipv=1)
        base_score = evaluate_score_from_info(info_list[0], board, pov_is_white) if info_list else 0

    if board.is_game_over() or current_depth >= side_line_depth \
       or has_clear_advantage(stockfish_engine, board, your_color, base_score):
        end_variation(board, node, your_color, stockfish_engine, maia_engine)
        return

    turn_is_white = board.turn
    is_our_turn = (turn_is_white and pov_is_white) or (not turn_is_white and not pov_is_white)

    if is_our_turn:
        move = select_best_move_for_us(stockfish_engine, maia_engine, board, your_color, is_main_line=False)
        if move is None:
            end_variation(board, node, your_color, stockfish_engine, maia_engine)
            return

        board.push(move)
        next_node = node.add_main_variation(move)

        generate_variation(
            board,
            next_node,
            your_color,
            stockfish_engine,
            maia_engine,
            current_depth + 1,
            side_line_depth,
            base_score=base_score
        )
        board.pop()

    else:
        try:
            maia_info = get_maia_analysis(maia_engine, board)
            if 'pv' not in maia_info or not maia_info['pv']:
                end_variation(board, node, your_color, stockfish_engine, maia_engine)
                return
            maia_move = maia_info['pv'][0]
            if maia_move not in board.legal_moves:
                end_variation(board, node, your_color, stockfish_engine, maia_engine)
                return

            board.push(maia_move)
            next_node = node.add_main_variation(maia_move)

            generate_variation(
                board,
                next_node,
                your_color,
                stockfish_engine,
                maia_engine,
                current_depth + 1,
                side_line_depth,
                base_score=base_score
            )
            board.pop()
        except Exception as e:
            print(f"Maia 引擎分析錯誤：{e}")
            end_variation(board, node, your_color, stockfish_engine, maia_engine)

# ==============================
# 產生「主變化」的遞迴函式
# ==============================
@measure_time
def generate_opening_preparation(board: chess.Board,
                                 node: chess.pgn.GameNode,
                                 depth: int,
                                 current_depth: int,
                                 your_color: str,
                                 stockfish_engine: chess.engine.SimpleEngine,
                                 maia_engine: chess.engine.SimpleEngine,
                                 base_score: int = None
                                 ):
    """
    遞迴產生開局準備的主線和支線。
    """
    pov_is_white = (your_color == 'white')

    if base_score is None:
        info_list = get_stockfish_analysis(stockfish_engine, board, depth=stockfish_depth_main, multipv=1)
        base_score = evaluate_score_from_info(info_list[0], board, pov_is_white) if info_list else 0

    if current_depth >= depth or board.is_game_over():
        end_variation(board, node, your_color, stockfish_engine, maia_engine)
        return

    turn_is_white = board.turn
    is_our_turn = (turn_is_white and pov_is_white) or (not turn_is_white and not pov_is_white)

    if is_our_turn:
        move = select_best_move_for_us(stockfish_engine, maia_engine, board, your_color, is_main_line=True)
        if move is None:
            end_variation(board, node, your_color, stockfish_engine, maia_engine)
            return

        board.push(move)
        next_node = node.add_main_variation(move)

        generate_opening_preparation(
            board,
            next_node,
            depth,
            current_depth + 1,
            your_color,
            stockfish_engine,
            maia_engine,
            base_score=base_score
        )
        board.pop()

    else:
        expected_infos = get_stockfish_analysis(stockfish_engine, board, depth=stockfish_depth_main, multipv=1)
        if not expected_infos:
            end_variation(board, node, your_color, stockfish_engine, maia_engine)
            return

        # 考慮 Stockfish 推薦的前 N 步
        for i, expected_info in enumerate(expected_infos):
            if 'pv' not in expected_info or not expected_info['pv']:
                continue
            expected_move = expected_info['pv'][0]
            expected_score = evaluate_score_from_info(expected_info, board, pov_is_white)

            try:
                maia_info = get_maia_analysis(maia_engine, board)
                maia_move = maia_info['pv'][0] if 'pv' in maia_info and maia_info['pv'] else None
            except Exception as e:
                print(f"Maia 引擎分析錯誤：{e}")
                maia_move = None

            # 如果 Maia 的走法在 Stockfish 的推薦中，則走主線
            if maia_move == expected_move:
                board.push(expected_move)
                main_next_node = node.add_main_variation(expected_move)
                generate_opening_preparation(
                    board,
                    main_next_node,
                    depth,
                    current_depth + 1,
                    your_color,
                    stockfish_engine,
                    maia_engine,
                    base_score=base_score
                )
                board.pop()
                break # 優先走 Maia 預測的主線

            # 如果 Maia 的走法不在 Stockfish 預期內，則產生支線
            elif i == 0 and maia_move is not None: # 只針對 Maia 的最佳預測產生支線
                board.push(maia_move)
                maia_score_list = get_stockfish_analysis(stockfish_engine, board, depth=stockfish_depth_variation, multipv=1)
                maia_score = evaluate_score_from_info(maia_score_list[0], board, pov_is_white) if maia_score_list else expected_score
                board.pop()

                score_diff = expected_score - maia_score
                if abs(score_diff) >= branching_score_diff_threshold:
                    board.push(expected_move)
                    main_next_node = node.add_main_variation(expected_move)
                    generate_opening_preparation(
                        board,
                        main_next_node,
                        depth,
                        current_depth + 1,
                        your_color,
                        stockfish_engine,
                        maia_engine,
                        base_score=base_score
                    )
                    board.pop()

                    side_line_max_depth = depth * side_line_depth_multiplier
                    board.push(maia_move)
                    variation_node = node.add_variation(maia_move)
                    generate_variation(
                        board,
                        variation_node,
                        your_color,
                        stockfish_engine,
                        maia_engine,
                        current_depth + 1,
                        side_line_max_depth,
                        base_score=base_score
                    )
                    board.pop()
                else: # 如果分差不大，可能只是不同的走法順序，仍然走主線
                    board.push(expected_move)
                    main_next_node = node.add_main_variation(expected_move)
                    generate_opening_preparation(
                        board,
                        main_next_node,
                        depth,
                        current_depth + 1,
                        your_color,
                        stockfish_engine,
                        maia_engine,
                        base_score=base_score
                    )
                    board.pop()
                break # 產生支線後跳出迴圈
            elif i == 0: # 如果 Maia 無法分析，則直接走 Stockfish 最佳預測
                 board.push(expected_move)
                 main_next_node = node.add_main_variation(expected_move)
                 generate_opening_preparation(
                     board,
                     main_next_node,
                     depth,
                     current_depth + 1,
                     your_color,
                     stockfish_engine,
                     maia_engine,
                     base_score=base_score
                 )
                 board.pop()
                 break

# ==============================
# 主程式入口
# ==============================
if __name__ == "__main__":
    print("請輸入局面狀態。範例：")
    print("  startpos moves e2e4 e7e5")
    print("或")
    print("  fen <FEN字串> moves <連續走法>")
    position_input = input("您的輸入：")

    your_color = input("請輸入您的顏色(white 或 black)：").strip().lower()
    while your_color not in ["white", "black"]:
        print("您輸入的顏色無效，請重新輸入：")
        your_color = input("white 或 black：").strip().lower()

    # 主線最大深度
    main_line_depth = int(input("請輸入主變化深度(正整數)："))

    opponent_elo = int(input("請輸入你要模擬的的等級分(1000-3000)："))

    check_engine_path(stockfish_path)
    check_engine_path(lc0_path)
    maia_model_path = get_maia_model_path(opponent_elo)
    check_model_path(maia_model_path)

    # 啟動引擎
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    lc0_command = [lc0_path, f'--weights={maia_model_path}', '--multipv=1']
    maia_engine = chess.engine.SimpleEngine.popen_uci(lc0_command)

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "White"
    game.headers["Black"] = "Black"
    node = parse_position_input(board, game, position_input)

    # 產生主線(深度 = main_line_depth)，同時支線可至 2*main_line_depth
    generate_opening_preparation(
        board,
        node,
        main_line_depth,    # 主線最大深度
        current_depth=0,
        your_color=your_color,
        stockfish_engine=stockfish_engine,
        maia_engine=maia_engine
    )

    # 輸出 PGN
    pgn_exporter = chess.pgn.StringExporter(headers=False, variations=True, comments=False)
    pgn_string = game.accept(pgn_exporter)

    print("\n===== Tricky 開局準備 (PGN 格式) =====")
    print(pgn_string)

    stockfish_engine.quit()
    maia_engine.quit()
    print("\n程式執行完畢！")