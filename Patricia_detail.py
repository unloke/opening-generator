import chess
import chess.engine
import chess.pgn
import os
import time
import requests
from functools import wraps
from typing import Dict, List, Tuple

# === 參數設定 ===
stockfish_path   = r'D:\python\my chess game code\stockfish\stockfish-windows-x86-64-avx2.exe'
lc0_path         = r'D:\吳冠頡\ChessGPT\lc0\lc0.exe'  # 只給 Maia 用
model_directory  = r'D:\吳冠頡\opening generator\weights'
patricia_path    = r'D:\吳冠頡\opening generator\patricia_4_v3.exe'

stockfish_depth_main      = 20
stockfish_depth_variation = 20
multipv_variation        = 1

advantage_threshold_base    = 150
candidate_move_tolerance    = 20
move_selection_iterations   = 0
branching_probability_threshold = 0.0
LICHESS_GAME_THRESHOLD = 1000

position_cache: Dict[str, List[chess.Move]] = {}

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
    if inp.startswith('startpos moves'):
        for uci in inp.split()[2:]:
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
        print("錯誤：無效的局面輸入格式")
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
    return info.get('score').pov(pov_is_white).score(mate_score=100000)

def has_clear_advantage(sf_engine, board, pov_white, base_score):
    infos = get_stockfish_analysis(sf_engine, board, depth=stockfish_depth_variation, multipv=1)
    if not infos: return False
    cur = evaluate_score_from_info(infos[0], board, pov_white)
    return (cur - base_score) >= advantage_threshold_base

# === 主要改動：改用 Patricia 的最佳走法 ===
def select_best_move_for_us(
    stockfish_engine: chess.engine.SimpleEngine,
    maia_engine:     chess.engine.SimpleEngine,
    patricia_engine: chess.engine.SimpleEngine,
    board:           chess.Board,
    your_color_is_white: bool,
    is_main_line:    bool
) -> chess.Move | None:
    # 直接請 Patricia 建議「最佳走法」
    res = patricia_engine.play(board, chess.engine.Limit(depth=stockfish_depth_main))
    return res.move

# === 其餘變異生成邏輯不變 ===
def end_variation_if_our_turn(board, node, pov_white, sf, maia, pat):
    if (board.turn==chess.WHITE)==pov_white and not board.is_game_over():
        mv = select_best_move_for_us(sf, maia, pat, board, pov_white, False)
        if mv:
            board.push(mv)
            node.add_main_variation(mv)
            board.pop()

@measure_time
def generate_variation(board, node, ply, max_ply, pov_white, sf, maia, pat, base_score):
    if board.is_game_over(): return
    if ply>=max_ply or has_clear_advantage(sf,board,pov_white,base_score):
        end_variation_if_our_turn(board,node,pov_white,sf,maia,pat)
        return

    if (board.turn==chess.WHITE)==pov_white:
        mv = select_best_move_for_us(sf,maia,pat,board,pov_white,False)
        if mv:
            board.push(mv)
            nxt = node.add_main_variation(mv)
            generate_variation(board,nxt,ply+1,max_ply,pov_white,sf,maia,pat,base_score)
            board.pop()
    else:
        # 對手用 Stockfish + Lichess Explorer 分支
        engine_mv = get_stockfish_analysis(sf,board,stockfish_depth_variation,1)[0]['pv'][0]
        moves_data = get_lichess_top_moves(board.fen(),10)
        if engine_mv and engine_mv in board.legal_moves:
            board.push(engine_mv)
            var = node.add_main_variation(engine_mv)
            generate_variation(board,var,ply+1,max_ply,pov_white,sf,maia,pat,base_score)
            board.pop()

        fen = board.fen()
        cache_moves = position_cache.get(fen)
        if cache_moves is None:
            moves_list, total_games = moves_data
            cache_moves = []
            if moves_list and total_games >= LICHESS_GAME_THRESHOLD:
                for md in moves_list:
                    try:
                        mv = chess.Move.from_uci(md['uci'])
                    except ValueError:
                        continue
                    if mv in board.legal_moves and mv != engine_mv:
                        cache_moves.append(mv)
            position_cache[fen] = sorted(cache_moves, key=lambda m: m.uci())

        for mv in position_cache[fen]:
            board.push(mv)
            var = node.add_variation(mv)
            generate_variation(board,var,ply+1,max_ply,pov_white,sf,maia,pat,base_score)
            board.pop()

@measure_time
def generate_opening_preparation(board, node, ply, max_ply, pov_white, sf, maia, pat, init_score):
    if board.is_game_over(): return
    if ply>=max_ply or has_clear_advantage(sf,board,pov_white,init_score):
        end_variation_if_our_turn(board,node,pov_white,sf,maia,pat)
        return

    if (board.turn==chess.WHITE)==pov_white:
        my_mv = select_best_move_for_us(sf,maia,pat,board,pov_white,True)
        if my_mv:
            board.push(my_mv)
            nxt = node.add_main_variation(my_mv)
            generate_opening_preparation(board,nxt,ply+1,max_ply,pov_white,sf,maia,pat,init_score)
            board.pop()
    else:
        engine_mv = get_stockfish_analysis(sf,board,stockfish_depth_variation,1)[0]['pv'][0]
        moves_data = get_lichess_top_moves(board.fen(),10)
        score_here = evaluate_score_from_info(get_stockfish_analysis(sf,board,stockfish_depth_variation,1)[0], board, pov_white)
        if engine_mv and engine_mv in board.legal_moves:
            board.push(engine_mv)
            nxt = node.add_main_variation(engine_mv)
            generate_opening_preparation(board,nxt,ply+1,max_ply,pov_white,sf,maia,pat,init_score)
            board.pop()
        fen = board.fen()
        cache_moves = position_cache.get(fen)
        if cache_moves is None:
            moves_list, total_games = moves_data
            cache_moves = []
            if moves_list and total_games >= LICHESS_GAME_THRESHOLD:
                for md in moves_list:
                    try:
                        mv = chess.Move.from_uci(md['uci'])
                    except ValueError:
                        continue
                    if mv in board.legal_moves and mv != engine_mv:
                        cache_moves.append(mv)
            position_cache[fen] = sorted(cache_moves, key=lambda m: m.uci())

        for mv in position_cache[fen]:
            board.push(mv)
            var = node.add_variation(mv)
            generate_variation(board,var,ply+1,max_ply,pov_white,sf,maia,pat,score_here)
            board.pop()

def get_lichess_top_moves(fen: str, num: int = 10) -> Tuple[List[Dict], int]:
    params = {
        'fen':fen, 'variant':'standard',
        'speeds':'blitz,rapid,classical',
        'ratings':'2000,2200,2500',
        'topGames':0, 'recentGames':0
    }
    r = requests.get('https://explorer.lichess.ovh/lichess', params=params, timeout=10)
    data = r.json()
    total = data.get('white',0)+data.get('draws',0)+data.get('black',0)
    moves = data.get('moves',[])
    if total==0:
        total = sum(m.get('white',0)+m.get('draws',0)+m.get('black',0) for m in moves)
    for m in moves:
        games = m.get('white',0)+m.get('draws',0)+m.get('black',0)
        m['probability'] = games/total if total else 0
    return sorted(moves, key=lambda x:x['probability'], reverse=True)[:num], total

# === 主程式入口 ===
if __name__ == "__main__":
    print("========= Tricky 開局準備產生器 (Patricia 版) =========")
    pos_in = input("局面輸入：").strip()
    col = ""
    while col not in ("white","black"):
        col = input("顏色 (white/black)：").strip().lower()
    pov_white = (col=="white")
    depth_ply = 0
    while depth_ply<=0:
        depth_ply = int(input("主準備深度 (ply)："))
    opponent_elo = 0
    while not (1000<=opponent_elo<=3000):
        opponent_elo = int(input("Maia 等級分 (1000-3000)："))

    # 檢查執行檔與模型
    check_engine_path(stockfish_path, "Stockfish")
    check_engine_path(lc0_path,      "Lc0")
    check_engine_path(patricia_path, "Patricia")
    maia_model = get_maia_model_path(opponent_elo)
    check_model_path(maia_model, "Maia 模型")

    # 啟動各引擎
    pat_cmd = [patricia_path]
    lc0_maia_cmd = [lc0_path, f'--weights={maia_model}']

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf,\
         chess.engine.SimpleEngine.popen_uci(lc0_maia_cmd)   as maia,\
         chess.engine.SimpleEngine.popen_uci(pat_cmd)        as pat:

        sf.configure({"UCI_ShowWDL": True, "Threads": os.cpu_count()//2 or 1})
        board = chess.Board()
        game  = chess.pgn.Game()
        node  = parse_position_input(board, game, pos_in)

        init_info = get_stockfish_analysis(sf, board, stockfish_depth_main, 1)
        base_score = evaluate_score_from_info(init_info[0], board, pov_white) if init_info else 0

        generate_opening_preparation(
            board, node, board.ply(), board.ply()+depth_ply,
            pov_white, sf, maia, pat, base_score
        )

        pgn = game.accept(chess.pgn.StringExporter(headers=True, variations=True, comments=False))
        print("\n===== PGN =====\n")
        print(pgn)
