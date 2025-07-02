from maia2 import model, inference
import chess

# 1. 下載並載入 Maia-2 的 rapid 權重（自動下載）
maia2_model = model.from_pretrained(type="rapid", device="gpu")

# 2. 測試用 FEN
fen = "rnbqkbnr/p1pp1ppp/8/1p2p3/5PP1/8/PPPPP1BP/RNBQK1NR b KQkq - 1 3"
board = chess.Board(fen)

# 3. 指定玩家與對手 Elo
elo_self = 2200  # 假設自己 1500 分
elo_oppo = 2500  # 假設對手 1500 分

# 4. 做推理
prepared = inference.prepare()
move_probs, win_prob = inference.inference_each(maia2_model, prepared, fen, elo_self, elo_oppo)

# 5. 印出所有可能走法的機率分佈
print(f"FEN: {fen}")
print(f"勝率預測: {win_prob:.2%}")
print("各走法機率分佈：")
for move, prob in sorted(move_probs.items(), key=lambda x: -x[1]):
    print(f"{move}: {prob:.2%}")

# 6. 最推薦走法
best_move = max(move_probs, key=move_probs.get)
print(f"\nMaia-2 最推薦走法：{best_move}")
