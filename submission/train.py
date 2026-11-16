"""从随包原表重建、训练并输出验证集结果。"""
import json
from pipeline import CONFIG, RAW, train

if __name__ == "__main__":
    train(RAW,json.loads(CONFIG.read_text("utf8")))
