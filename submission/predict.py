"""加载已训练模型，生成验证集结果。"""
from pipeline import ROOT, predict

if __name__ == "__main__":
    predict(ROOT/"验证集结果.csv")
