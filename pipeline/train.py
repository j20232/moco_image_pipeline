import argparse
from bengali_module import BengaliModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please set the index of the input config file")
    parser.add_argument("index", help="the index of the input config file")
    index = parser.parse_args().index
    assert len(index) == 4
    bengali = BengaliModule(index)
    print(bengali)
