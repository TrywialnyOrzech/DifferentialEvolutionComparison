from De import ClassicDE
import matplotlib.pyplot as plt


def main():
    de1 = ClassicDE()   # starting parameters given here
    result = list(de1.de())
    print(result[-1])
    print("End.")

if __name__ == '__main__':
    main()