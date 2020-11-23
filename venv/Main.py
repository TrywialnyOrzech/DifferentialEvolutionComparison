from De import ClassicDE


def main():
    de1 = ClassicDE()   # starting parameters given here
    result = list(de1.de())
    print(result)
    print("End.")

if __name__ == '__main__':
    main()