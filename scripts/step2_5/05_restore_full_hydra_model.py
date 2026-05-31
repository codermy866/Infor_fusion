#!/usr/bin/env python3
from common import restore_full_hydra_model


def main():
    path = restore_full_hydra_model()
    print(path)


if __name__ == "__main__":
    main()
