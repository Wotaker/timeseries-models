import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser, ArgumentError


DEFAULT_PATH = os.path.join(os.path.curdir, "MackeyGlass.csv")


if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--n', default=10000, type=int)
    parser.add_argument('--b', default=0.1, type=float)
    parser.add_argument('--c', default=0.2, type=float)
    parser.add_argument('--tau', default=25, type=int)
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--path', default=DEFAULT_PATH, type=str)
    args = parser.parse_args()

    # Get Mackey-Glass equation parameters
    N   = args.n
    b   = args.b
    c   = args.c
    tau = args.tau

    # Initial conditions
    y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
        1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]
    
    # Generate Data
    for n in range(17,N+99):
        y.append(y[n] - b*y[n] + c*y[n-tau]/(1+y[n-tau]**10))
    y = np.array(y[100:])

    # Plot generated
    if args.plot:
        plt.plot(y[:1000])
        plt.ylim(0, 2)
        plt.show()
    
    # Save to csv
    df = pd.DataFrame({"Total": y})
    df.to_csv(args.path)
    print(f"New Mackey-Glass dataset saved in {args.path}")
