import argparse
import random
import time

from functional import Functional
from mnist_data import load_mnist
from optimizer import Optimizer
from perceptron import MLP
from value import Value
from weights_json import save_mlp_weights_json


def predict_class(logits: list[Value]) -> int:
    return max(range(len(logits)), key=lambda i: logits[i].data)


def accuracy(model: MLP, x: list[list[float]], y: list[int], n_eval: int) -> float:
    n = min(n_eval, len(x))
    correct = 0
    for i in range(n):
        xi = [Value(v) for v in x[i]]
        logits = model(xi)
        if predict_class(logits) == y[i]:
            correct += 1
    return correct / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--train-limit", type=int, default=60000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--export-weights-json", type=str, default="")
    args = p.parse_args()

    random.seed(args.seed)

    (x_train, y_train), (x_test, y_test) = load_mnist("mnist.npz")

    model = MLP()
    opt = Optimizer(model.parameters(), lr=args.lr)

    n_train = min(args.train_limit, len(x_train))
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        for i in range(n_train):
            xi = [Value(v) for v in x_train[i]]
            yi = y_train[i]

            logits = model(xi)
            loss = Functional.softmax_cross_entropy(logits, yi)

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if args.eval_every > 0 and step % args.eval_every == 0:
                train_acc = accuracy(model, x_train, y_train, args.eval_n)
                test_acc = accuracy(model, x_test, y_test, args.eval_n)
                dt = time.time() - t0
                print(
                    f"step={step} epoch={epoch+1} i={i+1} "
                    f"loss={loss.data:.4f} "
                    f"train_acc@{args.eval_n}={train_acc:.3f} "
                    f"test_acc@{args.eval_n}={test_acc:.3f} "
                    f"time={dt:.1f}s"
                )

    train_acc = accuracy(model, x_train, y_train, 10000)
    test_acc = accuracy(model, x_test, y_test, 10000)
    print(f"final train_acc@10000={train_acc:.3f} test_acc@10000={test_acc:.3f}")

    if args.export_weights_json:
        save_mlp_weights_json(model, args.export_weights_json)
        print(f"saved weights to {args.export_weights_json}")


if __name__ == "__main__":
    main()
