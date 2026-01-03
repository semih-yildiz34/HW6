from common import load_and_prepare_mnist

x_train, y_train, x_test, y_test = load_and_prepare_mnist()

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)
