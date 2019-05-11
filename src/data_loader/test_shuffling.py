

x = 5
y = 10
def test(shuffle=True):
    global x, y
    if shuffle:
        x += 1
        yield x
    else:
        y += 1
        yield y


