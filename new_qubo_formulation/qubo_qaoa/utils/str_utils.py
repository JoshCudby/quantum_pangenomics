from typing import Generator


def genbin(n, bs='') -> Generator[str, None, None]:
    if len(bs) == n:
        yield bs
    else:
        yield from genbin(n, bs + '0')
        yield from genbin(n, bs + '1')