def to_screen(pos, width, height):
    x, y = pos
    return int(x + width // 2), int(height // 2 - y)

def from_screen(x, y, width, height):
    return x - width // 2, height // 2 - y
