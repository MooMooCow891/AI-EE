from PIL import Image

def get_pixel(path: str) -> list: # type: ignore
    img = Image.open(path)
    img = img.resize((50, 50), Image.LANCZOS).convert('L')
    return list(img.getdata())

if __name__ == "__main__":
    get_pixel("Training Data\\Test Dataset\\001_156.png")