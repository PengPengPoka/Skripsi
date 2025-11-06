import cv2 as cv
from pathlib import Path

def Counter(path: str):
    images = list(path.glob('*.jpg'))
    
    print(f'tea images in {path} directory: {len(images)}')

def main():
    DIR = Path('Tea Score Image Data').resolve()
    TEA_SCORES = ['Score 1', 'Score 2', 'Score 3', 'Score 4']
    
    for score in TEA_SCORES:
        score_path = DIR / score
        Counter(score_path)
    
if __name__ == "__main__":
    main()