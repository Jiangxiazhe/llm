# 下载TinyStories数据集
import os
import requests
import tarfile

def download_tiny_stories():
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true"
    response = requests.get(url)
    with open("TinyStories-train.txt", "wb") as f:
        f.write(response.content)

def main():
    if os.path.exists("./TinyStories-train.txt"):
        print("数据集已存在")
    else:
        download_tiny_stories()

if __name__ == "__main__":
    main()